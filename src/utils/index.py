# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List

import faiss
import hnswlib
import numpy as np

from utils.logger import logger


class FaissIndexer(object):
    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8, useIVF=False):
        if n_subquantizers > 0:
            # Use IndexIVFPQ with IndexFlatIP as coarse quantizer for better recall and ranking quality
            if useIVF:
                quantizer = faiss.IndexFlatIP(vector_sz)
                self.index = faiss.IndexIVFPQ(quantizer, vector_sz, 8192, n_subquantizers, n_bits)
                self.index.nprobe = 64
                self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, embeddings):
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

    def search_knn(self, query_vector: np.array, top_docs: int):
        query_vector = query_vector.reshape((1, -1)).astype('float32')
        scores, indexes = self.index.search(query_vector, top_docs)
        return scores[0], indexes[0]

    def batch_search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048):
        query_vectors = query_vectors.astype('float32')
        scores, indexes = [], []
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        for k in range(nbatch):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            logger.info(f"Batch Search Interval: {start_idx}: {end_idx} / Total {len(query_vectors)}")
            q = query_vectors[start_idx: end_idx]
            batch_scores, batch_indexes = self.index.search(q, top_docs)
            scores.append(batch_scores)
            indexes.append(batch_indexes)
        scores, indexes = np.concatenate(scores, axis=0), np.concatenate(indexes, axis=0)
        return scores, indexes


class HNSWLibIndexerV2(object):
    def __init__(
        self,
        vector_sz=768, ef_construction=200, ef_search=100, M=48, space='ip',
    ):
        self.vector_sz = vector_sz
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        self.space = space

        self.marked_deleted_doc_ids = set()

        self.expected_max_element = 0
        self.index: Optional[hnswlib.Index] = None

    def reset(self, expected_max_element):
        self.expected_max_element = expected_max_element
        self.index = hnswlib.Index(space=self.space, dim=self.vector_sz)
        self.index.init_index(
            max_elements=int(self.expected_max_element * 1.2),
            ef_construction=self.ef_construction,
            M=self.M,
            allow_replace_deleted=True
        )
        self.index.set_ef(self.ef_search)
        self.marked_deleted_doc_ids = set()

    def add_data(self, embeddings: np.ndarray, embd_ids: np.ndarray):
        need_to_add_idx = []
        for idx, (embd_id, embd) in enumerate(zip(embd_ids, embeddings)):
            if embd_id in self.marked_deleted_doc_ids:
                self.marked_deleted_doc_ids.remove(int(embd_id))
                if embd_id in self.index.get_ids_list():
                    self.index.unmark_deleted(int(embd_id))
                    continue
            need_to_add_idx.append(idx)
        embd_ids = embd_ids[need_to_add_idx]
        embeddings = embeddings[need_to_add_idx].astype("float32")
        self.index.add_items(embeddings, ids=embd_ids, replace_deleted=True)

    def remove_data(self, embd_ids: List):
        for idx in embd_ids:
            self.index.mark_deleted(int(idx))
            self.marked_deleted_doc_ids.add(int(idx))

    def search_knn(self, query_embedding: np.array, top_k: int):
        if self.index.get_current_count() == 0:
            return None, None
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        if self.space == "ip":
            return labels[0], 1 - distances[0]
        else:
            return labels[0], np.sqrt(distances[0])