import os
from collections import defaultdict, deque, Counter
from typing import List, Dict, Union

import numpy as np
import requests

from utils.encoder import Encoder
from utils.index import FaissIndexer, HNSWLibIndexerV2
from utils.logger import logger, log_function_call

os.environ["TOKENIZERS_PARALLELISM"] = "true"

cloud_rag_url = "http://127.0.0.1:6000/search"


def merge_doc_list(d_list_1: List, d_list_2: List) -> List:
    res = []
    seen_doc_ids = set()
    for d in d_list_1 + d_list_2:
        if d["id"] not in seen_doc_ids:
            res.append(d)
            seen_doc_ids.add(d["id"])
    return res


class HaS:
    def __init__(self, embeddings_dir, ids_dir, model_name, cache_max_size):
        self.encoder = Encoder(model_name)
        self.fuzzy_channel = FuzzyChannel(
            embeddings_dir=embeddings_dir,
            ids_dir=ids_dir,
            model_name=model_name
        )
        self.cache_channel = CacheChannel()
        self.cache_channel.reset(cache_max_size)

    def search(self, query_info: dict, n_docs=10, threshold=0.2):
        query = query_info["question"]
        query_embedding = self.encoder.encode(query)
        docs_from_fuzzy_channel = self.fuzzy_channel.search_document(query, n_docs)

        docs_from_cache_channel = self.cache_channel.search(query_embedding, top_n=n_docs)
        top_related_docs = sorted(merge_doc_list(docs_from_fuzzy_channel, docs_from_cache_channel), key=lambda doc: doc["score"], reverse=True)[:n_docs]
        top_related_doc_ids = [doc["id"] for doc in top_related_docs]

        most_similar_query_info, score = self.cache_channel.is_similar_query_exist([doc["id"] for doc in top_related_docs], threshold)
        if most_similar_query_info is None:
            resp = requests.post(cloud_rag_url, json={"query_str": query, "n_docs": n_docs, "require_doc_embeddings": True})
            resp = resp.json()
            doc_ids = [doc["id"] for doc in resp.get("docs")]
            doc_embeddings = resp.get("doc_embeddings")
            self.cache_channel.add(query_info, np.array(doc_ids), np.array(doc_embeddings))
            return doc_ids, False, None, resp.get("simulated_network_latency"), 0
        return top_related_doc_ids, True, most_similar_query_info, 0, score

class FuzzyChannel:
    @log_function_call(logger)
    def __init__(
            self,
            embeddings_dir,
            ids_dir,
            model_name,
            projection_size=768,
            n_subquantizers=128,
            n_bits=8,
            indexing_batch_size=1000000,
    ):
        self.encoder = Encoder(model_name)

        self.index = FaissIndexer(
            projection_size, n_subquantizers,
              n_bits, useIVF=True
        )

        logger.info(f"Indexing Embeddings {embeddings_dir}")
        embeddings = np.load(embeddings_dir, mmap_mode="r")
        while embeddings.shape[0] > 0:
            logger.info(f"Remaining embeddings to index: {embeddings.shape[0]}")
            end_idx = min(indexing_batch_size, embeddings.shape[0])
            embeddings_to_add = embeddings[:end_idx]
            embeddings = embeddings[end_idx:]
            self.index.index_data(embeddings_to_add)
        del embeddings
        self.embeddings = np.load(embeddings_dir, mmap_mode="r")
        self.ids = np.load(ids_dir)

    @log_function_call(logger)
    def search_document(
            self,
            query: Union[str, List[float]],
            n_docs: int,
    ):
        # Accepts either a str-format query or embeddings pre-computed by the Edge.
        if isinstance(query, str):
            query_embedding = self.encoder.encode(query)
        else:
            query_embedding = np.array(query)
        _, indexes = self.index.search_knn(query_embedding, n_docs * 5)

        doc_embeddings = self.embeddings[indexes]
        scores = np.matmul(query_embedding, doc_embeddings.T)[0]
        re_rank_order = np.argsort(-scores)[:n_docs]

        docs = [{"id": int(self.ids[indexes[idx]]), "score": float(scores[idx])} for idx in re_rank_order]
        return docs



class CacheChannel:
    def __init__(self):
        self.cache_max_size = 0
        self.index = HNSWLibIndexerV2()

        self.fifo_queue = deque()
        self.query_id_2_doc_id = defaultdict(list)
        self.query_id_2_query = {}
        self.doc_id_2_query_id = defaultdict(list)

    def reset(self, cache_max_size=1000):
        self.cache_max_size = cache_max_size
        self.index.reset(expected_max_element=self.cache_max_size * 30)

        self.fifo_queue = deque()
        self.query_id_2_doc_id = defaultdict(list)
        self.query_id_2_query = {}
        self.doc_id_2_query_id = defaultdict(list)

    @log_function_call(logger)
    def add(self, query_info, doc_ids: np.ndarray, doc_embeddings: np.ndarray):
        if len(self.fifo_queue) >= self.cache_max_size:
            self.delete(self.fifo_queue.popleft())

        query_id = query_info["id"]

        self.fifo_queue.append(query_id)
        self.query_id_2_doc_id[query_id] = doc_ids.tolist()
        self.query_id_2_query[query_id] = query_info

        not_exists_indices = []
        for idx, doc_id in enumerate(doc_ids):
            if len(self.doc_id_2_query_id[doc_id]) == 0:
                not_exists_indices.append(idx)
            self.doc_id_2_query_id[doc_id].append(query_id)

        self.index.add_data(doc_embeddings[not_exists_indices], doc_ids[not_exists_indices])
        return

    @log_function_call(logger)
    def delete(self, query_id: int):
        doc_ids = self.query_id_2_doc_id.pop(query_id)
        self.query_id_2_query.pop(query_id)

        doc_ids_to_remove = []
        for doc_id in doc_ids:
            self.doc_id_2_query_id[doc_id].remove(query_id)
            if len(self.doc_id_2_query_id[doc_id]) == 0:
                doc_ids_to_remove.append(doc_id)
        self.index.remove_data(doc_ids_to_remove)

    def search(self, current_query_embedding, top_n=10) -> List[Dict]:
        doc_ids, scores = self.index.search_knn(current_query_embedding, top_n)
        if doc_ids is None:
            return []
        docs = [{"id": int(doc_id), "score": float(score)} for doc_id, score in zip(doc_ids, scores)]
        return docs

    def is_similar_query_exist(self, doc_ids, threshold):
        similar_queries = []
        for doc_id in doc_ids:
            similar_queries += self.doc_id_2_query_id[doc_id]
        if len(similar_queries) == 0:
            return None, None
        most_similar_query_id, score = Counter(similar_queries).most_common(1)[0]
        score = score / len(doc_ids)
        if score >= threshold:
            query_info = self.query_id_2_query[most_similar_query_id]
            return query_info, score
        return None, None

