import os
from typing import List, Union

import numpy as np

from utils import index
from utils.encoder import Encoder
from utils.logger import logger, log_function_call

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class DenseRetriever:
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

        self.index = index.FaissIndexer(projection_size, n_subquantizers, n_bits)

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
            require_doc_embeddings: bool = False,
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

        if require_doc_embeddings:
            return docs, doc_embeddings.tolist()
        return docs, None
