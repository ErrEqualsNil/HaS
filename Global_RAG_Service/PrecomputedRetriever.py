import os

import numpy as np

from utils.load_save_file import load_jsonl_file, write_jsonl
from utils.logger import log_function_call, logger


class PrecomputedRetriever:
    @log_function_call(logger)
    def __init__(self, embeddings_dir, precomputed_path):
        self.query_to_docs = load_jsonl_file(precomputed_path)
        self.query_to_docs = {r["question"]: r["docs"] for r in self.query_to_docs}
        self.embeddings = np.load(embeddings_dir, mmap_mode="r")

    def search_document(
            self,
            query: str,
            n_docs: int,
            require_docs: bool = False,
    ):
        assert query is not None
        docs = self.query_to_docs[query][:n_docs]
        indexes = [d["id"] for d in docs]
        doc_embeddings = None
        if require_docs:
            doc_embeddings = self.embeddings[indexes].tolist()
        return docs, doc_embeddings

