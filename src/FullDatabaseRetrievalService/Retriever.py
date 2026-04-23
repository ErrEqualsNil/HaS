from .DenseRetriever import DenseRetriever
from utils.logger import log_function_call, logger
from typing import Union


class Retriever:
    @log_function_call(logger)
    def __init__(self, embedding_dir, ids_dir, model_name):
        self.embedding_dir = embedding_dir
        self.ids_dir = ids_dir
        self.model_name = model_name
        self.retriever: Union[DenseRetriever, None] = None

    def reset(self):
        self.retriever = DenseRetriever(self.embedding_dir, self.ids_dir, self.model_name)

    def search(self, query, n_docs, require_doc_embeddings):
        # Dense Retriever returns Doc Embeddings; a Sparse Retriever would return Doc Strings.
        assert self.retriever is not None
        docs, infos = self.retriever.search_document(query, n_docs, require_doc_embeddings)
        return docs, infos
