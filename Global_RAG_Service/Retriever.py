from Global_RAG_Service import FullCorpusRetriever, PrecomputedRetriever
from utils.logger import log_function_call, logger
from typing import Union


class Retriever:
    @log_function_call(logger)
    def __init__(self):
        self.precomputed_path = "data_sample_100.jsonl"
        self.embedding_dir = "data/enwiki_20231101/samples/100/corpus_embeddings.npy"
        self.ids_dir = "data/enwiki_20231101/samples/100/corpus_ids.npy"
        self.model_name = "nthakur/contriever-base-msmarco"
        self.retriever: Union[FullCorpusRetriever, PrecomputedRetriever, None] = None

    def reset(self, use_precomputed_retriever=True):
        if use_precomputed_retriever:
            self.retriever = PrecomputedRetriever.PrecomputedRetriever(self.embedding_dir, self.precomputed_path)
        else:
            self.retriever = FullCorpusRetriever.DenseRetriever(self.embedding_dir, self.ids_dir, self.model_name)

    def search(self, query, n_docs, require_infos):
        docs, infos = self.retriever.search_document(query, n_docs, require_infos)
        return docs, infos
