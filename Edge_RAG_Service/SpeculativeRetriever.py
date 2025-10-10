import random
from typing import Optional, Union

from Edge_RAG_Service.Retriever import Retriever


def simulate_network_delay(min_delay=0.01, max_delay=0.05):
    delay = random.uniform(min_delay, max_delay)
    return delay


class SpeculativeRetriever:
    def __init__(self):
        self.encoder_name = "nthakur/contriever-base-msmarco"

        self.mode: Optional[str] = None
        self.cache: Union[Retriever, None] = None

    def reset(self):
        self.cache = Retriever(
            model_name=self.encoder_name,
            precomputed_random_corpus_result_dir=f"data_sample_100_IVF.jsonl",
        )
        self.cache.reset()

    def search(self, query_info, n_docs, threshold):
        doc_ids, is_cache_match, similar_query, simulated_delay, similar_score = (
            self.cache.search(query_info, n_docs, threshold)
        )
        return doc_ids, is_cache_match, similar_query, simulated_delay + simulate_network_delay(), similar_score
