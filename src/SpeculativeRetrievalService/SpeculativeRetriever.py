import random
from typing import Optional, Union

from .HaS import HaS


def simulate_network_delay(min_delay=0.01, max_delay=0.05):
    delay = random.uniform(min_delay, max_delay)
    return delay


class SpeculativeRetriever:
    def __init__(self):
        self.encoder_name = "nthakur/contriever-base-msmarco"

        self.mode: Optional[str] = None
        self.HaS: Union[HaS, None] = None

    def reset(self, embeddings_dir, ids_dir, model_name, cache_max_size):
        self.HaS = HaS(embeddings_dir, ids_dir, model_name, cache_max_size)

    def search(self, query_info, n_docs, threshold):
        assert self.HaS is not None
        doc_ids, is_speculative_retrieval_match, similar_query, simulated_network_latency, similar_score = (
            self.HaS.search(query_info, n_docs, threshold)
        )
        return doc_ids, is_speculative_retrieval_match, similar_query, simulated_network_latency + simulate_network_delay(), similar_score
