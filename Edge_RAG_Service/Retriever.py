import os
from collections import defaultdict, deque, Counter
from typing import List, Dict

import numpy as np
import requests

from utils.encoder import Encoder
from utils.index import HNSWLibIndexerV2
from utils.load_save_file import load_jsonl_file
from utils.logger import logger, log_function_call

os.environ["TOKENIZERS_PARALLELISM"] = "true"

cloud_rag_url = "http://127.0.0.1:6000/search"


def simulate_retrieve_delay():
    delay = np.random.normal(loc=0.01713, scale=0.005671)
    return delay


def merge_doc_list(d_list_1: List, d_list_2: List) -> List:
    res = []
    seen_doc_ids = set()
    for d in d_list_1 + d_list_2:
        if d["id"] not in seen_doc_ids:
            res.append(d)
            seen_doc_ids.add(d["id"])
    return res


class Retriever:
    def __init__(self, model_name, precomputed_random_corpus_result_dir):
        self.encoder = Encoder(model_name)
        self.fuzzy_source = PrecomputedFuzzySourceRetriever(precomputed_random_corpus_result_dir)
        self.cache_source = CacheSource()

    def reset(self, cache_max_size=5000):
        self.cache_source.reset(cache_max_size)

    def search(self, query_info: dict, n_docs=10, threshold=0.2):
        query = query_info["question"]
        query_embedding = self.encoder.encode(query)
        docs_from_fuzzy_source = self.fuzzy_source.search(query, n_docs)
        fuzzy_source_retrieve_delay = simulate_retrieve_delay()  # simulated delay should be returned, actual retrieval delay (cache source) will be recorded into the end-to-end latency on user clients

        docs_from_cache_source = self.cache_source.search(query_embedding, top_n=n_docs)
        draft = sorted(merge_doc_list(docs_from_cache_source, docs_from_fuzzy_source), key=lambda doc: doc["score"], reverse=True)[:n_docs]

        most_potential_homologous_query_info, score = self.cache_source.is_homologous_query_exist([doc["id"] for doc in draft], threshold)
        if most_potential_homologous_query_info is None:
            # draft is rejected, call full-corpus retrieval from the cloud
            resp = requests.post(cloud_rag_url, json={"query_str": query, "n_docs": n_docs, "require_embeddings": True})
            resp = resp.json()
            doc_ids = [doc["id"] for doc in resp.get("docs")]
            doc_embeddings = resp.get("doc_embeddings")
            self.cache_source.add(query_info, np.array(doc_ids), np.array(doc_embeddings))
            return doc_ids, False, None, resp.get("simulate_delay") + fuzzy_source_retrieve_delay, 0
        return [doc["id"] for doc in draft], True, most_potential_homologous_query_info, fuzzy_source_retrieve_delay, score


class PrecomputedFuzzySourceRetriever:
    @log_function_call(logger)
    def __init__(self, precomputed_cache_dir):
        self.query_to_docs = load_jsonl_file(precomputed_cache_dir)
        self.query_to_docs = {r["question"]: r["docs"] for r in self.query_to_docs}

    def search(self, query_str: str, n_docs: int):
        assert query_str is not None
        docs = self.query_to_docs[query_str][:n_docs]
        return docs


class CacheSource:
    def __init__(self):
        self.cache_max_size = 0
        self.index = HNSWLibIndexerV2()

        self.fifo_queue = deque()
        self.query_id_2_doc_id = defaultdict(list)
        self.query_id_2_query = {}
        self.doc_id_2_query_id = defaultdict(list)

    def reset(self, cache_max_size):
        self.cache_max_size = cache_max_size
        self.index.reset(expected_max_element=self.cache_max_size * 15)  # 10 documents per query + buffer to prevent index overflow

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

    def is_homologous_query_exist(self, doc_ids, threshold):

        # use inverted index to find cached queries from draft documents
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

