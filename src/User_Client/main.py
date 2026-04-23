import argparse
import copy
import os
import random
from datetime import datetime

import numpy as np
import requests
from tqdm import tqdm

from utils.load_save_file import load_jsonl_file, write_jsonl

url_edge = "http://127.0.0.1:5999"

random.seed(42)
np.random.seed(42)


def send_request(query_info, arguments):
    packet = {
            "query_info": query_info,
            "n_docs": arguments.n_docs,
        }
    resp = requests.post(f"{url_edge}/search", json=packet)
    real_time_cost = resp.elapsed.total_seconds()
    resp = resp.json()
    doc_ids, is_speculative_retrieval_match, similar_query_info, simulated_network_latency, similar_score = (
        resp.get("doc_ids"), resp.get("is_speculative_retrieval_match"),
        resp.get("similar_query_info"), resp.get("simulated_network_latency"),
        resp.get("similar_score")
    )
    return real_time_cost + simulated_network_latency, doc_ids, is_speculative_retrieval_match, similar_query_info, similar_score


def batch_request(query_file, output_file_name, args):
    records = []
    queries = load_jsonl_file(query_file)
    random.shuffle(queries)

    for q in tqdm(queries):
        time_cost, doc_ids, is_speculative_retrieval_match, similar_query_info, similar_score = send_request(q, args)
        record = copy.deepcopy(q)
        record["rag_time_cost"] = time_cost
        record["doc_ids"] = doc_ids
        record["is_speculative_retrieval_match"] = is_speculative_retrieval_match
        record["similar_query_info"] = similar_query_info
        record["similar_score"] = similar_score
        records.append(record)
    write_jsonl(records, os.path.join("data/popqa_augmented/run_record", output_file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_docs', type=int, default=10)
    parser.add_argument('--prefix', type=str, default="DocSim")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{args.prefix}_{timestamp}.jsonl"
    os.makedirs("data/popqa_augmented/run_record", exist_ok=True)
    batch_request("data/popqa_augmented/data.jsonl", filename, args)
