import argparse
import copy
import os
import random
from datetime import datetime

import numpy as np
import requests
from tqdm import tqdm

from utils.load_save_file import load_jsonl_file, write_jsonl

url_cloud = "http://127.0.0.1:6000"
url_edge = "http://127.0.0.1:5999"

random.seed(42)
np.random.seed(42)


def generate_filename(arguments):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{arguments.save_prefix}_{arguments.threshold}_{timestamp}.jsonl"
    return filename


def send_request(query_info, arguments):
    packet = {
        "query_info": query_info,
        "n_docs": arguments.n_docs,
        "threshold": arguments.threshold,
    }

    resp = requests.post(f"{url_edge}/search", json=packet)
    real_time_cost = resp.elapsed.total_seconds()  # end-to-end response latency
    resp = resp.json()
    doc_ids, is_cache_match, similar_query_info, simulate_delay, similar_score = (
        resp.get("doc_ids"), resp.get("is_cache_match"),
        resp.get("similar_query_info"), resp.get("simulate_delay"),
        resp.get("similar_score")
    )

    return real_time_cost + simulate_delay, doc_ids, is_cache_match, similar_query_info, similar_score


def batch_request(query_file, output_file_name, arguments):
    records = []
    queries = load_jsonl_file(query_file)
    random.shuffle(queries)

    for q in tqdm(queries):
        time_cost, doc_ids, is_cache_match, similar_query_info, similar_score = send_request(q, arguments)
        record = copy.deepcopy(q)
        record["rag_time_cost"] = time_cost
        record["doc_ids"] = doc_ids
        record["is_cache_match"] = is_cache_match
        record["similar_query_info"] = similar_query_info
        record["similar_score"] = similar_score
        records.append(record)
    write_jsonl(records, os.path.join("/home/pp/noise_in_rag/data/own_dataset/repeat_test_1/run_record", output_file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_docs', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--save_prefix', type=str, default="")
    args = parser.parse_args()

    file_name = generate_filename(args)

    requests.post(f"{url_edge}/reset")  # reset the cache
    batch_request("query.jsonl", file_name, args)
