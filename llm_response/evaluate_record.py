import os.path
import random

from utils import evaluate
from utils.load_save_file import load_jsonl_file
import glob
import re
import pandas as pd


if __name__ == '__main__':
    results = []

    # select files to be evaluated
    pattern = re.compile(r"record_after_qwen/.*")
    all_files = glob.glob("record_after_qwen/*")
    matched_files = [f for f in all_files if pattern.fullmatch(f)]

    for file_name in sorted(matched_files):
        data = load_jsonl_file(file_name)
        data = random.sample(data, int(0.95 * len(data)))

        result = {
            "name": os.path.split(file_name)[1],
            "average_time_cost": evaluate.average_time_cost(data),
            "response_hit_rate": evaluate.response_hit_rate(data, only_cache_hit=False),
            "document_hit_rate": evaluate.document_hit_rate(data, k=10, only_cache_hit=False),
            "cache_hit_rate": evaluate.cache_hit_rate(data),
            "cache_hit_correct_rate": evaluate.cache_hit_correct_rate(data),
            "response_hit_rate_for_cache_hit_record": evaluate.response_hit_rate(data, only_cache_hit=True),
        }
        results.append(result)
    results = pd.DataFrame(results)
    results.to_csv("record_after_qwen/results.csv")
