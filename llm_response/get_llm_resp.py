import glob
import hashlib
import json
import os
import pickle
import random
from typing import Dict

import numpy as np
import torch

from llm_response.prompts import build_qa_prompt
from utils.llm import QwenLLM
from utils.load_save_file import load_jsonl_file, write_jsonl


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def get_responses(llm: QwenLLM, data, response_cache_dict: Dict):
    data = sorted(data, key=lambda d: d["id"])
    data_match, data_wo_match = [], []

    for d in data:
        d["prompt"] = build_qa_prompt(d)
        d["hash_value"] = str(stable_hash(d["prompt"]))
        # we use hash value to find historical response for reuse, thus saving LLM inference time
        if d["hash_value"] in response_cache_dict.keys():
            d["resp"] = response_cache_dict[d["hash_value"]]
            data_match.append(d)
        else:
            data_wo_match.append(d)

    if len(data_wo_match) != 0:
        prompts = [d["prompt"] for d in data_wo_match]
        responses = llm.generate(prompts)
        for resp, d in zip(responses, data_wo_match):
            d[f"resp"] = resp
            response_cache_dict[d["hash_value"]] = resp

    data = sorted(data_match + data_wo_match, key=lambda d: d["id"])
    return data, response_cache_dict


def read_jsonl_by_line_ids(data_path, line_offsets, line_ids):
    results = {}
    with open(data_path, 'rb') as f:
        for line_id in sorted(line_ids):
            if 0 <= line_id < len(line_offsets):
                f.seek(line_offsets[line_id])
                line = f.readline()
                obj = json.loads(line.decode('utf-8'))
                assert obj["id"] == line_id
                results[line_id] = obj
    return results


def normalize_answer_list(strings: list[str]) -> list[str]:
    strings = [string.lower() for string in strings]
    strings = sorted(set(strings), key=len)
    result = []

    for s in strings:
        if any(ele in s for ele in result):
            continue
        result.append(s)

    return result


if __name__ == '__main__':
    corpus_path = "enwiki_20231101/corpus.jsonl"  # path to the full corpus
    line_offset_path = "corpus_line_offset.pkl"  # help to fast search the document from the jsonl corpus file

    llm = QwenLLM()
    llm_resp_cache_path = "llm_resp_cache.json" # we put historical llm input into this file, Dict{hash_value: llm response}
    record_root = "run_record"  # read all files in the record folder
    save_root = "/home/pp/noise_in_rag/data/own_dataset/repeat_test_1/record_after_qwen3"  # save file to this folder
    for file_name in sorted(glob.glob(os.path.join(record_root, "*"))):  # you can use regex to select some record for LLM response
        file_name = os.path.split(file_name)[1]
        try:
            with open(llm_resp_cache_path, "r", encoding="utf-8") as f:
                response_cache_dict = json.load(f)
        except FileNotFoundError:
            response_cache_dict = {}

        with open(line_offset_path, 'rb') as f:
            line_offsets = pickle.load(f)

        # find documents from the corpus by document id
        if not os.path.exists(os.path.join(save_root, file_name)):
            data = load_jsonl_file(os.path.join(record_root, file_name))
            doc_ids = []
            for d in data:
                doc_ids += d["doc_ids"]
            doc_ids = list(set(doc_ids))
            doc_id_to_doc = read_jsonl_by_line_ids(corpus_path, line_offsets, doc_ids)
            idx = 0
            for d in data:
                d["id"] = idx
                idx += 1
                d["docs"] = [doc_id_to_doc[doc_id] for doc_id in d["doc_ids"]]
                d["answer_list"] = normalize_answer_list(d["answer_list"])
            write_jsonl(data, os.path.join(save_root, file_name))

        # get llm response
        data = load_jsonl_file(os.path.join(save_root, file_name))
        data, response_cache_dict = get_responses(llm, data, response_cache_dict)
        write_jsonl(data, os.path.join(save_root, file_name))
        with open(llm_resp_cache_path, "w", encoding="utf-8") as f:
            json.dump(response_cache_dict, f)
