import glob
import hashlib
import json
import os
import pickle
import random
from typing import Dict

import numpy as np
import torch

from utils.load_save_file import load_jsonl_file, write_jsonl
from utils.batch_llm import MultiClientOpenAILLM


# Fix random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def build_qa_prompt(query, top_k=10):
    def build_documents(docs, top_k):
        return "\n".join(
            [f"[Document {i + 1}, Title {doc['title']}]\n{doc['text']}" for i, doc in enumerate(docs[:top_k])]
        )

    documents_str = build_documents(query["docs"], top_k)

    prompt = (
        "You are given several documents and a question.\n"
        "Respond with a short answer (MAX 5 tokens) based strictly on the documents.\n"
        "Do NOT add any explanation or reasoning.\n\n"
        f"{documents_str}\n\n"
        f"Question: {query['question']}\n"
    )

    return prompt


def get_responses(llm: MultiClientOpenAILLM, data, response_cache_dict: Dict):
    data = sorted(data, key=lambda d: d["id"])
    data_match, data_wo_match = [], []

    for d in data:
        d["prompt"] = build_qa_prompt(d)
        d["hash_value"] = str(stable_hash(d["prompt"]))
        if d["hash_value"] in response_cache_dict.keys():
            d["resp"] = response_cache_dict[d["hash_value"]]
            data_match.append(d)
        else:
            data_wo_match.append(d)

    if len(data_wo_match) != 0:
        prompts = [d["prompt"] for d in data_wo_match]
        responses, _ = llm.generate(prompts)
        for resp, d in zip(responses, data_wo_match):
            d["resp"] = resp
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
    # Deduplicate, lowercase, and sort by length ascending (shorter answers take precedence).
    strings = [string.lower() for string in strings]
    strings = sorted(set(strings), key=len)
    result = []

    for s in strings:
        if any(ele in s for ele in result):
            continue
        result.append(s)

    return result


if __name__ == '__main__':
    api = ""  # API for OpenAI Service
    url = ""  # URL for OpenAI Service
    llm_name = "Qwen/Qwen3-8B"

    corpus_path = "data/enwiki_20231101/corpus.jsonl"
    line_offset_path = "data/enwiki_20231101/corpus_line_offset.pkl"
    
    llm = MultiClientOpenAILLM([api], url, llm_name, max_workers=1)

    llm_resp_cache_path = "data/llm_resp_cache/qwen_cache.json"
    record_root = "data/llm_resp_cache/run_record"
    save_root = "data/llm_resp_cache/record_after_qwen"
    
    for file_name in sorted(glob.glob(os.path.join(record_root, "*"))):
        file_name = os.path.split(file_name)[1]
        try:
            with open(llm_resp_cache_path, "r", encoding="utf-8") as f:
                response_cache_dict = json.load(f)
        except FileNotFoundError:
            response_cache_dict = {}

        with open(line_offset_path, 'rb') as f:
            line_offsets = pickle.load(f)

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
                d["answer_list"] = normalize_answer_list(d["answers"])
            write_jsonl(data, os.path.join(save_root, file_name))

        data = load_jsonl_file(os.path.join(save_root, file_name))
        data, response_cache_dict = get_responses(llm, data, response_cache_dict)
        write_jsonl(data, os.path.join(save_root, file_name))
        with open(llm_resp_cache_path, "w", encoding="utf-8") as f:
            json.dump(response_cache_dict, f)
