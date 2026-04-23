from typing import List
import os
import psutil
import numpy as np

from utils.text_processing import normalize_answer, normalize_text


def _is_answer_in_text(text: str, answers: List[str]) -> bool:
    """
    Checks if any of the provided answers are present in the given text.
    """
    text = normalize_text(text, do_normalize=True, do_lowercase=True)
    for a in answers:
        normalized_answer_lower = normalize_answer(a, lowercase=True)
        if len(normalized_answer_lower) == 0:
            continue
        if normalized_answer_lower in text:
            return True
    return False


def document_hit_rate(data, k, only_cache_hit=False):
    hit_cnt = 0
    total_cnt = 0
    for d in data:
        if only_cache_hit and not d["is_cache_match"]:
            continue
        total_cnt += 1
        for docs in d["docs"][:k]:
            if _is_answer_in_text(docs["text"], d["answer_list"]):
                hit_cnt += 1
                break
    return hit_cnt / total_cnt


def response_hit_rate(data, only_cache_hit=False):
    correct_cnt = 0
    total_cnt = 0
    for d in data:
        if only_cache_hit and not d["is_cache_match"]:
            continue
        total_cnt += 1
        if _is_answer_in_text(d["resp"], d["answer_list"]):
            correct_cnt += 1
    return correct_cnt / total_cnt


def delta_response_hit_rate(data, global_data):
    a, b = 0, 0
    matrix = np.zeros((2, 2))
    total_cnt = 0
    for d, gd in zip(data, global_data):
        if not d["is_cache_match"]:
            continue
        total_cnt += 1
        d_match = _is_answer_in_text(d["resp"], d["answer_list"])
        gd_match = _is_answer_in_text(gd["resp"], gd["answer_list"])
        matrix[int(d_match)][int(gd_match)] += 1
        if d_match:
            a += 1
        if gd_match:
            b += 1
    return (b - a) / total_cnt


def exactly_match_rate(data, only_cache_hit=False):
    em_cnt = 0
    total_cnt = 0
    for d in data:
        if only_cache_hit and not d["is_cache_match"]:
            continue
        total_cnt += 1
        resp = normalize_text(d["resp"], do_lowercase=True, do_normalize=True)
        for answer in d["answer_list"]:
            answer = normalize_answer(answer, lowercase=True)
            if resp == answer:
                em_cnt += 1
                break
    return em_cnt / total_cnt


def average_time_cost(data):
    time_costs = np.array([d["rag_time_cost"] for d in data])
    avg_time_cost = time_costs.mean()
    return avg_time_cost


def cache_hit_rate(data):
    is_hits = np.array([d["is_cache_match"] for d in data]).astype(int)
    hit_rate = np.sum(is_hits) / len(data)
    return hit_rate


def cache_hit_correct_rate(data):
    tot, cnt = 0, 0
    sim_scores = [[], []]
    for r in data:
        if not r["is_cache_match"]:
            continue
        tot += 1
        if r["similar_query_info"]["question_entity"] == r["question_entity"]:
            sim_scores[0].append(r["similar_score"])
            cnt += 1
        else:
            sim_scores[1].append(r["similar_score"])
    hit_correct_rate = cnt / tot
    return hit_correct_rate


def cache_hit_time_cost(data):
    time_costs = np.array([d["rag_time_cost"] for d in data if d["is_cache_match"]])
    avg_time_cost = time_costs.mean()
    return avg_time_cost


def cache_miss_time_cost(data):
    time_costs = np.array([d["rag_time_cost"] for d in data if not d["is_cache_match"]])
    avg_time_cost = time_costs.mean()
    return avg_time_cost


def print_memory_usage():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()

    rss_in_mb = mem_info.rss / 1024 / 1024  # 转换为 MB
    vms_in_mb = mem_info.vms / 1024 / 1024  # 转换为 MB

    print(f"RSS (Resident Set Size): {rss_in_mb:.2f} MB")
    print(f"VMS (Virtual Memory Size): {vms_in_mb:.2f} MB")