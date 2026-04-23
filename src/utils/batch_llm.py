import datetime
import itertools
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List, Union, Dict, Tuple

import openai
from openai import OpenAI
from tqdm import tqdm

socket.setdefaulttimeout(60)


class OpenAIConnectionManager:
    def __init__(self, api_key_list, base_url):
        self.base_url = base_url
        self.lock = threading.Lock()

        self.clients = {
            key: {
                "api_key": key,
                "client": OpenAI(api_key=key, base_url=base_url),
                "status": True,
                "last_fail": datetime.datetime.now(),
            }
            for key in api_key_list
        }

        self._refresh_iterator()

    def _refresh_iterator(self):
        # snapshot → safe for iteration
        self.client_iter = itertools.cycle(list(self.clients.values()))

    def get_client(self):
        while True:
            with self.lock:
                for client_info in self.client_iter:
                    if client_info["status"]:
                        return client_info

                    if datetime.datetime.now() > client_info["last_fail"] + datetime.timedelta(minutes=2):
                        client_info["status"] = True
                        return client_info

            print("Fail to get Available Client, Wait for 10 seconds...")
            time.sleep(10)

    def set_client_status(self, client_info, status):
        with self.lock:
            client_info["status"] = status
            client_info["last_fail"] = datetime.datetime.now()

    def reset_single_client(self, client_info):
        with self.lock:
            key = client_info["api_key"]
            client_info["client"] = OpenAI(api_key=key, base_url=self.base_url)
            client_info["status"] = True
            client_info["last_fail"] = datetime.datetime.now()
            self._refresh_iterator()


class MultiClientOpenAILLM:
    def __init__(
        self,
        api_keys: List[str],
        base_url: str,
        model_name: str,
        max_workers: int,
    ):
        self.client_manager = OpenAIConnectionManager(api_keys, base_url)
        self.model_name = model_name

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _sync_call(self, prompt: str, max_tokens=4096) -> Tuple[str, Dict]:
        retries = 0
        backoff = 1

        while True:
            client_info = self.client_manager.get_client()
            client = client_info["client"]

            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                resp = completion.choices[0].message.content.strip()
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                    "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
                    "total_tokens": completion.usage.total_tokens if completion.usage else None,
                }
                self.client_manager.set_client_status(client_info, True)
                break

            except (openai.RateLimitError, openai.PermissionDeniedError) as e:
                print(f"[ERROR] {e}")
                self.client_manager.set_client_status(client_info, False)

            except (openai.APIConnectionError, openai.InternalServerError) as e:
                retries += 1
                print(f"[ERROR] API 调用失败（第 {retries} 次）：{e}")

                print("[INFO] 正在重新构建 OpenAI client...")
                self.client_manager.reset_single_client(client_info)

                print(f"[INFO] 等待 {backoff} 秒后重试...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        return resp, usage

    def generate(self, prompts: List[str], batch_size=1024) -> Tuple[List[str], List[Union[None, Dict]]]:
        results = [""] * len(prompts)
        usages = [None] * len(prompts)

        with tqdm(total=len(prompts), desc="LLM Generating") as pbar:
            for batch_start_idx in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start_idx:batch_start_idx + batch_size]

                future_to_idx = {
                    self.executor.submit(self._sync_call, prompt): batch_start_idx + i
                    for i, prompt in enumerate(batch_prompts)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        resp, usage = future.result()
                        results[idx] = resp
                        usages[idx] = usage
                    except Exception as e:
                        results[idx] = ""
                        usages[idx] = None
                        print(f"[ERROR] Error at index {idx}: {e}")
                    finally:
                        pbar.update(1)

        return results, usages
