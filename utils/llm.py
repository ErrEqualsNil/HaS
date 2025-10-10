from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLLM:
    def __init__(self, batch_size=8):
        model_name = "Qwen/Qwen3-8B-AWQ"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', revision="4a2d74b63371c8cd724aa15dc5efbd329b16f573")
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", revision="4a2d74b63371c8cd724aa15dc5efbd329b16f573")

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> Optional[List[str]]:
        text_list = []
        for prompt in prompts:
            message = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            text_list.append(text)

        sorted_list = sorted(enumerate(text_list), reverse=True, key=lambda t: len(t[1].split(" ")))
        sort_indices, text_list = zip(*sorted_list)

        sorted_responses = []
        for idx in tqdm(range(0, len(text_list), self.batch_size)):
            batch_text = text_list[idx: idx + self.batch_size]
            model_inputs = self.tokenizer(batch_text, return_tensors="pt", padding=True).to(self.llm.device)
            outputs = self.llm.generate(
                **model_inputs,
                max_new_tokens=20
            )
            output_ids = outputs[:, model_inputs.input_ids.shape[1]:]
            contents = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            sorted_responses += contents

        responses = [""] * len(prompts)
        for i, idx in enumerate(sort_indices):
            responses[idx] = sorted_responses[i]
        return responses
