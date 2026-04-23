import os

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils.logger import logger, log_function_call
from utils.text_processing import normalize_text
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def compute_scores_to_batch_embeddings(anchor_embedding, batch_embeddings):
    anchor_embedding = torch.from_numpy(anchor_embedding).to("cuda").unsqueeze(0)
    batch_embeddings = torch.from_numpy(batch_embeddings).to("cuda")
    scores = torch.matmul(anchor_embedding, batch_embeddings.T)
    scores = scores.squeeze(0).cpu().numpy()
    return scores


class Encoder:
    @log_function_call(logger)
    def __init__(
            self, model_name, use_fp16=True, batch_size=512
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.model.similarity_fn_name = "dot"
        self.batch_size = batch_size
        if use_fp16:
            self.model = self.model.half()
        logger.info(f"Model loaded : {model_name}")

    def encode(
            self, query: str,
            do_lowercase: bool = False,
            do_normalize: bool = False,
    ):
        query = normalize_text(query, do_lowercase=do_lowercase, do_normalize=do_normalize)
        query_embedding = self.model.encode(query).reshape((1, -1))

        return query_embedding

    def batch_encode(self, queries, do_lowercase=False, do_normalize=False, tqdm_bar=True):
        queries = [normalize_text(q, do_lowercase=do_lowercase, do_normalize=do_normalize) for q in queries]
        query_embeddings = []
        if tqdm_bar:
            iterator = tqdm(range(0, len(queries), self.batch_size))
        else:
            iterator = range(0, len(queries), self.batch_size)
        for i in iterator:
            batch_queries = queries[i: min(i + self.batch_size, len(queries))]
            query_embedding = self.model.encode(batch_queries)
            query_embeddings.append(query_embedding)
        query_embeddings = np.concatenate(query_embeddings, axis=0)
        return query_embeddings

    def compute_scores(self, batch_data, batch_embeddings=None, block_size=512):
        if batch_embeddings is None:
            batch_embeddings = self.batch_encode(queries=batch_data)

        batch_embeddings = torch.from_numpy(batch_embeddings).to("cuda")
        N = batch_embeddings.size(0)
        scores = torch.zeros((N, N), device="cuda", dtype=torch.float16)
        for i in tqdm(range(0, N, block_size)):
            end_i = min(i + block_size, N)
            scores[i:end_i] = torch.matmul(batch_embeddings[i:end_i], batch_embeddings.T)
        scores = scores.cpu().numpy()
        return scores
