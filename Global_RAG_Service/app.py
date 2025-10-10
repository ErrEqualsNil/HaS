import random
import numpy as np
from flask import Flask, request, jsonify

from Global_RAG_Service.Retriever import Retriever

app = Flask(__name__)

use_precomputed_retriever = True
retriever = Retriever()
retriever.reset(use_precomputed_retriever)


def simulate_network_delay(min_delay=0.1, max_delay=0.2):
    delay = random.uniform(min_delay, max_delay)
    return delay


def simulate_retrieve_delay(mean=1.23411, std=0.056904):
    delay = np.random.normal(loc=mean, scale=std)
    return delay


@app.route('/search', methods=["POST"])
def search():  # put application's code here
    network_delay = simulate_network_delay()

    data = request.get_json()
    query_embedding = data.get("query_embedding", None)
    query_str = data.get("query_str", None)
    n_docs = data.get("n_docs")
    require_doc_info = data.get("require_embeddings", False)

    query = query_embedding if query_embedding is not None else query_str
    docs, doc_embeddings = retriever.search(query, n_docs, require_doc_info)
    retrieval_delay = simulate_retrieve_delay() if use_precomputed_retriever else 0
    return jsonify({
        "docs": docs, "doc_embeddings": doc_embeddings,
        "simulate_delay": network_delay + retrieval_delay,
    })


if __name__ == '__main__':
    app.run("0.0.0.0", 6000)
