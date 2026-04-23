import random

from flask import Flask, request, jsonify

from .Retriever import Retriever

app = Flask(__name__)

retriever = Retriever(
    "data/enwiki_20231101/corpus_embeddings.npy", 
    "data/enwiki_20231101/corpus_ids.npy", 
    "BAAI/bge-base-en-v1.5"
)
retriever.reset()


def simulate_network_delay(min_delay=0.1, max_delay=0.2):
    delay = random.uniform(min_delay, max_delay)
    return delay



@app.route('/search', methods=["POST"])
def search():  # put application's code here
    network_delay = simulate_network_delay()

    data = request.get_json()
    query_embedding = data.get("query_embedding", None)
    query_str = data.get("query_str", None)
    n_docs = data.get("n_docs")
    require_doc_embeddings = data.get("require_doc_embeddings", False)

    query = query_embedding if query_embedding is not None else query_str
    docs, doc_embeddings = retriever.search(query, n_docs, require_doc_embeddings)
    return jsonify({
        "docs": docs, "doc_embeddings": doc_embeddings,
        "simulated_network_latency": network_delay,  # end-to-end delay will be included by the Edge
    })


if __name__ == '__main__':
    app.run("0.0.0.0", 6000)
