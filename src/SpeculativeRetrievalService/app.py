from flask import Flask, request, jsonify

from .SpeculativeRetriever import SpeculativeRetriever

app = Flask(__name__)
retriever: SpeculativeRetriever = SpeculativeRetriever()

retriever.reset(
    "data/enwiki_20231101/corpus_embeddings.npy", 
    "data/enwiki_20231101/corpus_ids.npy", 
    "BAAI/bge-base-en-v1.5",
    cache_max_size=5000
)

@app.route('/search', methods=["POST"])
def search():
    global retriever
    data = request.get_json()
    query_info = data.get("query_info")
    n_docs = data.get("n_docs")
    threshold = data.get("threshold")
    doc_ids, is_speculative_retrieval_match, similar_query, simulated_network_latency, similar_score = (
        retriever.search(query_info, n_docs, threshold)
    )
    return jsonify({
        "doc_ids": doc_ids,
        "is_speculative_retrieval_match": is_speculative_retrieval_match,
        "similar_query_info": similar_query,
        "simulated_network_latency": simulated_network_latency,
        "similar_score": similar_score,
    })


if __name__ == '__main__':
    app.run("0.0.0.0", 5999)
