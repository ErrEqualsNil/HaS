from flask import Flask, request, jsonify

from Edge_RAG_Service.SpeculativeRetriever import SpeculativeRetriever

app = Flask(__name__)
speculative_retriever: SpeculativeRetriever = SpeculativeRetriever()


@app.route('/reset', methods=["POST"])
def reset():
    global speculative_retriever
    speculative_retriever.reset()
    return jsonify({"status": True})


@app.route('/search', methods=["POST"])
def search():
    global speculative_retriever
    data = request.get_json()
    query_info = data.get("query_info")
    n_docs = data.get("n_docs")
    threshold = data.get("threshold")
    doc_ids, is_cache_match, similar_query, simulate_delay, similar_score = (
        speculative_retriever.search(query_info, n_docs, threshold)
    )
    return jsonify({
        "doc_ids": doc_ids,
        "is_cache_match": is_cache_match,
        "similar_query_info": similar_query,
        "simulate_delay": simulate_delay,
        "similar_score": similar_score,
    })


if __name__ == '__main__':
    app.run("0.0.0.0", 5999)
