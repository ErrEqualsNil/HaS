def build_qa_prompt(query, top_k=10):
    def build_documents(docs, top_k):
        return "\n".join(
            [f"[Document {i + 1}, Title {doc['title']}]\n{doc['text']}" for i, doc in enumerate(docs[:top_k])]
        )

    documents_str = build_documents(query["docs"], top_k)

    prompt = (
        "You are given several documents and a question.\n"
        "Respond with a short answer (MAX 5 tokens) based strictly on the documents.\n"
        # "If the documents do not contain sufficient information, respond with: NO-RES\n"
        "Do NOT add any explanation or reasoning.\n\n"
        f"{documents_str}\n\n"
        f"Question: {query['question']}\n"
    )

    return prompt
