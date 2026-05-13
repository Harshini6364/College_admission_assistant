# only retriever logic

def retrieve_chunks(
    query,
    docs,
    bm25,
    vectorstore,
    top_k=3
):

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    bm25_docs = [docs[i] for i in bm25_idx]

    vector_docs = vectorstore.similarity_search(
        query,
        k=top_k
    )

    combined = []
    seen = set()

    for d in bm25_docs + vector_docs:

        if d.page_content not in seen:

            combined.append(d)

            seen.add(d.page_content)

    return combined[:top_k]