from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
 
from config import TOP_K
from utils.logger import get_logger
 
logger = get_logger(__name__)
 
 
def retrieve_chunks(
    query: str,
    docs: list[Document],
    bm25: BM25Okapi,
    vectorstore: FAISS,
    top_k: int = TOP_K
) -> list[Document]:
    """
    Hybrid retrieval: BM25 keyword search + FAISS vector search.
    Deduplicates and returns top_k unique chunks.
    """
    tokenized_query = query.lower().split()
 
    bm25_scores = bm25.get_scores(tokenized_query)
 
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]
 
    bm25_docs = [docs[i] for i in bm25_indices]
 
    vector_docs = vectorstore.similarity_search(query, k=top_k)
 
    combined = []
    seen = set()
 
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)
 
    logger.info(f"Retrieved {len(combined[:top_k])} chunks for query")
    return combined[:top_k]
 
