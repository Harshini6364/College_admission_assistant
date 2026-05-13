# Handles:
# chunking
# embeddings
# FAISS
# BM25

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL


def setup_rag(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return docs, vectorstore, bm25