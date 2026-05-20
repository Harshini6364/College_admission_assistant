from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
 
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from utils.logger import get_logger
 
logger = get_logger(__name__)
 
 
def setup_rag(
    documents: list[Document]
) -> tuple[list[Document], FAISS, BM25Okapi]:
    """
    Chunk documents, build FAISS vector store, and BM25 index.
    Returns (docs, vectorstore, bm25).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
 
    docs = splitter.split_documents(documents)
    logger.info(f"Split into {len(docs)} chunks")
 
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
 
    vectorstore = FAISS.from_documents(docs, embedding_model)
    logger.info("FAISS vector store built")
 
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in docs
    ]
 
    bm25 = BM25Okapi(tokenized_docs)
    logger.info("BM25 index built")
 
    return docs, vectorstore, bm25
 
