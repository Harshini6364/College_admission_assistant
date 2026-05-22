# ===========================================
# ABSTRACTION LAYER: RAG Pipeline
# GOF Pattern: Template Method
# Responsibility: End-to-end RAG orchestration
# Steel Thread: Query → Retrieve → Generate → Answer
# ===========================================

from langchain_core.documents import Document

from core.loader import load_documents
from core.rag import setup_rag
from core.retriever import retrieve_chunks
from core.llm_handler import LLMSingleton, generate_response
from core.prompts import build_prompt

from config import DATA_FOLDER, TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    GOF Template Method Pattern:
    Defines the skeleton of the RAG algorithm.
    Each stage can be overridden by subclasses for customization.

    Steel Thread:
    The thinnest end-to-end working pipeline:
    Query → Retrieve → Generate → Answer

    Reusable:
    Any project can instantiate this with their own data folder.

    Usage:
        pipeline = RAGPipeline(data_folder="data/")
        answer = pipeline.run("What is the admission fee?")
    """

    def __init__(self, data_folder: str = DATA_FOLDER):
        self.data_folder = data_folder
        self.docs = None
        self.vectorstore = None
        self.bm25 = None
        self.llm = LLMSingleton.get_instance()
        self._initialized = False

    # ---------------------------
    # SETUP
    # ---------------------------

    def initialize(self) -> None:
        """
        Load documents and build indexes.
        Call once before running queries.
        """
        if self._initialized:
            logger.info("Pipeline already initialized — skipping")
            return

        logger.info("Initializing RAG pipeline...")

        documents = self.load(self.data_folder)
        self.docs, self.vectorstore, self.bm25 = self.index(documents)
        self._initialized = True

        logger.info("Pipeline initialized successfully")

    # ---------------------------
    # STAGE 1: LOAD (Rewrite happens inside index)
    # ---------------------------

    def load(self, data_folder: str) -> list[Document]:
        """
        Template Method Stage 1: Load documents.
        Override this to support different data sources.
        """
        logger.info(f"Loading documents from: {data_folder}")
        return load_documents(data_folder)

    # ---------------------------
    # STAGE 2: INDEX (Rewrite — chunking + embedding)
    # ---------------------------

    def index(self, documents: list[Document]) -> tuple:
        """
        Template Method Stage 2: Chunk and index documents.
        Override this to use different vector stores or chunking strategies.
        """
        logger.info("Indexing documents...")
        return setup_rag(documents)

    # ---------------------------
    # STAGE 3: RETRIEVE + RERANK + REFINE
    # ---------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Document]:
        """
        Template Method Stage 3: Retrieve relevant chunks.
        Includes BM25 + FAISS + CrossEncoder reranking + refining.
        Override this to use a different retrieval strategy.
        """
        logger.info(f"Retrieving for: '{query}'")
        return retrieve_chunks(
            query,
            self.docs,
            self.bm25,
            self.vectorstore,
            top_k
        )

    # ---------------------------
    # STAGE 4: INSERT (Build Prompt)
    # ---------------------------

    def insert(
        self,
        query: str,
        relevant_docs: list[Document],
        history: str = ""
    ) -> str:
        """
        Template Method Stage 4: Insert context into prompt.
        Override this to use different prompt templates.
        """
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return build_prompt(history, context, query)

    # ---------------------------
    # STAGE 5: GENERATE
    # ---------------------------

    def generate(self, prompt: str) -> str:
        """
        Template Method Stage 5: Generate answer from LLM.
        Override this to use a different LLM.
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ---------------------------
    # STEEL THREAD — Full Pipeline Run
    # ---------------------------

    def run(self, query: str, history: str = "") -> dict:
        """
        Steel Thread: Complete end-to-end RAG pipeline.

        Input:  query (str)
        Output: dict with answer, retrieved docs, context

        This is the single entry point for running the pipeline.
        Reusable across different projects.
        """
        if not self._initialized:
            self.initialize()

        # Stage 3: Retrieve + Rerank + Refine
        relevant_docs = self.retrieve(query)

        # Stage 4: Insert
        prompt = self.insert(query, relevant_docs, history)

        # Stage 5: Generate
        answer = self.generate(prompt)

        logger.info(f"Pipeline complete for: '{query}'")

        return {
            "query": query,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "context": "\n\n".join([doc.page_content for doc in relevant_docs])
        }