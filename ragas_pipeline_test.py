from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import LLM_MODEL, GROQ_API_KEY, DATA_FOLDER

from core.loader import load_documents
from core.rag import setup_rag
from core.retriever import retrieve_chunks
from core.llm_handler import generate_response
from core.evaluation import run_ragas_evaluation

from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def run_pipeline_test():

    # ---------------------------
    # LOAD DOCUMENTS
    # ---------------------------
    logger.info("Loading documents...")
    documents = load_documents(DATA_FOLDER)

    # ---------------------------
    # SETUP RAG
    # ---------------------------
    logger.info("Setting up RAG pipeline...")
    docs, vectorstore, bm25 = setup_rag(documents)

    # ---------------------------
    # LLM
    # ---------------------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0
    )

    # ---------------------------
    # TEST QUERIES
    # Add your own questions and ground truths here
    # ---------------------------
    questions = [
        "What is the admission fee?",
        "What documents are required for admission?",
        "What is the last date to apply?"
    ]

    ground_truths = [
        "The admission fee is 1 lakh rupees per year.",
        "Required documents include 10th and 12th marksheets, transfer certificate, and passport photo.",
        "The last date to apply is 30th June."
    ]

    # ---------------------------
    # RETRIEVE + GENERATE
    # ---------------------------
    answers = []
    contexts = []

    for question in questions:

        logger.info(f"Processing: {question}")

        retrieved_docs = retrieve_chunks(
            question, docs, bm25, vectorstore
        )

        context_chunks = [doc.page_content for doc in retrieved_docs]

        context_str = "\n\n".join(context_chunks)

        answer = generate_response(
            query=question,
            context=context_str,
            history="",
            llm=llm
        )

        answers.append(answer)
        contexts.append(context_chunks)  # list of strings per question

        logger.info(f"Answer: {answer}")

    # ---------------------------
    # RAGAS EVALUATION
    # ---------------------------
    logger.info("Starting RAGAS evaluation...")

    results = run_ragas_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        output_file="evaluation_results.json"
    )

    # ---------------------------
    # PRINT AGGREGATE SCORES
    # ---------------------------
    print("\n========== RAGAS Evaluation Results ==========")
    for metric, score in results["aggregate"].items():
        print(f"  {metric:<25} : {round(score, 4)}")
    print("===============================================\n")

    logger.info("Pipeline test complete.")


if __name__ == "__main__":
    run_pipeline_test()