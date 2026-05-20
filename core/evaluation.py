import json

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL
)
from utils.logger import get_logger

logger = get_logger(__name__)


# def _build_ragas_llm() -> LangchainLLMWrapper:
#     """
#     Wrap the Groq LLM for RAGAS evaluation.
#     """
#     llm = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name=LLM_MODEL,
#         temperature=0
#     )
#     return LangchainLLMWrapper(llm)


# def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
#     """
#     Wrap HuggingFace embeddings for RAGAS evaluation.
#     """
#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL
#     )
#     return LangchainEmbeddingsWrapper(embeddings)


def _build_ragas_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0
    )

def _build_ragas_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

def build_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str]
) -> Dataset:
    """
    Build a HuggingFace Dataset from evaluation data.
    contexts must be a list of lists — one list of chunks per question.
    """
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    output_file: str = "evaluation_results.json"
) -> dict:
    """
    Run RAGAS evaluation and return scores.
    Saves results to a JSON file.

    Metrics:
    - faithfulness       : Is the answer grounded in the context?
    - answer_relevancy   : Is the answer relevant to the question?
    - context_precision  : Are retrieved chunks actually useful?
    - context_recall     : Did retrieval find all necessary information?
    """
    logger.info("Building RAGAS dataset...")
    dataset = build_dataset(questions, answers, contexts, ground_truths)

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    logger.info("Running RAGAS evaluation...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    scores = result.to_pandas().to_dict(orient="records")

    output = {
        "scores": scores,
        "aggregate": {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"]
        }
    }

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

    return output


def score_response(response: str, ground_truth: str) -> str:
    """
    Simple word overlap scorer used in the Streamlit eval panel.
    Returns 'High', 'Medium', or 'Low'.
    """
    no_answer = "the information is not available in the uploaded documents."

    if response.strip().lower() == no_answer:
        return "Low"

    response_words = set(response.lower().split())
    truth_words = set(ground_truth.lower().split())

    if not truth_words:
        return "Low"

    overlap = response_words & truth_words
    ratio = len(overlap) / len(truth_words)

    if ratio >= 0.6:
        return "High"
    elif ratio >= 0.3:
        return "Medium"
    return "Low"


def save_evaluation_results(
    questions: list[str],
    answers: list[str],
    contexts: list[str],
    ground_truths: list[str],
    output_file: str = "evaluation_results.json"
) -> None:
    """
    Save simple word-overlap evaluation results to JSON.
    Used by test_eval.py.
    """
    results = [
        {
            "question": q,
            "answer": a,
            "context": c,
            "ground_truth": g,
            "score": score_response(a, g)
        }
        for q, a, c, g in zip(questions, answers, contexts, ground_truths)
    ]

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")