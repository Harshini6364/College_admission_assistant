import json
import os

from utils.logger import get_logger

logger = get_logger(__name__)

QA_FILE = "qa_pairs.json"


def load_qa_pairs() -> list[dict]:
    if not os.path.exists(QA_FILE):
        logger.warning(f"{QA_FILE} not found.")
        return []

    try:
        with open(QA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load QA pairs: {e}")
        return []


def find_ground_truth(query: str, qa_pairs: list[dict]) -> str:
    """
    Match query to closest question in qa_pairs.
    Returns ground truth if found, empty string if not.
    """
    query_words = set(query.lower().split())

    best_match = ""
    best_score = 0

    for pair in qa_pairs:
        question_words = set(pair["question"].lower().split())
        overlap = query_words & question_words
        score = len(overlap) / max(len(question_words), 1)

        if score > best_score:
            best_score = score
            best_match = pair["ground_truth"]

    # Only return if match is confident enough
    if best_score >= 0.5:
        return best_match

    return ""