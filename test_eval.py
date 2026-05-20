from core.evaluation import score_response, save_evaluation_results
from utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation_test():

    # ---------------------------
    # SAMPLE TEST DATA
    # ---------------------------
    questions = [
        "What is the admission fee?",
        "What documents are required for admission?",
        "What is the last date to apply?"
    ]

    answers = [
        "The admission fee is 1 lakh rupees.",
        "The required documents are 10th and 12th marksheets, transfer certificate, and passport photo.",
        "The last date to apply is 30th June."
    ]

    contexts = [
        "The admission fee for students is 1 lakh rupees per year.",
        "Students must submit 10th and 12th marksheets, transfer certificate, and passport size photograph.",
        "Applications must be submitted before 30th June every year."
    ]

    ground_truths = [
        "The admission fee is 1 lakh rupees per year.",
        "Required documents include 10th and 12th marksheets, transfer certificate, and passport photo.",
        "The last date to apply is 30th June."
    ]

    # ---------------------------
    # SCORE EACH RESPONSE
    # ---------------------------
    logger.info("Running evaluation...")

    for i, (answer, ground_truth) in enumerate(
        zip(answers, ground_truths)
    ):
        score = score_response(answer, ground_truth)
        logger.info(f"Q{i+1}: {questions[i]}")
        logger.info(f"  Score: {score}")

    # ---------------------------
    # SAVE RESULTS
    # ---------------------------
    save_evaluation_results(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        output_file="test_evaluation_results.json"
    )

    logger.info("Evaluation test complete. Results saved to test_evaluation_results.json")


if __name__ == "__main__":
    run_evaluation_test()