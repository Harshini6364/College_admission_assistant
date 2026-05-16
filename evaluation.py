from datasets import Dataset

import json


def evaluate_rag(
    questions,
    answers,
    contexts,
    ground_truths
):

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    results = []

    for i in range(len(questions)):

        result = {
            "question": questions[i],
            "answer": answers[i],
            "context": contexts[i],
            "ground_truth": ground_truths[i]
        }

        results.append(result)

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nEvaluation results saved successfully\n"
    )

    return dataset