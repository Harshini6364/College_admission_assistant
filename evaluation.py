from datasets import Dataset


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

    print("\nRAGAS-compatible dataset prepared successfully\n")

    print("Questions:")
    print(questions)

    print("\nAnswers:")
    print(answers)

    print("\nGround Truths:")
    print(ground_truths)

    return dataset