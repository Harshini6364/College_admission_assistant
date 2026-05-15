from evaluation import evaluate_rag

questions = [
    "What is the admission fee?"
]

answers = [
    "The admission fee is 1 lakh rupees."
]

contexts = [[
    "The admission fee for students is 1 lakh rupees per year."
]]

ground_truths = [
    "The admission fee is 1 lakh rupees per year."
]

result = evaluate_rag(
    questions,
    answers,
    contexts,
    ground_truths
)

print(result)