from dotenv import load_dotenv

from langchain_groq import ChatGroq

from config import LLM_MODEL

from loader import load_documents

from rag import setup_rag

from retriever import retrieve_chunks

from evaluation import evaluate_rag

from llm_handler import generate_response


load_dotenv()



documents = load_documents("data")

docs, vectorstore, bm25 = setup_rag(
    documents
)


query = "What is the admission fee?"


retrieved_docs = retrieve_chunks(
    query,
    docs,
    bm25,
    vectorstore
)


context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
)

llm = ChatGroq(
    model=LLM_MODEL
)
history = []
answer = generate_response(
    query=query,
    context=context,
    history=history,
    llm=llm
)

questions = [query]

answers = [answer]

contexts = [[context]]

ground_truths = [
    "The admission fee is 1 lakh rupees."
]


result = evaluate_rag(
    questions,
    answers,
    contexts,
    ground_truths
)


print(result)