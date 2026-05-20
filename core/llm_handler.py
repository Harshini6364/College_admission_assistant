from langchain_groq import ChatGroq

from core.prompts import build_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_response(
    query: str,
    context: str,
    history: str,
    llm: ChatGroq
) -> str:
    """
    Build prompt and invoke the LLM.
    Returns the response text.
    """
    prompt = build_prompt(history, context, query)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return "An error occurred while generating the response."