# ===========================================
# ABSTRACTION LAYER: LLM
# GOF Pattern: Singleton
# Responsibility: Generate responses from LLM
# Input: query, context, history, llm
# Output: str (response)
# ===========================================

from langchain_groq import ChatGroq

from core.prompts import build_prompt
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------
# GOF SINGLETON PATTERN
# ---------------------------

class LLMSingleton:
    """
    GOF Singleton Pattern:
    Ensures only one LLM instance is created
    and reused across the entire application.

    Why Singleton here:
    - Creating a ChatGroq instance on every request is wasteful
    - API connections are expensive to initialise repeatedly
    - One instance shared across pipeline, app.py, and tests

    High Cohesion: only manages LLM instance lifecycle
    Low Coupling: rest of the app calls get_instance() — never creates ChatGroq directly
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> ChatGroq:
        """
        Return the single shared LLM instance.
        Creates it on first call, reuses on all subsequent calls.
        """
        if cls._instance is None:
            logger.info("Creating LLM instance (Singleton)...")
            cls._instance = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE
            )
            logger.info(f"LLM ready: {LLM_MODEL}")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton — useful for testing.
        """
        cls._instance = None
        logger.info("LLM Singleton reset")


# ---------------------------
# GENERATE FUNCTION
# ---------------------------

def generate_response(
    query: str,
    context: str,
    history: str,
    llm: ChatGroq = None
) -> str:
    """
    Build prompt and invoke the LLM.
    Returns the response text.

    llm parameter is optional — if not provided,
    uses the Singleton instance automatically.

    Single Responsibility: only generates, does not retrieve or evaluate.
    """
    if llm is None:
        llm = LLMSingleton.get_instance()

    prompt = build_prompt(history, context, query)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return "An error occurred while generating the response."