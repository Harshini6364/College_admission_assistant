import os
import json

from config import CHAT_HISTORY_FILE
from utils.logger import get_logger

logger = get_logger(__name__)


def load_chat_history() -> dict:
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")

    return {"Chat 1": []}


def save_chat_history(chat_sessions: dict) -> None:
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_sessions, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")