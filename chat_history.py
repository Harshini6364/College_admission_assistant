import os
import json

CHAT_HISTORY_FILE = "chat_history.json"


def load_chat_history():

    if os.path.exists(CHAT_HISTORY_FILE):

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return {
        "Chat 1": []
    }


def save_chat_history(chat_sessions):

    with open(
        CHAT_HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chat_sessions,
            f,
            indent=4
        )