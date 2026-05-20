import streamlit as st

from utils.chat_history import save_chat_history


def render_sidebar(chat_sessions: dict) -> None:
    """
    Render the sidebar with new chat button and chat history list.
    Mutates st.session_state on selection/creation.
    """
    with st.sidebar:

        st.header("⚙️ Controls")

        if st.button("➕ New Chat"):
            new_name = f"Chat {len(chat_sessions) + 1}"
            chat_sessions[new_name] = []
            st.session_state.current_chat = new_name
            save_chat_history(chat_sessions)
            st.rerun()

        st.divider()
        st.subheader("🕘 History")

        for chat_name in reversed(list(chat_sessions.keys())):

            messages = chat_sessions[chat_name]
            preview = chat_name

            for role, msg in messages:
                if role == "user":
                    preview = msg[:30] + "..."
                    break

            if st.button(preview, key=chat_name):
                st.session_state.current_chat = chat_name
                st.rerun()