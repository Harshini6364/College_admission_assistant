import os
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain.memory.buffer import (
    ConversationBufferMemory
)

from loader import load_documents
from rag import setup_rag
from retriever import retrieve_chunks

from config import LLM_MODEL

from chat_history import (
    load_chat_history,
    save_chat_history
)

from llm_handler import generate_response

# ---------------------------
# LOAD ENV
# ---------------------------
load_dotenv()

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="College Admission Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Admission Assistant")
st.write("Ask anything about your college 👇")

# ---------------------------
# SESSION STATE
# ---------------------------
if "chat_sessions" not in st.session_state:

    st.session_state.chat_sessions = (
        load_chat_history()
    )

if "current_chat" not in st.session_state:

    st.session_state.current_chat = list(
        st.session_state.chat_sessions.keys()
    )[0]

if "cache" not in st.session_state:

    st.session_state.cache = {}

if "memories" not in st.session_state:

    st.session_state.memories = {}

current_chat = st.session_state.current_chat

if current_chat not in st.session_state.memories:

    st.session_state.memories[
        current_chat
    ] = ConversationBufferMemory(
        return_messages=False
    )

memory = st.session_state.memories[
    current_chat
]

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:

    st.header("⚙️ Controls")

    if st.button("➕ New Chat"):

        new_chat_name = (
            f"Chat {len(st.session_state.chat_sessions)+1}"
        )

        st.session_state.chat_sessions[
            new_chat_name
        ] = []

        st.session_state.current_chat = (
            new_chat_name
        )

        save_chat_history(
            st.session_state.chat_sessions
        )

        st.rerun()

    st.divider()

    st.subheader("🕘 History")

    for chat_name in reversed(
        list(st.session_state.chat_sessions.keys())
    ):

        messages = (
            st.session_state.chat_sessions[
                chat_name
            ]
        )

        preview = chat_name

        for role, msg in messages:

            if role == "user":

                preview = msg[:30] + "..."

                break

        if st.button(
            preview,
            key=chat_name
        ):

            st.session_state.current_chat = (
                chat_name
            )

            st.rerun()

# ---------------------------
# LOAD DATA
# ---------------------------
data_folder = os.path.join(
    os.getcwd(),
    "data"
)

@st.cache_resource(show_spinner=False)
def initialize_rag(data_folder):

    documents = load_documents(
        data_folder
    )

    return setup_rag(documents)

docs, vectorstore, bm25 = initialize_rag(
    data_folder
)

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0
)

# ---------------------------
# CHAT UI
# ---------------------------
current_messages = (
    st.session_state.chat_sessions[
        st.session_state.current_chat
    ]
)

for role, msg in current_messages:

    with st.chat_message(role):

        st.markdown(msg)

# ---------------------------
# USER INPUT
# ---------------------------
query = st.chat_input(
    "Ask your question..."
)

if query:

    st.session_state.chat_sessions[
        current_chat
    ].append(
        ("user", query)
    )

    save_chat_history(
        st.session_state.chat_sessions
    )

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.markdown(
            "⏳ Searching documents..."
        )

        start = time.time()

        relevant_docs = retrieve_chunks(
            query,
            docs,
            bm25,
            vectorstore
        )

        context = "\n\n".join([
            d.page_content
            for d in relevant_docs
        ])

        history = memory.load_memory_variables(
            {}
        )["history"]

        response_text = generate_response(
            query,
            context,
            history,
            llm
        )

        memory.save_context(
            {"input": query},
            {"output": response_text}
        )

        end = time.time()

        final = f"""
{response_text}

---

⏱️ Response Time:
{round(end-start, 2)} sec
"""

        thinking.markdown(final)

        # ---------------------------
        # EVALUATION DASHBOARD
        # ---------------------------

        st.divider()

        st.subheader(
            "📊 Evaluation Dashboard"
        )

        st.write("### ❓ Question")

        st.info(query)

        st.write(
            "### 📚 Retrieved Chunks"
        )

        with st.expander(
            "View Retrieved Context"
        ):

            for i, doc in enumerate(
                relevant_docs
            ):

                st.write(
                    f"Chunk {i+1}"
                )

                st.code(
                    doc.page_content
                )

        st.write(
            "### 🤖 Generated Answer"
        )

        st.success(response_text)

        # Example Ground Truth

        ground_truth = (
            "Expected answer from documents"
        )

        st.write(
            "### ✅ Ground Truth"
        )

        st.info(ground_truth)

        # ---------------------------
        # SIMPLE EVALUATION SCORE
        # ---------------------------

        if (
            response_text.lower()
            ==
            "the information is not available in the uploaded documents."
            ):
            evaluation_score = "Low"
            st.warning(
                "Evaluation Status: No Relevant Context Found"
                )

        elif (
            response_text.lower()
            in
            ground_truth.lower()
            ):
            evaluation_score = "High"
            st.success(
                "Evaluation Status: Correct Answer"
                )

        else:
            evaluation_score = "Medium"
            st.info(
                "Evaluation Status: Partially Correct"
                )
        st.write(
            f"### 📈 Evaluation Score: {evaluation_score}"
        )

        st.session_state.chat_sessions[
            current_chat
        ].append(
            ("assistant", final)
        )

        save_chat_history(
            st.session_state.chat_sessions
        )