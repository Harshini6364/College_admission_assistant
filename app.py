import time

import streamlit as st
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    DATA_FOLDER
)

from core.loader import load_documents
from core.rag import setup_rag
from core.retriever import retrieve_chunks
from core.llm_handler import generate_response

from components.sidebar import render_sidebar
from components.eval_panel import render_eval_panel

from utils.chat_history import load_chat_history, save_chat_history
from utils.qa_loader import load_qa_pairs, find_ground_truth

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
    st.session_state.chat_sessions = load_chat_history()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = list(
        st.session_state.chat_sessions.keys()
    )[0]

if "memories" not in st.session_state:
    st.session_state.memories = {}

# ---------------------------
# MEMORY (per chat session)
# ---------------------------
current_chat = st.session_state.current_chat

if current_chat not in st.session_state.memories:
    st.session_state.memories[current_chat] = []

memory = st.session_state.memories[current_chat]

# ---------------------------
# LOAD QA PAIRS
# ---------------------------
qa_pairs = load_qa_pairs()

# ---------------------------
# SIDEBAR
# ---------------------------
render_sidebar(st.session_state.chat_sessions)

# ---------------------------
# LOAD RAG PIPELINE (cached)
# ---------------------------
@st.cache_resource(show_spinner="Loading documents...")
def initialize_rag(folder: str):
    documents = load_documents(folder)
    return setup_rag(documents)

docs, vectorstore, bm25 = initialize_rag(DATA_FOLDER)

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)

# ---------------------------
# CHAT HISTORY DISPLAY
# ---------------------------
current_messages = st.session_state.chat_sessions[current_chat]

for role, msg in current_messages:
    with st.chat_message(role):
        st.markdown(msg)

# ---------------------------
# USER INPUT
# ---------------------------
query = st.chat_input("Ask your question...")

if query:

    # Save and display user message
    st.session_state.chat_sessions[current_chat].append(("user", query))
    save_chat_history(st.session_state.chat_sessions)

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        placeholder.markdown("⏳ Searching documents...")

        start = time.time()

        # Retrieve relevant chunks
        relevant_docs = retrieve_chunks(query, docs, bm25, vectorstore)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Load memory
        history = "\n".join(memory)

        # Generate response
        response_text = generate_response(query, context, history, llm)

        # Update memory
        memory.append(f"User: {query}")
        memory.append(f"Assistant: {response_text}")

        elapsed = round(time.time() - start, 2)

        final = f"{response_text}\n\n---\n\n⏱️ Response time: {elapsed} sec"

        placeholder.markdown(final)

        # Match ground truth from QA file
        ground_truth = find_ground_truth(query, qa_pairs)

        # Evaluation dashboard
        render_eval_panel(
            query=query,
            response_text=response_text,
            relevant_docs=relevant_docs,
            ground_truth=ground_truth
        )

        # Save assistant message
        st.session_state.chat_sessions[current_chat].append(("assistant", final))
        save_chat_history(st.session_state.chat_sessions)