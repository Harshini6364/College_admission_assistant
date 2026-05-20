import streamlit as st

from langchain_core.documents import Document
from core.evaluation import score_response


_SCORE_CONFIG = {
    "High":   ("✅ Correct answer",              st.success),
    "Medium": ("⚠️ Partially correct",           st.info),
    "Low":    ("❌ No relevant context found",   st.warning),
}


def render_eval_panel(
    query: str,
    response_text: str,
    relevant_docs: list[Document],
    ground_truth: str = ""
) -> None:
    """
    Render the evaluation dashboard below a chat response.
    """
    st.divider()
    st.subheader("📊 Evaluation Dashboard")

    st.write("### ❓ Question")
    st.info(query)

    st.write("### 📚 Retrieved chunks")
    with st.expander("View retrieved context"):
        for i, doc in enumerate(relevant_docs):
            st.write(f"**Chunk {i + 1}**")
            st.code(doc.page_content)

    st.write("### 🤖 Generated answer")
    st.success(response_text)

    if ground_truth:
        st.write("### ✅ Ground truth")
        st.info(ground_truth)

        score = score_response(response_text, ground_truth)
        label, renderer = _SCORE_CONFIG[score]
        renderer(f"Evaluation status: {label}")
        st.write(f"### 📈 Evaluation score: **{score}**")