def build_prompt(history: str, context: str, query: str) -> str:
    """
    Build the final prompt sent to the LLM.
    """
    return f"""You are a college document assistant.

STRICT RULES:
1. Answer ONLY using the provided context.
2. Do NOT hallucinate or make up information.
3. If the answer is not in the context, say exactly:
   "The information is not available in the uploaded documents."
4. Keep answers concise and factual.

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:"""