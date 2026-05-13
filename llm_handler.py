from prompts import build_prompt


def generate_response(
    query,
    context,
    history,
    llm
):

    prompt = build_prompt(
        history,
        context,
        query
    )

    response = llm.invoke(prompt)

    return response.content.strip()