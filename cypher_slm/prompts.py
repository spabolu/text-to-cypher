from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are an expert Neo4j engineer.
    Generate a valid Cypher query for the given graph schema and user request.
    Return only raw Cypher.
    Do not add markdown fences, commentary, or explanations.
    """
).strip()


def render_user_prompt(schema_text: str, question: str) -> str:
    return dedent(
        f"""
        Graph schema:
        {schema_text}

        Task:
        Write a Cypher query that answers the following question.

        Question: {question}
        """
    ).strip()


def build_messages(schema_text: str, question: str, cypher: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(schema_text, question)},
        {"role": "assistant", "content": cypher},
    ]
