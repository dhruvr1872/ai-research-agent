"""Analyst agent: synthesizes search results into a coherent analysis."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings
from src.state import ResearchState


ANALYST_PROMPT = """You are a research analyst. Synthesize the search results into a structured analysis.

Topic: {topic}

Research Plan:
{plan}

Search Results:
{results}

Write a structured analysis that:
1. Identifies key findings across all subtasks
2. Notes contradictions or gaps
3. Highlights the most important insights
4. Identifies what is still unknown or uncertain"""


def analyst_node(state: ResearchState) -> dict:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.2,
    )

    results_text = "\n\n".join(
        f"Subtask: {r['subtask']}\nFindings:\n{r['content']}"
        for r in state["search_results"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a rigorous research analyst."),
        ("human", ANALYST_PROMPT),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "topic": state["topic"],
        "plan": "\n".join(f"- {t}" for t in state["plan"]),
        "results": results_text,
    })

    return {"analysis": result.content}
