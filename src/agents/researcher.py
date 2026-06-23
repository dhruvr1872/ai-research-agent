"""Researcher agent: executes search queries for each subtask."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import settings
from src.state import ResearchState
from src.tools.search import web_search


RESEARCHER_SYSTEM = """You are a research agent with access to a web search tool.
For each subtask you receive, search for relevant, current information.
Be thorough — search multiple angles if needed. Summarize what you find."""


def researcher_node(state: ResearchState) -> dict:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    ).bind_tools([web_search])

    all_results: list[dict] = []
    messages = [{"role": "system", "content": RESEARCHER_SYSTEM}]

    for subtask in state["plan"]:
        messages.append({"role": "user", "content": f"Research this subtask: {subtask}"})
        response = llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "web_search":
                    search_result = web_search.invoke(tc["args"])
                    all_results.append({
                        "subtask": subtask,
                        "query": tc["args"].get("query", subtask),
                        "content": search_result,
                    })
        else:
            all_results.append({
                "subtask": subtask,
                "query": subtask,
                "content": response.content,
            })

    return {"search_results": all_results}
