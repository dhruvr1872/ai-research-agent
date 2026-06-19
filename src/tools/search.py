"""Web search tool using Tavily."""
from __future__ import annotations

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from src.config import settings


def get_search_tool() -> TavilySearchResults:
    return TavilySearchResults(
        max_results=settings.max_search_results,
        api_key=settings.tavily_api_key,
        include_answer=True,
        include_raw_content=False,
    )


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic."""
    tool_instance = get_search_tool()
    results = tool_instance.invoke({"query": query})
    if isinstance(results, list):
        return "\n\n".join(
            f"[{r.get('title', 'Result')}]\n{r.get('content', '')}"
            for r in results
        )
    return str(results)
