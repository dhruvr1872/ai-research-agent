"""Shared state schema for the LangGraph research workflow."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """State passed between all nodes in the research graph."""
    topic: str
    plan: list[str]
    search_results: list[dict]
    analysis: str
    report: str
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
