"""LangGraph workflow definition for the research agent."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.state import ResearchState
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.writer import writer_node


def should_continue(state: ResearchState) -> str:
    if state.get("search_results"):
        return "analyst"
    iteration = state.get("iteration", 0)
    if iteration >= state.get("max_iterations", 3):
        return END
    return "researcher"


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_conditional_edges(
        "researcher",
        should_continue,
        {"analyst": "analyst", "researcher": "researcher", END: END},
    )
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


research_graph = build_graph()
