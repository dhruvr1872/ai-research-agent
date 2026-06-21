"""Planner agent: breaks a research topic into concrete subtasks."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import settings
from src.state import ResearchState


class ResearchPlan(BaseModel):
    subtasks: list[str]


PLANNER_PROMPT = """You are a research planning expert. Given a topic, break it into 3-5 specific,
searchable subtasks that together will produce a comprehensive research report.

Each subtask should be a concrete question or area to investigate, not a generic instruction.

Topic: {topic}

Return a list of 3-5 specific subtasks."""


def planner_node(state: ResearchState) -> dict:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    ).with_structured_output(ResearchPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research planning expert."),
        ("human", PLANNER_PROMPT),
    ])
    chain = prompt | llm
    result: ResearchPlan = chain.invoke({"topic": state["topic"]})

    return {
        "plan": result.subtasks,
        "iteration": state.get("iteration", 0),
    }
