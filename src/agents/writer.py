"""Writer agent: produces the final structured research report."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings
from src.state import ResearchState


WRITER_PROMPT = """You are a professional research writer. Using the analysis provided,
write a comprehensive, well-structured research report.

Topic: {topic}

Analysis:
{analysis}

Write a report with:
- Executive summary (2-3 sentences)
- Key findings (bulleted)
- Detailed sections for each major area
- Conclusions and implications
- Limitations of this research"""


def writer_node(state: ResearchState) -> dict:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional research writer."),
        ("human", WRITER_PROMPT),
    ])
    chain = prompt | llm
    result = chain.invoke({"topic": state["topic"], "analysis": state["analysis"]})
    return {"report": result.content}
