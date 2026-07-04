# AI Research Agent

A LangGraph-powered multi-agent system that researches any topic by orchestrating specialized agents through a stateful graph.

## Agent Graph

```
topic
  └── [Planner] Break into 3-5 subtasks
        └── [Researcher] Web search per subtask (Tavily)
              └── [Analyst] Synthesize findings
                    └── [Writer] Generate final report
```

Each node is a specialized agent with a distinct role. LangGraph manages state, routing, and retry logic between them.

## Features

- **Stateful graph** — typed `ResearchState` flows through all nodes
- **Conditional routing** — retries researcher if results are empty, up to `max_iterations`
- **Tool-calling researcher** — uses Tavily web search with LLM-driven query planning
- **Multi-agent separation** — planner, researcher, analyst, and writer each have distinct prompts and temperature settings
- **LangSmith tracing** — set `LANGCHAIN_TRACING_V2=true` for full observability

## Quickstart

```bash
git clone https://github.com/dhruvr1872/ai-research-agent
cd ai-research-agent
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY and TAVILY_API_KEY (free at tavily.com)

python main.py "The impact of LLMs on data engineering workflows in 2025"
python main.py "RAG vs fine-tuning: when to use each" --output report.md
```

## Stack

| Component | Tech |
|---|---|
| Graph orchestration | LangGraph |
| LLM | GPT-4o-mini (OpenAI) |
| Web search | Tavily |
| State schema | TypedDict + Pydantic |
| Tracing | LangSmith (optional) |

## Graph Design

Uses `conditional_edges` on the researcher node — if search returns empty (e.g. Tavily not configured), retries up to `max_iterations` before terminating gracefully. Makes the system robust to tool failures without crashing.
