#!/usr/bin/env python3
"""Run the AI research agent."""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Research Agent")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--output", help="Save report to file")
    args = parser.parse_args()

    from src.graph import research_graph

    print(f"\nResearching: {args.topic}\n{'='*60}")

    final_state = research_graph.invoke({
        "topic": args.topic,
        "plan": [],
        "search_results": [],
        "analysis": "",
        "report": "",
        "messages": [],
        "iteration": 0,
        "max_iterations": args.max_iter,
    })

    print("\n=== RESEARCH PLAN ===")
    for i, task in enumerate(final_state.get("plan", []), 1):
        print(f"  {i}. {task}")

    print("\n=== REPORT ===")
    report = final_state.get("report", "No report generated.")
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# Research Report: {args.topic}\n\n{report}")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
