"""CLI entry point for the multi-agent research pipeline."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def run_research_pipeline(topic: str, *, resume: bool = True) -> dict:
    """Run the LangGraph multi-agent research pipeline."""
    from cache import clear_checkpoint, load_initial_state
    from graph import SEPARATOR, research_graph
    from llm_retry import LLMRequestError
    from metrics import get_metrics, reset_metrics

    topic = topic.strip()
    if not topic:
        raise ValueError("Research topic cannot be empty.")

    reset_metrics()
    if resume:
        initial = load_initial_state(topic)
    else:
        clear_checkpoint(topic)
        initial = {"topic": topic}

    print(f"\n{SEPARATOR}")
    print(f"Topic: {topic}")
    if resume and initial.keys() - {"topic"}:
        print("Resuming from saved checkpoint (cached stages will be reused).")
    print(SEPARATOR)

    try:
        result = research_graph.invoke(initial)
        state = dict(result)

        paths = state.get("output_paths") or {}
        if paths.get("markdown"):
            print(f"\nMarkdown saved: {paths['markdown']}")
        if paths.get("pdf"):
            print(f"PDF saved:      {paths['pdf']}")

        print(f"\n{get_metrics().summary()}")
        return state
    except LLMRequestError as e:
        print(f"\nPipeline paused at stage: {e.step or 'unknown'}")
        print(str(e))
        print("\nRerun the same topic to resume from the last successful stage.")
        return {"topic": topic, "error": str(e), "failed_stage": e.step}
    except Exception as e:
        print(f"\nPipeline Failed!\n{e}")
        return {"topic": topic, "error": str(e)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advanced Multi-Agent Research Assistant (LangGraph + OpenAI + Tavily)",
        epilog='Example: python pipeline.py "India space mission 2026"',
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint and start from scratch (stage caches still apply)",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear saved checkpoint for the topic before running",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    topic = args.topic or input("\nEnter a research topic: ").strip()

    if not topic:
        print("Error: no topic provided.", file=sys.stderr)
        sys.exit(1)

    if args.clear_checkpoint:
        from cache import clear_checkpoint
        clear_checkpoint(topic)
        print(f"Cleared checkpoint for: {topic}")

    outcome = run_research_pipeline(topic, resume=not args.fresh)
    sys.exit(1 if outcome.get("error") else 0)
