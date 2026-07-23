from graph import SEPARATOR, research_graph


def run_research_pipeline(topic: str) -> dict:
    """Run the LangGraph multi-agent research pipeline."""
    try:
        result = research_graph.invoke({"topic": topic})
        return dict(result)
    except Exception as e:
        print(f"\nPipeline Failed!\n{e}")
        return {"topic": topic, "error": str(e)}


if __name__ == "__main__":
    print(SEPARATOR)
    print("Multi-Agent Research System")
    print(SEPARATOR)

    topic = input("\nEnter a research topic: ").strip()
    run_research_pipeline(topic)
