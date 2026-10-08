import graph


def test_critic_route_respects_revision_decision_and_limit():
    assert graph._route_after_critic({"needs_revision": False, "revision_count": 0}) == "export"
    assert graph._route_after_critic({"needs_revision": True, "revision_count": 0}) == "revision"
    assert graph._route_after_critic({"needs_revision": True, "revision_count": graph.MAX_REVISION_ITERATIONS}) == "export"


def test_happy_path_runs_each_node_and_exports(monkeypatch):
    visited = []
    def node(name, update):
        def run(state):
            visited.append(name)
            return update
        return run
    monkeypatch.setattr(graph, "search_node", node("search", {"search_results": "s", "ranked_sources": []}))
    monkeypatch.setattr(graph, "reader_node", node("reader", {"reader_summary": "summary"}))
    monkeypatch.setattr(graph, "writer_node", node("writer", {"report": "draft"}))
    monkeypatch.setattr(graph, "critic_node", node("critic", {"needs_revision": False, "final_report": "draft"}))
    monkeypatch.setattr(graph, "revision_node", node("revision", {}))
    monkeypatch.setattr(graph, "export_node", node("export", {"output_paths": {"markdown": "r.md", "pdf": "r.pdf"}}))
    result = graph.build_research_graph().invoke({"topic": "test"})
    assert visited == ["search", "reader", "writer", "critic", "export"]
    assert result["output_paths"]["pdf"] == "r.pdf"


def test_revision_loop_terminates_at_configured_limit(monkeypatch):
    visited = []
    def search(state):
        visited.append("search")
        return {"ranked_sources": []}
    def reader(state):
        visited.append("reader")
        return {"reader_summary": "summary"}
    def writer(state):
        visited.append("writer")
        return {"report": "draft"}
    def critic(state):
        visited.append("critic")
        return {"needs_revision": True, "feedback": "revise"}
    def revision(state):
        visited.append("revision")
        return {"revision_count": state.get("revision_count", 0) + 1}
    def export(state):
        visited.append("export")
        return {"final_report": "report", "output_paths": {}}
    for key, value in [("search_node", search), ("reader_node", reader), ("writer_node", writer),
                       ("critic_node", critic), ("revision_node", revision), ("export_node", export)]:
        monkeypatch.setattr(graph, key, value)
    graph.build_research_graph().invoke({"topic": "test"})
    assert visited.count("revision") == graph.MAX_REVISION_ITERATIONS
    assert visited.count("critic") == graph.MAX_REVISION_ITERATIONS + 1
    assert visited[-1] == "export"


def test_invalid_revision_counter_is_not_repaired_and_routes_to_export():
    assert graph._route_after_critic({"needs_revision": True, "revision_count": 99}) == "export"


def test_real_revision_node_advances_until_revision_cap(monkeypatch):
    visited = []
    revisions = []

    def search(state):
        visited.append("search")
        return {"ranked_sources": []}

    def reader(state):
        visited.append("reader")
        return {"reader_summary": "evidence"}

    def writer(state):
        visited.append("writer")
        return {"report": "draft"}

    def critic(state):
        visited.append("critic")
        return {"feedback": f"review {len(revisions)}", "needs_revision": True}

    def export_node(state):
        visited.append("export")
        return {"final_report": state.get("final_report") or state["report"], "output_paths": {}}

    def revise(stage, key, invoke_fn):
        revisions.append(f"revision {len(revisions) + 1}")
        return revisions[-1]

    monkeypatch.setattr(graph, "search_node", search)
    monkeypatch.setattr(graph, "reader_node", reader)
    monkeypatch.setattr(graph, "writer_node", writer)
    monkeypatch.setattr(graph, "critic_node", critic)
    monkeypatch.setattr(graph, "export_node", export_node)
    monkeypatch.setattr(graph, "_cached_llm", revise)
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)

    result = graph.build_research_graph().invoke({"topic": "test"})

    assert len(revisions) == graph.MAX_REVISION_ITERATIONS
    assert visited.count("critic") == graph.MAX_REVISION_ITERATIONS + 1
    assert visited[-1] == "export"
    assert result["revision_count"] == graph.MAX_REVISION_ITERATIONS
    assert result["final_report"].startswith(revisions[-1])
