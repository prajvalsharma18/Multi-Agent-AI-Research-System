"""Streamlit frontend for the multi-agent research pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_STEPS = [
    ("search", "Search Agent", "Tavily web search + source ranking"),
    ("reader", "Reader Agent", "Batch scrape URLs (Trafilatura) + summarize"),
    ("writer", "Report Writer", "Draft structured research report"),
    ("critic", "Research Critic", "OpenAI score & review the draft"),
    ("revision", "Writer Revision", "Reflection loop → final report"),
    ("export", "Export", "Save Markdown + PDF"),
]

STEP_KEYS = {
    "search": ["search_results", "ranked_sources"],
    "reader": ["reader_summary"],
    "writer": ["report"],
    "critic": ["feedback"],
    "revision": ["final_report"],
    "export": ["output_paths"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_keys_ok() -> tuple[bool, list[str]]:
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    return len(missing) == 0, missing


def _init_state() -> None:
    defaults = {
        "running": False,
        "topic": "",
        "topic_input": "",
        "completed_steps": [],
        "pipeline_state": {},
        "error": None,
        "active_step": None,
        "pipeline_requested": False,
        "metrics_summary": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _score_color(score: int) -> str:
    if score >= 9:
        return "#0d9488"
    if score >= 7:
        return "#2563eb"
    if score >= 5:
        return "#d97706"
    return "#dc2626"


def _render_pipeline_progress(completed: list[str], active: str | None) -> None:
    cols = st.columns(len(PIPELINE_STEPS))
    for col, (node_id, label, _) in zip(cols, PIPELINE_STEPS):
        if node_id in completed:
            status = "✅"
            style = "background:#ecfdf5;border:1px solid #6ee7b7;"
        elif node_id == active:
            status = "⏳"
            style = "background:#eff6ff;border:1px solid #93c5fd;"
        else:
            status = "○"
            style = "background:#f8fafc;border:1px solid #e2e8f0;"

        col.markdown(
            f"""
            <div style="{style}border-radius:10px;padding:10px 8px;text-align:center;min-height:72px;">
              <div style="font-size:1.1rem;">{status}</div>
              <div style="font-size:0.78rem;font-weight:600;color:#0f172a;margin-top:4px;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_ranked_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources passed the authority threshold (7/10).")
        return

    for i, src in enumerate(sources, start=1):
        score = int(src.get("score", 0))
        color = _score_color(score)
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{i}. {src.get('title', 'Untitled')}**")
                st.caption(src.get("url", ""))
                snippet = (src.get("snippet") or "").strip()
                if snippet:
                    st.write(snippet[:280] + ("…" if len(snippet) > 280 else ""))
            with c2:
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:8px;">
                      <div style="font-size:1.4rem;font-weight:700;color:{color};">{score}/10</div>
                      <div style="font-size:0.75rem;color:#64748b;">quality</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_step_output(node_id: str, state: dict) -> None:
    if node_id == "search":
        sources = state.get("ranked_sources") or []
        st.subheader("Ranked sources")
        st.caption(f"{len(sources)} source(s) passed the authority threshold (7/10); ordered by composite score")
        _render_ranked_sources(sources)

        with st.expander("Raw search output", expanded=False):
            st.text(state.get("search_results") or "—")

    elif node_id == "reader":
        summary = state.get("reader_summary") or "—"
        st.markdown(summary)
        quality = state.get("quality_metrics") or {}
        sources = quality.get("sources") or {}
        evidence = quality.get("evidence") or {}
        if quality:
            st.caption(
                f"Evidence: {evidence.get('supported_claims', 0)}/{evidence.get('total_claims', 0)} claims supported · "
                f"Sources: {sources.get('unique_domains', 0)} domains / {sources.get('unique_sources', 0)} URLs"
            )

    elif node_id == "writer":
        report = state.get("report") or "—"
        st.markdown(report)

    elif node_id == "critic":
        feedback = state.get("feedback") or "—"
        st.markdown(feedback)
        critique = state.get("critique") or {}
        if critique:
            st.metric("Research quality score", f"{critique.get('overall_score', 0)}/10")

    elif node_id == "revision":
        final = state.get("final_report") or "—"
        st.markdown(final)

    elif node_id == "export":
        paths = state.get("output_paths") or {}
        md_path = paths.get("markdown")
        pdf_path = paths.get("pdf")

        c1, c2 = st.columns(2)
        with c1:
            if md_path and Path(md_path).exists():
                st.success(f"Markdown saved\n\n`{md_path}`")
                st.download_button(
                    "Download Markdown",
                    data=Path(md_path).read_bytes(),
                    file_name=Path(md_path).name,
                    mime="text/markdown",
                    key="dl_md",
                )
            else:
                st.warning("Markdown file not found.")
        with c2:
            if pdf_path and Path(pdf_path).exists():
                st.success(f"PDF saved\n\n`{pdf_path}`")
                st.download_button(
                    "Download PDF",
                    data=Path(pdf_path).read_bytes(),
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    key="dl_pdf",
                )
            else:
                st.warning("PDF file not found.")


def run_pipeline(topic: str, *, resume: bool = True) -> None:
    """Stream LangGraph nodes and update the UI after each step."""
    from cache import load_initial_state
    from graph import MAX_REVISION_ITERATIONS, research_graph
    from llm_retry import LLMRequestError
    from metrics import get_metrics, reset_metrics

    reset_metrics()
    initial = load_initial_state(topic) if resume else {"topic": topic}

    st.session_state.running = True
    st.session_state.topic = topic
    st.session_state.completed_steps = []
    st.session_state.pipeline_state = dict(initial)
    st.session_state.error = None
    st.session_state.active_step = "search"
    st.session_state.metrics_summary = ""

    progress = st.progress(0, text="Starting pipeline…")
    status = st.empty()
    live = st.empty()

    try:
        # The critic can run after each revision, so reserve progress for the
        # bounded revision loop as well as the linear path.
        total = len(PIPELINE_STEPS) + MAX_REVISION_ITERATIONS + 1
        for i, event in enumerate(research_graph.stream(initial), start=1):
            node_id, partial = next(iter(event.items()))
            st.session_state.pipeline_state.update(partial or {})
            st.session_state.completed_steps = list(
                dict.fromkeys([*st.session_state.completed_steps, node_id])
            )

            if node_id == "revision":
                next_step = "critic"
            elif node_id == "critic":
                can_revise = (
                    st.session_state.pipeline_state.get("needs_revision", False)
                    and st.session_state.pipeline_state.get("revision_count", 0) < MAX_REVISION_ITERATIONS
                )
                next_step = "revision" if can_revise else "export"
            else:
                order = [step[0] for step in PIPELINE_STEPS]
                next_step = order[order.index(node_id) + 1] if node_id in order and order.index(node_id) + 1 < len(order) else None
            st.session_state.active_step = next_step

            label = next((lbl for nid, lbl, _ in PIPELINE_STEPS if nid == node_id), node_id)
            progress.progress(min(i / total, 1.0), text=f"Completed: {label}")
            status.info(f"Finished **{label}** ({i}/{total})")

            with live.container():
                _render_pipeline_progress(
                    st.session_state.completed_steps,
                    st.session_state.active_step,
                )

        st.session_state.active_step = None
        st.session_state.metrics_summary = get_metrics().summary()
        progress.progress(1.0, text="Pipeline complete")
        status.success("Research pipeline finished.")

    except LLMRequestError as exc:
        st.session_state.error = f"OpenAI request failed at stage **{exc.step or 'unknown'}**. Cached stages are saved — click **Resume** to continue later. Details: {exc}"
        st.session_state.metrics_summary = get_metrics().summary()
        status.error(st.session_state.error)
    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.metrics_summary = get_metrics().summary()
        status.error(f"Pipeline failed: {st.session_state.error}")
    finally:
        st.session_state.running = False
        st.session_state.pipeline_requested = False


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_init_state()

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 1200px; }
      div[data-testid="stSidebar"] { background: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Research Pipeline")
    st.caption("LangGraph · OpenAI · Tavily · Trafilatura")

    st.markdown("### Flow")
    for i, (_, label, desc) in enumerate(PIPELINE_STEPS, start=1):
        st.markdown(f"**{i}. {label}**  \n<span style='color:#64748b;font-size:0.85rem;'>{desc}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Source scoring")
    st.markdown(
        """
        | Type | Score |
        |------|-------|
        | Government | 10 |
        | Academic / major news | 8–9 |
        | Wikipedia | 7 |
        | Unknown blog | 5 |
        | Social media | 2–3 |
        """
    )
    st.caption("Only sources with authority ≥ 7/10 are scraped and cited; composite scores rank those accepted sources.")

    st.divider()
    st.markdown("### Caching")
    st.caption(
        "Search, scrape, LLM stages, and checkpoints are cached under `.cache/`. "
        "Rerun the same topic to reuse results and resume after quota errors."
    )

    ok, missing = _api_keys_ok()
    st.divider()
    if ok:
        st.success("API keys loaded")
    else:
        st.error(f"Missing: {', '.join(missing)}")
        st.caption("Add them to a `.env` file in the project root.")

st.title("Multi-Agent Research Assistant")
st.markdown(
    "Enter a topic to run the full pipeline: **Search → Reader → Writer → Critic → Revision → Export**."
)

topic = st.text_input(
    "Research topic",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g. India space mission 2026",
    disabled=st.session_state.running,
    key="topic_input_widget",
)

run_col, resume_col, clear_col = st.columns([1, 1, 4])
with run_col:
    run_clicked = st.button(
        "Run research",
        type="primary",
        disabled=st.session_state.running or not ok,
        use_container_width=True,
    )
with resume_col:
    resume_clicked = st.button(
        "Resume",
        disabled=st.session_state.running or not ok or not st.session_state.topic,
        use_container_width=True,
        help="Resume the last topic from saved checkpoint",
    )
with clear_col:
    if st.button("Clear results", disabled=st.session_state.running):
        from cache import clear_checkpoint

        if st.session_state.topic:
            clear_checkpoint(st.session_state.topic)
        st.session_state.completed_steps = []
        st.session_state.pipeline_state = {}
        st.session_state.error = None
        st.session_state.active_step = None
        st.session_state.topic = ""
        st.session_state.topic_input = ""
        st.session_state.metrics_summary = ""
        st.rerun()

if run_clicked and not st.session_state.running:
    cleaned = topic.strip()
    if not cleaned:
        st.warning("Please enter a research topic.")
    else:
        st.session_state.topic = cleaned
        st.session_state.topic_input = cleaned
        st.session_state.pipeline_requested = True

if resume_clicked and not st.session_state.running and st.session_state.topic:
    st.session_state.pipeline_requested = True

if st.session_state.pipeline_requested and not st.session_state.running:
    run_pipeline(st.session_state.topic, resume=True)

# ----- Results -----
state = st.session_state.pipeline_state
completed = st.session_state.completed_steps

if st.session_state.error and not completed:
    st.error(st.session_state.error)

if completed or st.session_state.running:
    st.divider()
    st.subheader("Pipeline progress")
    _render_pipeline_progress(completed, st.session_state.active_step)

    if state.get("topic"):
        st.caption(f"Topic: **{state['topic']}**")

    if st.session_state.metrics_summary:
        st.caption(f"Metrics: {st.session_state.metrics_summary}")

if completed:
    st.divider()
    labels = {nid: label for nid, label, _ in PIPELINE_STEPS}
    tab_labels = [labels[n] for n in completed if n in labels]
    tabs = st.tabs(tab_labels)

    for tab, node_id in zip(tabs, [n for n in completed if n in labels]):
        with tab:
            desc = next(d for nid, _, d in PIPELINE_STEPS if nid == node_id)
            st.caption(desc)
            _render_step_output(node_id, state)

    # Final report highlight when fully done
    if "revision" in completed and state.get("final_report"):
        st.divider()
        st.subheader("Final report")
        st.markdown(state["final_report"])

    if st.session_state.error:
        st.error(st.session_state.error)
elif not st.session_state.running:
    st.info("Results for every pipeline stage will appear here after you run a topic.")
