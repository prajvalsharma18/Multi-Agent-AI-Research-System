from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_handles_running_pipeline_before_first_completed_step() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path))
    app.session_state["running"] = True

    app.run()

    assert not app.exception
    assert any(
        "Pipeline stage results will appear here as stages complete." in element.value
        for element in app.info
    )
