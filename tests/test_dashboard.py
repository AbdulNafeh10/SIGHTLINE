import json
from pathlib import Path
import shutil

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_all_pages_render_with_saved_evidence_without_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGHTLINE_DATA_ROOT", str(tmp_path))
    workspace = tmp_path / "study_data"
    workspace.mkdir()
    for filename in ["captures.csv", "trials.csv", "selector_calls.csv", "evaluation_attempts.csv"]:
        shutil.copyfile(ROOT / "evidence" / filename, workspace / filename)
    (workspace / "objects.json").write_text(json.dumps({"matchbox_2022_infiniti_qx80": {
        "ground_truth": "Matchbox 2022 Infiniti QX80", "aliases": [], "notes": ""}}))
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=20).run()
    assert not at.exception
    for page in ["Evaluate & retry", "Results & graphs", "Review predictions", "Data & exports", "Methodology"]:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, page
    at.sidebar.radio(key="workspace").set_value("Research").run()
    assert not at.exception
    assert not (workspace / "reviews.json").exists()
