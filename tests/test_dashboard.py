import json
from pathlib import Path

import dashboard as lab

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_all_pages_render_with_synthetic_records_without_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGHTLINE_DATA_ROOT", str(tmp_path))
    workspace = tmp_path / "study_data"
    paths = lab.data_paths(workspace)
    # UI fixtures are invented test data, independent of experimental records.
    (workspace / "objects.json").write_text(json.dumps({"sample_object": {
        "ground_truth": "Example test object", "aliases": [], "notes": "Synthetic UI fixture"}}))
    capture = dict(phase="Testing", trial_id="sample_trial", object_id="sample_object",
                   method="uniform", view_number=1, angle=0, image_path="images/sample.jpg",
                   view_budget=5)
    lab.append_csv(paths["captures"], lab.CAPTURE_COLUMNS, capture)
    for model in lab.MODEL_ORDER:
        lab.append_csv(paths["trials"], lab.TRIAL_COLUMNS,
                       dict(capture, model=model, model_id=lab.MODEL_IDS[model],
                            prediction="example object", category="test object",
                            confidence_reported=0.5, evaluation_status="ok",
                            evaluation_id=model, request_attempts=1, total_tokens=20,
                            response_raw='{"prediction":"example object"}',
                            notes="Synthetic UI fixture"))
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=20).run()
    assert not at.exception
    for page in ["Evaluate & retry", "Results & graphs", "Review predictions", "Data & exports", "Methodology"]:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, page
    at.sidebar.radio(key="workspace").set_value("Research").run()
    assert not at.exception
    assert not (workspace / "reviews.json").exists()
