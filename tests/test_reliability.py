import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

import camera
import dashboard as lab
import motor_controller as motor


def capture_records(paths, count=3, method="uniform"):
    records = []
    for n in range(1, count + 1):
        image = paths["captures"].parent / "images" / f"{n}.jpg"
        image.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(image), np.zeros((40, 60, 3), dtype=np.uint8))
        row = dict.fromkeys(lab.CAPTURE_COLUMNS, "")
        row.update(trial_id="trial", object_id="withheld_identity", method=method,
                   view_number=str(n), image_path=f"images/{n}.jpg", sha256=lab.image_checksum(image))
        lab.append_csv(paths["captures"], lab.CAPTURE_COLUMNS, row)
        records.append(row)
    return pd.DataFrame(records)


def valid_response():
    return json.dumps(dict(prediction="visible toy", category="toy", brand=None,
                           product_family=None, model=None, evidence="wheels",
                           uncertainties="exact variant", confidence=0.8))


@pytest.mark.parametrize("raw", ["not json", '{"prediction":["toy"]}',
                                '{"prediction":"toy"}', valid_response().replace('0.8', 'true')])
def test_incomplete_or_wrong_typed_json_is_not_structured_success(raw):
    assert lab.parse_recognition(raw)["evaluation_status"] == "unstructured"


def test_valid_response_and_out_of_range_confidence():
    assert lab.parse_recognition(valid_response())["evaluation_status"] == "ok"
    assert lab.parse_recognition(valid_response().replace('0.8', '1.8'))["evaluation_status"] == "unstructured"


def test_corrupt_input_stops_instead_of_becoming_empty_workspace(tmp_path):
    path = tmp_path / "objects.json"
    path.write_text("{unfinished", encoding="utf-8")
    with pytest.raises(RuntimeError):
        lab.load_json(path, {})
    assert path.read_text() == "{unfinished"
    csv = tmp_path / "captures.csv"
    csv.write_text('trial_id,view_number\na,"unfinished', encoding="utf-8")
    with pytest.raises(RuntimeError):
        lab.read_csv(csv, lab.CAPTURE_COLUMNS)


def test_portable_path_prefers_current_workspace_and_rejects_escape(tmp_path):
    paths = lab.data_paths(tmp_path)
    capture_records(paths, 1)
    assert lab.resolve_capture_path(r"C:\old\study_data\images\1.jpg", paths) == tmp_path / "images" / "1.jpg"
    assert lab.resolve_capture_path("images/1.jpg", paths) == tmp_path / "images" / "1.jpg"
    with pytest.raises(FileNotFoundError):
        lab.resolve_capture_path("images/../../elsewhere.jpg", paths)


def test_cumulative_checkpoints_stop_at_gap_and_do_not_add_transfer(tmp_path):
    paths = lab.data_paths(tmp_path)
    captures = capture_records(paths, 3, "qwen_guided")
    captures = captures[captures.view_number.ne("2")]
    jobs = lab.eligible_checkpoints(captures, pd.DataFrame(columns=lab.TRIAL_COLUMNS), lab.MODEL_ORDER)
    assert [(model, len(prefix)) for model, prefix in jobs] == [("Qwen local", 1)]


def test_resume_preserves_successes_and_retries_failures_only(tmp_path):
    paths = lab.data_paths(tmp_path)
    captures = capture_records(paths, 3)
    old = pd.DataFrame([dict(trial_id="trial", view_number="1", model="Qwen local", evaluation_status="ok"),
                        dict(trial_id="trial", view_number="2", model="Qwen local", evaluation_status="error")])
    jobs = lab.eligible_checkpoints(captures, old, ["Qwen local"], retry_errors=True)
    assert [len(prefix) for _, prefix in jobs] == [2, 3]


def test_api_retry_records_each_attempt_and_closes_client(tmp_path, monkeypatch):
    paths = lab.data_paths(tmp_path)
    capture_records(paths, 1)
    client = Mock()
    monkeypatch.setattr(lab, "init_clients", lambda _: ({"Qwen local": client}, {}))
    monkeypatch.setattr(lab.time, "sleep", lambda _: None)
    call = Mock(side_effect=[TimeoutError("timeout"), (valid_response(), (10, 20, 30))])
    monkeypatch.setattr(lab, "call_model", call)
    counts, n = lab.evaluate_saved(paths, "Testing", ["Qwen local"], True, [], Mock())
    assert n == 1 and counts["ok"] == 1 and call.call_count == 2
    attempts = lab.read_csv(paths["evaluation_attempts"], lab.EVALUATION_ATTEMPT_COLUMNS)
    result = lab.read_csv(paths["trials"], lab.TRIAL_COLUMNS).iloc[0]
    assert attempts.status.tolist() == ["error", "ok"]
    assert set(attempts.evaluation_id) == {result.evaluation_id}
    assert result.request_attempts == "2"
    client.close.assert_called_once()


def test_changed_original_is_not_sent_to_model(tmp_path, monkeypatch):
    paths = lab.data_paths(tmp_path)
    capture_records(paths, 1)
    (tmp_path / "images" / "1.jpg").write_bytes(b"changed")
    client = Mock()
    monkeypatch.setattr(lab, "init_clients", lambda _: ({"Qwen local": client}, {}))
    inference = Mock()
    monkeypatch.setattr(lab, "call_model", inference)
    counts, _ = lab.evaluate_saved(paths, "Testing", ["Qwen local"], True, [], Mock())
    assert counts["error"] == 1
    inference.assert_not_called()
    assert (tmp_path / "images" / "1.jpg").read_bytes() == b"changed"


def test_request_contains_only_current_photos_and_generic_prompt():
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=valid_response()))], usage=None))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    lab.call_vision("Qwen local", [b"photo1", b"photo2"], {"Qwen local": client}, lab.RECOGNITION_PROMPT)
    request = create.call_args.kwargs
    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    assert len(request["messages"][0]["content"]) == 3
    assert "withheld_identity" not in str(request)
    assert "object_id" not in str(request)


def test_equal_graph_points_are_never_displaced():
    df = pd.DataFrame([dict(model="Qwen local", method=m, view_number=n, score=0.5)
                       for m in ["uniform", "random", "qwen_guided"] for n in [1, 2]])
    chart = lab.graph_panel(df, "score", "Score", "Qwen local").to_dict()
    data = next(iter(chart["datasets"].values()))
    assert all(row["display_y"] == row["score"] and row["display_x"] == row["view_number"] for row in data)


def test_camera_open_failure_releases_handle(monkeypatch):
    cap = Mock()
    cap.isOpened.return_value = False
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *_: cap)
    with pytest.raises(RuntimeError):
        camera.Camera(3)
    cap.release.assert_called_once()


def test_serial_boot_failure_closes_handle(monkeypatch):
    port = Mock()
    type(port).in_waiting = property(lambda _: (_ for _ in ()).throw(OSError("disconnected")))
    monkeypatch.setattr(motor.serial, "Serial", lambda *a, **k: port)
    monkeypatch.setattr(motor.time, "sleep", lambda _: None)
    with pytest.raises(OSError):
        motor.connect("unused")
    port.close.assert_called_once()


def test_lost_rotation_ack_does_not_trigger_another_move(tmp_path, monkeypatch):
    paths = lab.data_paths(tmp_path)
    port = Mock()
    monkeypatch.setattr(motor, "connect", lambda: port)
    monkeypatch.setattr(motor, "set_status", Mock())
    rotation = Mock(side_effect=TimeoutError("DONE lost after possible movement"))
    monkeypatch.setattr(motor, "rotate_to", rotation)
    fake_camera = Mock()
    fake_camera.capture.side_effect = lambda path: cv2.imwrite(path, np.zeros((40, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(camera, "Camera", lambda **_: fake_camera)
    with pytest.raises(TimeoutError):
        lab.capture_method(paths, "Testing", "object", "uniform", 3, 3, "batch", Mock())
    rotation.assert_called_once_with(port, 0, 120)
    fake_camera.close.assert_called_once()
    port.close.assert_called_once()
    saved = lab.read_csv(paths["captures"], lab.CAPTURE_COLUMNS)
    assert len(saved) == 1 and not Path(saved.iloc[0].image_path).is_absolute()


def test_selector_rejects_repeated_angle_and_research_never_falls_back(tmp_path, monkeypatch):
    paths = lab.data_paths(tmp_path)
    capture_records(paths, 1)
    monkeypatch.setattr(lab.time, "sleep", lambda _: None)
    monkeypatch.setattr(lab, "call_vision", lambda *a, **k: ('{"angle":0,"reason":"repeat"}', (1,2,3)))
    with pytest.raises(RuntimeError, match="trial stopped"):
        lab.select_angle(paths, dict(phase="Research", batch_id="b", object_id="o", trial_id="t", method="qwen_guided"),
                         "Qwen local", [0], [tmp_path/"images"/"1.jpg"], {}, Mock())
    assert len(lab.read_csv(paths["selectors"], lab.SELECTOR_COLUMNS)) == 3


def test_preprocessing_preserves_original_and_fingerprint(tmp_path):
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), np.zeros((1600, 3000, 3), dtype=np.uint8))
    original = path.read_bytes()
    blobs, fingerprint = lab.prepared_prefix([path])
    assert path.read_bytes() == original
    image = cv2.imdecode(np.frombuffer(blobs[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[0] <= 1440 and image.shape[1] <= 2560
    assert lab.prepared_prefix([path])[1] == fingerprint
