"""SIGHTLINE Research Lab 4.1 — fine-grained prompts, independent scans, audit logs.

Run: python -m streamlit run dashboard.py
Dependencies: streamlit, pandas, altair, openai, google-genai, opencv-python.
Hardware modules (unchanged): motor_controller.py, camera.py.
The viewpoint selector is now integrated for all three recognition models.

Three fixed evaluators: Qwen3-VL-4B-Instruct (local), Gemini 3.5 Flash Lite
(Google), Nemotron 3.5 Super VL (NVIDIA). No other providers.

Data from previous dashboard releases is maintained in study_data/ (Testing).
A separate final_study_data/ folder is used for a clean Research run.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import cv2
import pandas as pd
import streamlit as st
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
PHASE_ROOTS = {"Testing": ROOT / "study_data", "Research": ROOT / "final_study_data"}
LOCAL_ENDPOINT = "http://127.0.0.1:9931/v1"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
MODEL_IDS = {
    "Qwen local": "Qwen3-VL-4B-Instruct",
    "Gemini cloud": "gemini-3.5-flash-lite",
    "NVIDIA Nemotron": "nvidia/nemotron-3.5-super-vl-120b-a12b",
}
MODEL_ORDER = list(MODEL_IDS)
# Each recognizer selects its OWN images for its own guided scan. Uniform/random
# are shared baselines. Historical "ai_guided" rows are still readable and shown
# explicitly as old Qwen-guided trials; they are never relabeled as new trials.
GUIDED_METHODS = {
    "Qwen local": "qwen_guided",
    "Gemini cloud": "gemini_guided",
    "NVIDIA Nemotron": "nvidia_guided",
}
GUIDED_OWNERS = {method: model for model, method in GUIDED_METHODS.items()}
METHODS = ("uniform", "random", *GUIDED_METHODS.values())
METHOD_NAMES = {
    "uniform": "Uniform", "random": "Random",
    "qwen_guided": "Qwen-guided",
    "gemini_guided": "Gemini-guided",
    "nvidia_guided": "NVIDIA-guided",
    "ai_guided": "Legacy Qwen-guided",
}
CANDIDATES = list(range(0, 360, 5))
MAX_WIDTH, MAX_HEIGHT, JPEG_QUALITY = 2560, 1440, 90
PROMPT_VERSION = "sightline-recognition-v5"
SELECTOR_PROMPT_VERSION = "sightline-view-selection-v5"
# Shared verbatim between all three models and both kinds of prompt.
# This describes the apparatus, NEVER the object or its correct identity.
PLATFORM_CONTEXT = (
    "The apparatus is a large, flat, circular matte-black 3D-printed rotating disc "
    "approximately 150 mm across. Its surface has a fine grainy, slightly speckled "
    "texture, and the circle may look elliptical due to the fixed camera's perspective. "
    "Small mechanical tolerances can cause a slight tilt as it rotates. "
    "The disc, its shadows, edges and changing tilt are NOT features of the target object. "
)

# Per-model pacing applies to BOTH angle selection and recognition. These are
# minimum start-to-start intervals, not guarantees of provider quota compliance.
REQUEST_GAP = {"Gemini cloud": 4.2, "NVIDIA Nemotron": 2.1}
LAST_REQUEST_START = {}
RECOGNITION_PROMPT = (
    "You are independently identifying ONE physical object from photographs captured "
    "at different angles on the same turntable. "
    + PLATFORM_CONTEXT +
    "Identify only the object RESTING on the disc; ignore the disc, equipment and background. "
    "Use every supplied photograph, but no prior answers, trial history or ground truth. "
    "Describe the most specific identity actually supported by visible evidence. "
    "Inspect distinctive shape, color, surface details, writing, logos, connectors, "
    "clothing, and other identifying features when present. "
    "Do not stop at the generic category if evidence supports a subtype or family, "
    "but do not guess a specific brand, function, series or model without evidence. "
    "Do not misinterpret turntable features as part of the object. "
    "Return exactly one valid JSON object without Markdown, containing: "
    '"prediction" (short specific identity), "category" (broad type), '
    '"brand" (string or null), "product_family" (string or null), '
    '"model" (exact model or null), "evidence" (directly visible details), '
    '"uncertainties" (what the images cannot establish), '
    '"confidence" (subjective certainty number 0 to 1, or null). '
    "Use null for unknown identity fields. Self-reported confidence is not calibrated."
)

CAPTURE_COLUMNS = [
    "timestamp", "phase", "batch_id", "object_id", "trial_id", "method", "view_number",
    "angle", "image_path", "sha256", "photo_width", "photo_height", "selection_source", "selection_reason",
    "selection_seconds", "capture_seconds", "camera_index", "view_budget", "random_seed",
    "selection_model", "selection_model_id", "selection_status", "selection_attempts",
    "selection_prompt_tokens", "selection_completion_tokens", "selection_total_tokens",
    "selection_input_sha256", "selection_prompt_version", "selection_raw",
]
SELECTOR_COLUMNS = [
    "timestamp", "phase", "batch_id", "object_id", "trial_id", "method",
    "model", "model_id", "view_number", "observed_angles", "candidate_count",
    "request_attempt", "status", "suggested_angle", "reason", "response_raw",
    "error_type", "error_message", "duration_seconds", "prompt_tokens",
    "completion_tokens", "total_tokens", "input_sha256", "selector_prompt_version",
]
TRIAL_COLUMNS = [
    "timestamp", "phase", "batch_id", "object_id", "trial_id", "method", "model", "model_id",
    "view_number", "view_count", "angle", "image_path", "selection_source",
    "selection_reason", "selection_seconds", "capture_seconds", "camera_index",
    "view_budget", "prediction", "category", "brand", "product_family", "product_model", "evidence", "uncertainties",
    "confidence_reported", "correct", "evaluation_status", "error_type", "error_message",
    "evaluation_seconds", "request_attempts", "prompt_tokens", "completion_tokens",
    "total_tokens", "input_max_width", "input_max_height", "jpeg_quality",
    "input_sha256", "recognition_prompt_version", "response_raw", "notes", "evaluation_id",
]
EVALUATION_ATTEMPT_COLUMNS = [
    "timestamp", "phase", "batch_id", "object_id", "trial_id", "method",
    "model", "model_id", "view_number", "attempt", "status",
    "duration_seconds", "input_sha256", "recognition_prompt_version",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "error_type", "error_message", "response_raw",
]
# Legacy predictions.csv is left intact; trials.csv stores the final result per evaluation.
# evaluation_attempts.csv additionally saves every individual API attempt.


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def data_paths(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    return {name: data_dir / file for name, file in (
        ("objects", "objects.json"), ("captures", "captures.csv"),
        ("trials", "trials.csv"), ("reviews", "reviews.json"),
        ("selectors", "selector_calls.csv"),
        ("batches", "batches.jsonl"),
        ("evaluation_attempts", "evaluation_attempts.csv"))}


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_csv(path, columns, row):
    """One durable append per image/evaluation; old CSV schemas auto-upgrade with backup."""
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8-sig", newline="") as f:
            old_fields = csv.DictReader(f).fieldnames or []
        missing = [c for c in columns if c not in old_fields]
        if missing:
            original = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
            backup = path.with_name(path.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            shutil.copy2(path, backup)
            merged = list(dict.fromkeys(old_fields + missing))
            tmp = path.with_name(path.name + ".migrating")
            with tmp.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(original)
            os.replace(tmp, path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        cols = columns
        if exists:
            with path.open(encoding="utf-8-sig", newline="") as src:
                cols = csv.DictReader(src).fieldnames or columns
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_csv(path, columns):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df:
            df[column] = ""
    # The oldest releases were local-Qwen-only and did not store a model/status field.
    if path.name == "trials.csv":
        df.loc[df["model"].eq(""), "model"] = "Qwen local"
        no_status = df["evaluation_status"].eq("")
        df.loc[no_status & df["prediction"].ne(""), "evaluation_status"] = "ok"
        df.loc[no_status & df["prediction"].eq(""), "evaluation_status"] = "error"
    return df


def load_objects(paths):
    return load_json(paths["objects"], {})


def load_reviews(paths):
    return load_json(paths["reviews"], {})


def migrate_legacy_captures(paths):
    """Recover image manifest from old per-model evaluation rows, never delete old results."""
    evaluations = read_csv(paths["trials"], TRIAL_COLUMNS)
    captures = read_csv(paths["captures"], CAPTURE_COLUMNS)
    if evaluations.empty:
        return
    present = set(zip(captures.trial_id, captures.view_number))
    for _, row in evaluations.iterrows():
        key = (str(row.get("trial_id", "")), str(row.get("view_number", "")))
        if not all(key) or key in present:
            continue
        image = Path(str(row.get("image_path", "")))
        # Keep legacy image path as recorded. It resolves on the user's Windows machine.
        if not str(row.get("image_path", "")):
            continue
        record = {k: row.get(k, "") for k in CAPTURE_COLUMNS}
        record["timestamp"] = row.get("timestamp", utc_now())
        record["phase"] = row.get("phase", "") or "Testing"
        record["sha256"] = ""  # Original path may not exist on a different machine.
        append_csv(paths["captures"], CAPTURE_COLUMNS, record)
        present.add(key)


def image_checksum(image_path):
    digest = hashlib.sha256()
    with Path(image_path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared_jpeg(path):
    """Never changes the captured original; identical JPEG bytes sent to all models."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Can't read captured image: {path}")
    height, width = image.shape[:2]
    ratio = min(1.0, MAX_WIDTH / width, MAX_HEIGHT / height)
    if ratio < 1.0:
        image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        raise RuntimeError(f"Image encoding failed: {path}")
    return buf.tobytes()


def prepared_prefix(paths):
    blobs = [prepared_jpeg(p) for p in paths]
    digest = hashlib.sha256()
    for blob in blobs:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return blobs, digest.hexdigest()


def image_part(blob):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")}}


def parse_recognition(raw):
    """Preserve raw model output. Unstructured responses are NOT silently scored as valid JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
    obj = None
    try:
        obj = json.loads(cleaned)
    except (ValueError, TypeError):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                obj = json.loads(match.group())
            except ValueError:
                pass
    if not isinstance(obj, dict) or not str(obj.get("prediction") or "").strip():
        # Store full output for later human/assistant review; a readable fallback
        # label helps display but remains explicitly 'unstructured'.
        match = re.search(r"(?:object type|object)\s*:\s*(.+)", cleaned, re.I)
        prediction = match.group(1).strip(" *#\r\n")[:250] if match else cleaned.splitlines()[0][:250] if cleaned else ""
        return {"prediction": prediction, "category": "", "brand": "", "product_family": "",
                "product_model": "", "evidence": "", "uncertainties": "",
                "confidence_reported": "", "evaluation_status": "unstructured"}
    value = obj.get("confidence")
    try:
        confidence = float(value) if value is not None and not isinstance(value, bool) else ""
        if confidence != "" and not 0 <= confidence <= 1:
            confidence = ""
    except (TypeError, ValueError):
        confidence = ""
    return {
        "prediction": str(obj.get("prediction") or "").strip(),
        "category": str(obj.get("category") or "").strip(),
        "brand": str(obj.get("brand") or "").strip(),
        "product_family": str(obj.get("product_family") or "").strip(),
        "product_model": str(obj.get("model") or "").strip(),
        "evidence": str(obj.get("evidence") or "").strip(),
        "uncertainties": str(obj.get("uncertainties") or "").strip(),
        "confidence_reported": confidence, "evaluation_status": "ok",
    }


def is_retryable(exc):
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        if int(code) in (408, 409, 429, 500, 502, 503, 504):
            return True
    except (ValueError, TypeError):
        pass
    text = (str(exc) + " " + type(exc).__name__).lower()
    return any(term in text for term in (
        "timeout", "timed out", "connection error", "temporarily unavailable",
        "high demand", "rate limit", "resource exhausted", "429", "503", "502", "504"))


def pace_request(model_name):
    interval = REQUEST_GAP.get(model_name)
    if not interval:
        return
    elapsed = time.monotonic() - LAST_REQUEST_START.get(model_name, -1e9)
    if elapsed < interval:
        time.sleep(interval - elapsed)
    LAST_REQUEST_START[model_name] = time.monotonic()


def call_vision(model_name, blobs, clients, prompt, max_tokens=450):
    """One fresh API request; no history, model answers, object labels or other trials."""
    if model_name == "Gemini cloud":
        from google.genai import types
        parts = [prompt] + [
            types.Part.from_bytes(data=blob, mime_type="image/jpeg") for blob in blobs
        ]
        response = clients[model_name].models.generate_content(
            model=MODEL_IDS[model_name], contents=parts,
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
        raw = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        token_counts = (
            getattr(usage, "prompt_token_count", "") if usage else "",
            getattr(usage, "candidates_token_count", "") if usage else "",
            getattr(usage, "total_token_count", "") if usage else "",
        )
    else:
        message = [{"role": "user", "content": [{"type": "text", "text": prompt}] +
                   [image_part(blob) for blob in blobs]}]
        kwargs = dict(model=MODEL_IDS[model_name], messages=message, max_tokens=max_tokens)
        if model_name == "Qwen local":
            kwargs["temperature"] = 0
        else:
            kwargs.update(temperature=0.6, top_p=0.95,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        response = clients[model_name].chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        token_counts = (
            getattr(usage, "prompt_tokens", "") if usage else "",
            getattr(usage, "completion_tokens", "") if usage else "",
            getattr(usage, "total_tokens", "") if usage else "",
        )
    return raw, token_counts


def call_model(model_name, blobs, clients):
    return call_vision(model_name, blobs, clients, RECOGNITION_PROMPT, 450)


def models_for_method(method):
    """Baseline images are judged by all models; guided images by their own model.

    Old Qwen-guided study_data captures remain evaluable by all, as before.
    """
    if method in ("uniform", "random", "ai_guided"):
        return list(MODEL_ORDER)
    owner = GUIDED_OWNERS.get(method)
    return [owner] if owner else []


def selector_prompt(angles_seen, candidates):
    """Withhold true identity and evaluator output; share *only* this scan's photos."""
    return (
        "You control a fixed camera viewing an object on a rotating platform. "
        + PLATFORM_CONTEXT +
        "Choose the absolute rotation angle of the PLATFORM, not a different camera position. "
        "Your purpose is to select the ONE next platform angle that will expose "
        "NEW visual evidence useful for identifying this same object. "
        "Review only the photographs attached to THIS request. "
        "Already viewed platform angles (degrees): " + json.dumps(angles_seen) + ". "
        "The supplied photos follow this angle list in the SAME order, one photo per angle. "
        "Candidate next angles (degrees): " + json.dumps(candidates) + ". "
        "Choose exactly one angle from the candidate list; favor previously hidden sides, "
        "fine details, labels, unique shapes, and minimize redundant views. "
        "The object must stay on the disc; ignore changes to the disc's edges, "
        "texture and mechanical tilt when deciding what is informative. "
        "You do not know the correct identity and must not make use of prior "
        "recognition answers. Never infer the object name from the prompt. "
        'Return ONLY a JSON object with integer field "angle" and short string field "reason". '
        "The angle must be one of the candidate integers."
    )


def parse_selector(raw, candidates):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
    try:
        obj = json.loads(cleaned)
    except (ValueError, TypeError):
        found = re.search(r"\{[\s\S]*?\}", cleaned)
        if not found:
            raise ValueError("Selector did not return a JSON object")
        obj = json.loads(found.group())
    if not isinstance(obj, dict):
        raise ValueError("Selector response is not a JSON object")
    angle = obj.get("angle")
    if isinstance(angle, str) and angle.strip().isdigit():
        angle = int(angle)
    if type(angle) is not int or angle not in candidates:
        raise ValueError(f"Invalid or repeated selector angle: {angle!r}")
    return angle, str(obj.get("reason") or "")[:400]


def select_angle(paths, metadata, guide_model, seen, images, clients, progress):
    """Independently select angle using model's own scan evidence. Log each attempt.

    A failed selector can use a *clearly flagged* coverage fallback in Testing.
    In Research a failure stops the trial rather than invalidating AI-guided data.
    """
    candidates = [a for a in CANDIDATES if a not in seen]
    blobs, fingerprint = prepared_prefix(images)
    prompt = selector_prompt(seen, candidates)
    last_exception = None
    begin = time.perf_counter()
    for attempt in range(1, 4):
        started = time.perf_counter()
        raw, usage, chosen, reason = "", ("", "", ""), "", ""
        error_type, error_message = "", ""
        try:
            pace_request(guide_model)
            raw, usage = call_vision(guide_model, blobs, clients, prompt, max_tokens=280)
            chosen, reason = parse_selector(raw, candidates)
            status = "ok"
        except Exception as exc:
            last_exception = exc
            status = "error"
            error_type, error_message = type(exc).__name__, safe_error(exc)
        row = dict(
            timestamp=utc_now(), phase=metadata["phase"], batch_id=metadata["batch_id"],
            object_id=metadata["object_id"], trial_id=metadata["trial_id"],
            method=metadata["method"], model=guide_model, model_id=MODEL_IDS[guide_model],
            view_number=len(seen) + 1, observed_angles=json.dumps(seen),
            candidate_count=len(candidates), request_attempt=attempt, status=status,
            suggested_angle=chosen, reason=reason, response_raw=raw,
            error_type=error_type, error_message=error_message,
            duration_seconds=round(time.perf_counter() - started, 2),
            prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2],
            input_sha256=fingerprint, selector_prompt_version=SELECTOR_PROMPT_VERSION,
        )
        append_csv(paths["selectors"], SELECTOR_COLUMNS, row)
        if status == "ok":
            return dict(angle=chosen, reason=reason, status="ok", source="ai",
                        seconds=round(time.perf_counter() - begin, 2), attempts=attempt,
                        raw=raw, usage=usage, fingerprint=fingerprint)
        if attempt == 3 or (last_exception is not None and not is_retryable(last_exception)
                            and not isinstance(last_exception, ValueError)):
            break
        time.sleep(min(2 ** attempt, 5))
    # No silent substitution of a deterministic angle for "AI-guided" Research results.
    if metadata["phase"] == "Research":
        raise RuntimeError(f"{guide_model} viewpoint selection failed; trial stopped. "
                           "Partial captures preserved; rerun this trial after fixing the API. "
                           f"Last error: {type(last_exception).__name__}")
    fallback = select_fallback(candidates, seen)
    progress.warning(f"{guide_model} selector failed; using coverage fallback {fallback}°. "
                     "This Testing trial is not a valid AI-guided scientific comparison.")
    return dict(angle=fallback, reason=f"Selector failed: {type(last_exception).__name__}",
                status="fallback", source="coverage_fallback",
                seconds=round(time.perf_counter() - begin, 2), attempts=attempt,
                raw="", usage=("", "", ""), fingerprint=fingerprint)


def init_clients(selected):
    """Only instantiate selected models; unavailable keys do not interrupt scanning."""
    clients, problems = {}, {}
    for model_name in selected:
        if model_name == "Qwen local":
            clients[model_name] = OpenAI(base_url=LOCAL_ENDPOINT, api_key="local", timeout=120, max_retries=0)
        elif model_name == "NVIDIA Nemotron":
            key = os.getenv("NVIDIA_API_KEY")
            if not key:
                problems[model_name] = "NVIDIA_API_KEY not found. Restart Streamlit from a new terminal."
            else:
                clients[model_name] = OpenAI(base_url=NVIDIA_ENDPOINT, api_key=key, timeout=120, max_retries=0)
        elif model_name == "Gemini cloud":
            if not os.getenv("GEMINI_API_KEY"):
                problems[model_name] = "GEMINI_API_KEY not found. Restart Streamlit from a new terminal."
            else:
                try:
                    from google import genai
                    from google.genai import types
                    clients[model_name] = genai.Client(http_options=types.HttpOptions(timeout=120000))
                except Exception as exc:
                    problems[model_name] = f"Gemini client setup failed: {type(exc).__name__}: {exc}"
    return clients, problems


def select_fallback(available, observed):
    return max(available, key=lambda a: min(min(abs(a - b), 360 - abs(a - b)) for b in observed))


def uniform_angles(budget):
    points = [int(round((i * 360 / budget) / 5)) * 5 % 360 for i in range(budget)]
    if len(set(points)) < budget:
        raise ValueError("Uniform viewpoints are not unique; reduce photograph count")
    return points


def append_batch(paths, record):
    with paths["batches"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def capture_method(paths, phase, object_id, method, budget, camera_index, batch_id, progress):
    """Save every photo promptly. Guided paths use the owner's model independently."""
    from camera import Camera
    from motor_controller import connect, rotate_to, set_status

    guide_model = GUIDED_OWNERS.get(method)
    clients, problems = init_clients([guide_model]) if guide_model else ({}, {})
    if guide_model and problems and phase == "Research":
        raise RuntimeError(f"Cannot run valid {METHOD_NAMES[method]} Research scan: {problems[guide_model]}")
    trial_id = f"{object_id}_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    directory = paths["captures"].parent / "images" / object_id / method
    directory.mkdir(parents=True, exist_ok=True)
    rng_seed = int.from_bytes(os.urandom(8), "big")
    rng = random.Random(rng_seed)
    random_angles = [0] + rng.sample(CANDIDATES[1:], budget - 1)
    uniform = uniform_angles(budget)
    camera = arduino = None
    angle_now = 0
    observed, images = [], []
    succeeded = False
    try:
        arduino = connect()
        set_status(arduino, "CAMERA STARTING")
        camera = Camera(index=camera_index)
        for i in range(budget):
            chosen = 0
            selection_source, reason = "initial", "Same 0-degree starting viewpoint"
            selection_status, selection_attempts, select_sec = "initial", 0, 0.0
            selection_raw, selection_fingerprint = "", ""
            selection_usage = ("", "", "")
            if i:
                if method == "uniform":
                    chosen, selection_source, reason = uniform[i], "uniform", "Equal-angle spacing"
                    selection_status = "fixed"
                elif method == "random":
                    chosen, selection_source, reason = random_angles[i], "random", "Seeded unvisited angle"
                    selection_status = "fixed"
                else:
                    set_status(arduino, "AI SELECTING")
                    if guide_model in problems:
                        if phase == "Research":
                            raise RuntimeError(problems[guide_model])
                        options = [a for a in CANDIDATES if a not in observed]
                        chosen, selection_source = select_fallback(options, observed), "coverage_fallback"
                        selection_status, reason = "fallback", problems[guide_model]
                    else:
                        selection = select_angle(
                            paths, dict(phase=phase, batch_id=batch_id, object_id=object_id,
                                        trial_id=trial_id, method=method),
                            guide_model, observed, images, clients, progress)
                        chosen, selection_source = selection["angle"], selection["source"]
                        reason, selection_status = selection["reason"], selection["status"]
                        selection_attempts, select_sec = selection["attempts"], selection["seconds"]
                        selection_raw, selection_usage = selection["raw"], selection["usage"]
                        selection_fingerprint = selection["fingerprint"]
            if chosen != angle_now:
                set_status(arduino, "ROTATING")
                angle_now = rotate_to(arduino, angle_now, chosen)
            set_status(arduino, "CAPTURING")
            photo = directory / f"{trial_id}_view_{i+1:02d}_{chosen:03d}.jpg"
            started = time.perf_counter()
            camera.capture(str(photo))
            capture_sec = time.perf_counter() - started
            if not photo.exists() or photo.stat().st_size == 0:
                raise IOError(f"Camera did not save the image: {photo}")
            saved_image = cv2.imread(str(photo), cv2.IMREAD_UNCHANGED)
            if saved_image is None:
                raise IOError(f"Saved JPEG cannot be read back: {photo}")
            photo_height, photo_width = saved_image.shape[:2]
            record = dict(
                timestamp=utc_now(), phase=phase, batch_id=batch_id, object_id=object_id,
                trial_id=trial_id, method=method, view_number=i + 1, angle=chosen,
                image_path=str(photo), sha256=image_checksum(photo),
                photo_width=photo_width, photo_height=photo_height,
                selection_source=selection_source, selection_reason=reason,
                selection_seconds=round(select_sec, 3), capture_seconds=round(capture_sec, 3),
                camera_index=camera_index, view_budget=budget, random_seed=rng_seed,
                selection_model=guide_model or "", selection_model_id=MODEL_IDS.get(guide_model, ""),
                selection_status=selection_status, selection_attempts=selection_attempts,
                selection_prompt_tokens=selection_usage[0],
                selection_completion_tokens=selection_usage[1],
                selection_total_tokens=selection_usage[2],
                selection_input_sha256=selection_fingerprint,
                selection_prompt_version=SELECTOR_PROMPT_VERSION if guide_model else "",
                selection_raw=selection_raw,
            )
            append_csv(paths["captures"], CAPTURE_COLUMNS, record)
            images.append(photo)
            observed.append(chosen)
            progress.info(f"Captured {METHOD_NAMES[method]} photo {i+1}/{budget} at {chosen}° — saved")
        succeeded = True
    finally:
        if camera:
            try:
                camera.close()
            except Exception as err:
                progress.warning(f"Camera release warning: {err}")
        if arduino:
            try:
                if angle_now:
                    set_status(arduino, "RETURNING HOME")
                    rotate_to(arduino, angle_now, 0)
                set_status(arduino, "SIGHTLINE READY" if succeeded else "SCAN ERROR")
            except Exception as err:
                progress.warning(f"Return-home failed. Check physical 0° before the next run: {err}")
                if succeeded:
                    succeeded = False
            finally:
                try:
                    arduino.close()
                except Exception:
                    pass
    if not succeeded:
        raise RuntimeError("Scan or return-home failed; verify the platform is at 0°. Saved photos were preserved.")
    return trial_id


def capture_batch(paths, phase, object_id, budget, camera_index, progress):
    batch_id = uuid.uuid4().hex[:12]
    append_batch(paths, dict(timestamp=utc_now(), event="start", phase=phase,
                             batch_id=batch_id, object_id=object_id, budget=budget,
                             camera_index=camera_index))
    completed = []
    method_order = list(METHODS)
    if phase == "Research":
        # Counterbalance order over objects, without changing any scanning strategy.
        order_seed = int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16)
        shift = order_seed % len(METHODS)
        method_order = method_order[shift:] + method_order[:shift]
    append_batch(paths, dict(timestamp=utc_now(), event="method_order", batch_id=batch_id,
                             order=method_order))
    for method in method_order:
        progress.info(f"Collecting {METHOD_NAMES[method]} — hardware starts at 0°")
        try:
            completed.append(capture_method(paths, phase, object_id, method, budget,
                                            camera_index, batch_id, progress))
        except Exception as err:
            append_batch(paths, dict(timestamp=utc_now(), event="interrupted",
                                     batch_id=batch_id, method=method, error=str(err)))
            raise
    append_batch(paths, dict(timestamp=utc_now(), event="complete", batch_id=batch_id,
                             trials=completed))
    return batch_id


def latest_evaluations(evaluations):
    if evaluations.empty:
        return evaluations.copy()
    # All attempts remain stored; newest attempt wins for UI and re-evaluation logic.
    return evaluations.drop_duplicates(subset=["trial_id", "view_number", "model"], keep="last").copy()


def capture_groups(captures):
    """Return ordered, complete observed prefixes only from the SAME trial."""
    if captures.empty:
        return {}
    out = {}
    for trial, df in captures.groupby("trial_id", sort=False):
        df = df.copy()
        df["n"] = pd.to_numeric(df.view_number, errors="coerce")
        df = df.dropna(subset=["n"]).sort_values("n")
        df = df.drop_duplicates(subset=["n"], keep="last")
        out[str(trial)] = df.to_dict(orient="records")
    return out


def eligible_checkpoints(captures, evaluations, selected_models, retry_errors=False, force=False):
    done = latest_evaluations(evaluations)
    existing = {}
    for _, row in done.iterrows():
        existing[(str(row["trial_id"]), str(row["view_number"]), str(row["model"]))] = str(row.get("evaluation_status", "ok") or "ok")
    jobs = []
    for trial, records in capture_groups(captures).items():
        valid = []
        for record in records:
            expected = len(valid) + 1
            if int(float(record["n"])) != expected:
                break  # Never fabricate a missing checkpoint.
            valid.append(record)
            for model in selected_models:
                if model not in models_for_method(record["method"]):
                    continue
                status = existing.get((trial, str(expected), model))
                if force or status is None or (retry_errors and status not in ("ok", "unstructured")):
                    jobs.append((model, list(valid)))
    return jobs


def safe_error(exc):
    # Do not store anything that might include an API bearer token in plaintext.
    message = str(exc)
    for env_name in ("GEMINI_API_KEY", "NVIDIA_API_KEY"):
        secret = os.getenv(env_name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:1000]


def evaluate_saved(paths, phase, selected_models, retry_errors, trial_filter, progress, force=False):
    captures = read_csv(paths["captures"], CAPTURE_COLUMNS)
    if trial_filter:
        captures = captures[captures["trial_id"].isin(trial_filter)]
    trials = read_csv(paths["trials"], TRIAL_COLUMNS)
    jobs = eligible_checkpoints(captures, trials, selected_models, retry_errors, force=force)
    if not jobs:
        return Counter(), 0
    clients, setup_problems = init_clients(selected_models)
    counts = Counter()
    # Same pacing for recognition and viewpoint-selection calls.
    # The quotas are not guarantees of server availability.
    prefix_cache = {}  # image hashes prepared once per checkpoint, reused across models
    for index, (model, prefix) in enumerate(jobs, start=1):
        last = prefix[-1]
        label = f"{model} · {last['method']} · {len(prefix)} photos ({index}/{len(jobs)})"
        progress.info(f"Evaluating {label}")
        images = [r["image_path"] for r in prefix]
        base = dict(
            timestamp=utc_now(), phase=phase, batch_id=last.get("batch_id", ""),
            object_id=last["object_id"], trial_id=last["trial_id"], method=last["method"],
            model=model, model_id=MODEL_IDS[model], view_number=len(prefix), view_count=len(prefix),
            angle=last["angle"], image_path=last["image_path"],
            selection_source=last["selection_source"], selection_reason=last["selection_reason"],
            selection_seconds=last["selection_seconds"], capture_seconds=last["capture_seconds"],
            camera_index=last["camera_index"], view_budget=last["view_budget"],
            recognition_prompt_version=PROMPT_VERSION, input_max_width=MAX_WIDTH,
            input_max_height=MAX_HEIGHT, jpeg_quality=JPEG_QUALITY,
            evaluation_id=uuid.uuid4().hex, notes="Raw prediction retained; scoring requires review",
        )
        if model in setup_problems:
            record = dict(base, evaluation_status="blocked", error_type="MissingSetup",
                          error_message=setup_problems[model], request_attempts=0)
            append_csv(paths["trials"], TRIAL_COLUMNS, record)
            counts["blocked"] += 1
            continue
        started = time.perf_counter()
        attempts = 0
        record = {}
        try:
            key = tuple(images)
            if key not in prefix_cache:
                # Keep memory bounded during 20+ object collection sessions.
                if len(prefix_cache) >= 8:
                    prefix_cache.clear()
                prefix_cache[key] = prepared_prefix(images)
            blobs, fingerprint = prefix_cache[key]
            base["input_sha256"] = fingerprint
            for attempt in range(1, 4):
                attempts = attempt
                attempt_started = time.perf_counter()
                try:
                    pace_request(model)
                    raw, usage = call_model(model, blobs, clients)
                    prediction = parse_recognition(raw)
                    append_csv(paths["evaluation_attempts"], EVALUATION_ATTEMPT_COLUMNS,
                               dict(timestamp=utc_now(), phase=phase, batch_id=last.get("batch_id", ""),
                                    object_id=last["object_id"], trial_id=last["trial_id"],
                                    method=last["method"], model=model, model_id=MODEL_IDS[model],
                                    view_number=len(prefix), attempt=attempt,
                                    status=prediction["evaluation_status"],
                                    duration_seconds=round(time.perf_counter()-attempt_started, 3),
                                    input_sha256=fingerprint, recognition_prompt_version=PROMPT_VERSION,
                                    prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2],
                                    response_raw=raw))
                    record = dict(base, **prediction, response_raw=raw,
                                  prompt_tokens=usage[0], completion_tokens=usage[1],
                                  total_tokens=usage[2], request_attempts=attempts)
                    counts[record["evaluation_status"]] += 1
                    break
                except Exception as exc:
                    append_csv(paths["evaluation_attempts"], EVALUATION_ATTEMPT_COLUMNS,
                               dict(timestamp=utc_now(), phase=phase, batch_id=last.get("batch_id", ""),
                                    object_id=last["object_id"], trial_id=last["trial_id"],
                                    method=last["method"], model=model, model_id=MODEL_IDS[model],
                                    view_number=len(prefix), attempt=attempt, status="error",
                                    duration_seconds=round(time.perf_counter()-attempt_started, 3),
                                    input_sha256=fingerprint, recognition_prompt_version=PROMPT_VERSION,
                                    error_type=type(exc).__name__, error_message=safe_error(exc)))
                    if attempt == 3 or not is_retryable(exc):
                        record = dict(base, evaluation_status="error", error_type=type(exc).__name__,
                                      error_message=safe_error(exc), request_attempts=attempts)
                        counts["error"] += 1
                        break
                    # Limited backoff, no nested SDK retries. No automatic quota grinding.
                    time.sleep(min(2 ** attempt + random.uniform(0, 0.5), 10))
        except Exception as exc:
            record = dict(base, evaluation_status="error", error_type=type(exc).__name__,
                          error_message=safe_error(exc), request_attempts=attempts)
            counts["error"] += 1
        record["evaluation_seconds"] = round(time.perf_counter() - started, 2)
        append_csv(paths["trials"], TRIAL_COLUMNS, record)
        if record.get("evaluation_status") != "ok":
            progress.warning(f"{label}: {record.get('evaluation_status')} — {record.get('error_message') or 'raw response saved for review'}")
    return counts, len(jobs)


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def provisional_score(prediction, brand, model, truth, aliases):
    """Weak text-overlap proxy, NOT a scientific recognition accuracy measurement."""
    guesses = [normalize_label(prediction), normalize_label(brand), normalize_label(model)]
    target = normalize_label(truth)
    accepted = {normalize_label(a) for a in aliases}
    if not target or not guesses[0]:
        return None, "Missing label"
    if target in guesses or any(g in accepted for g in guesses if g):
        return 1.0, "Exact identity or approved synonym (draft)"
    nontrivial = set(target.split()) - {
        "case", "charging", "charger", "toy", "figure", "figurine", "object", "wireless",
        "the", "for", "of", "a", "mini", "small", "device", "plastic", "product", "and",
    }
    intersection = set(" ".join(guesses).split()) & nontrivial
    if intersection:
        return 0.5, "Overlapping words (draft; human review needed)"
    return 0.0, "No match in raw labels (draft; human review needed)"


def review_key(row):
    """Tie human judgment to a specific response, not just its view/model."""
    base = f"{row['trial_id']}::{row['view_number']}::{row['model']}"
    evaluation_id = str(row.get("evaluation_id", "") or "").strip()
    return f"{base}::{evaluation_id}" if evaluation_id else base


def scored_latest(evals, objects, reviews):
    df = latest_evaluations(evals)
    if df.empty:
        return df.assign(provisional_score=pd.Series(dtype=float), reviewed_score=pd.Series(dtype=float))
    drafts, reasons, human = [], [], []
    for _, row in df.iterrows():
        ob = objects.get(str(row.get("object_id", "")), {})
        status = str(row.get("evaluation_status", "ok") or "ok")
        if status not in ("ok", "unstructured"):
            draft, reason = None, "Failed or blocked request"
        else:
            draft, reason = provisional_score(row.get("prediction", ""), row.get("brand", ""),
                                              row.get("product_model", ""), ob.get("ground_truth", ""),
                                              ob.get("aliases", []))
        key = review_key(row)
        answer = reviews.get(key)
        if answer is None and str(row.get("recognition_prompt_version", "")) != PROMPT_VERSION:
            # Prior versions used a view-level key; keep their reviews visible.
            old_key = f"{row['trial_id']}::{row['view_number']}::{row['model']}"
            answer = reviews.get(old_key)
        if answer is None:
            answer = {}
        if isinstance(answer, dict):
            human.append(answer.get("score", None))
        elif answer is not None:
            human.append(answer)  # legacy numeric reviews
        else:
            human.append(None)
        drafts.append(draft)
        reasons.append(reason)
    df["provisional_score"] = pd.to_numeric(pd.Series(drafts, index=df.index), errors="coerce")
    df["reviewed_score"] = pd.to_numeric(pd.Series(human, index=df.index), errors="coerce")
    df["draft_explanation"] = reasons
    df["ground_truth"] = [objects.get(str(oid), {}).get("ground_truth", "") for oid in df.object_id]
    df["review_status"] = df["reviewed_score"].apply(lambda val: "Reviewed" if pd.notna(val) else "Unreviewed")
    return df


def remove_trials(paths, trial_ids):
    """Explicit deletion only. Never wipe study_data when starting final_study_data."""
    trial_ids = set(str(i) for i in trial_ids)
    if not trial_ids:
        return
    captures = read_csv(paths["captures"], CAPTURE_COLUMNS)
    target = captures[captures.trial_id.isin(trial_ids)]
    allowed = (paths["captures"].parent / "images").resolve()
    for photo in target.image_path:
        try:
            path = Path(photo).resolve()
            if path.is_relative_to(allowed) and path.is_file():
                path.unlink()
        except (OSError, ValueError):
            pass
    for name, columns in (("captures", CAPTURE_COLUMNS), ("trials", TRIAL_COLUMNS),
                          ("selectors", SELECTOR_COLUMNS)):
        path = paths[name]
        if not path.exists():
            continue
        df = read_csv(path, columns)
        original_fields = list(df.columns)
        df = df[~df.trial_id.isin(trial_ids)]
        tmp = path.with_name(path.name + ".deleting")
        df.to_csv(tmp, columns=original_fields, index=False)
        os.replace(tmp, path)
    reviews = load_reviews(paths)
    for k in list(reviews):
        if k.split("::")[0] in trial_ids:
            del reviews[k]
    atomic_json(paths["reviews"], reviews)


def export_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def graph_panel(df, value_column, title, model, split_offsets, color_domain=None):
    sub = df[df["model"].eq(model)].copy()
    allowed = ("uniform", "random", GUIDED_METHODS[model], "ai_guided")
    sub = sub[sub.method.isin(allowed)]
    sub[value_column] = pd.to_numeric(sub[value_column], errors="coerce")
    sub = sub.dropna(subset=[value_column, "view_number"])
    if sub.empty:
        return None
    # Include count in tooltips; a mean across one object is exploratory, not evidence.
    sub = sub.groupby(["method", "view_number"], as_index=False).agg(
        **{value_column: (value_column, "mean")}, sample_size=(value_column, "count"))
    shifts = {"uniform": -0.12, "random": 0.0,
              GUIDED_METHODS[model]: 0.12, "ai_guided": 0.24} if split_offsets else {}
    sub["display_x"] = sub.apply(lambda r: float(r["view_number"]) + shifts.get(r["method"], 0.0), axis=1)
    # Shifting only X would still completely hide overlapping horizontal lines.
    # Apply a *tiny visible Y separation ONLY at equal-value checkpoints*, with
    # the TRUE scores retained in value_column and displayed in tooltips/exports.
    sub["display_y"] = sub[value_column].copy()
    if split_offsets:
        for view, rows in sub.groupby("view_number"):
            # Round ONLY to group visually indistinguishable points; actual data stays intact.
            for _, equal_rows in rows.groupby(rows[value_column].round(3)):
                equal_rows = equal_rows.sort_values("method")
                if len(equal_rows) < 2:
                    continue
                base = float(equal_rows[value_column].mean())
                offsets = [(i - (len(equal_rows) - 1) / 2) * 0.025 for i in range(len(equal_rows))]
                # Stay inside the 0-1 chart at perfect 0 and 1.
                if base + min(offsets) < 0:
                    offsets = [o - (base + min(offsets)) for o in offsets]
                elif base + max(offsets) > 1:
                    offsets = [o + (1 - base - max(offsets)) for o in offsets]
                for idx, off in zip(equal_rows.index, offsets):
                    sub.loc[idx, "display_y"] = min(1.0, max(0.0, float(sub.loc[idx, value_column]) + off))
    sub["strategy"] = sub.method.map(METHOD_NAMES)
    strategy_order = [METHOD_NAMES[m] for m in allowed if m in set(sub.method)]
    max_photos = max(5, int(sub.view_number.max()))
    enc = dict(
        x=alt.X("display_x:Q", title="Number of photographs",
                scale=alt.Scale(domain=[0.6, max_photos + 0.5]),
                axis=alt.Axis(values=list(range(1, max_photos + 1)), format="d")),
        y=alt.Y("display_y:Q", title=title, scale=alt.Scale(domain=[0, 1], nice=False)),
        color=alt.Color("strategy:N", title="Scanning strategy", sort=strategy_order),
        strokeDash=alt.StrokeDash("strategy:N", title="Line style", sort=strategy_order),
        tooltip=[alt.Tooltip("strategy:N"),
                 alt.Tooltip("view_number:Q", title="True photo count", format=".0f"),
                 alt.Tooltip(f"{value_column}:Q", title=title, format=".2f"),
                 alt.Tooltip("sample_size:Q", title="Evaluations", format="d")],
    )
    line = alt.Chart(sub).mark_line(strokeWidth=2.7).encode(**enc)
    dots = alt.Chart(sub).mark_point(filled=True, size=120, stroke="white", strokeWidth=1.1).encode(**enc)
    return (line + dots).properties(title=model, height=240)


def render_graph(df, value, y_title, offset=True):
    panels = [p for model in MODEL_ORDER if (p := graph_panel(df, value, y_title, model, offset)) is not None]
    if not panels:
        st.info("No data for this graph yet.")
    else:
        # Each model gets its own panel: fewer hidden overlaps, clearer comparisons.
        st.altair_chart(alt.vconcat(*panels).resolve_scale(color="shared"), use_container_width=True)
        if offset:
            st.caption("Identical scores are separated by a tiny 0.025 vertical spacing, with small "
                       "horizontal offsets at each photo count, **for visibility only**. Hover to see "
                       "the actual unmodified score and photo number; exports use original values.")


def main():
    st.set_page_config(page_title="SIGHTLINE | Research Lab", page_icon="◉", layout="wide")
    st.markdown("""<style>
    .block-container {max-width: 1150px; padding-top: 1.9rem;}
    h1 {letter-spacing:-.035em;} [data-testid="stMetric"] {background:rgba(120,140,165,.08);
    border:1px solid rgba(120,140,165,.16);padding:12px 15px;border-radius:10px;}
    </style>""", unsafe_allow_html=True)
    st.title("◉ SIGHTLINE")
    st.caption("Visual recognition · Viewpoint selection · Reproducible experiments")

    st.sidebar.header("Research lab")
    phase = st.sidebar.radio("Workspace", ["Testing", "Research"],
                             help="Testing preserves existing trials. Research starts in an independent clean folder.")
    data_dir = PHASE_ROOTS[phase]
    paths = data_paths(data_dir)
    migrate_legacy_captures(paths)
    page = st.sidebar.radio("Page", ["Collection", "Evaluate & retry", "Results & graphs",
                                       "Review predictions", "Data & exports", "Methodology"])
    st.sidebar.divider()
    st.sidebar.caption("Evaluators: Qwen local · Gemini 3.5 Flash Lite · NVIDIA Nemotron 3.5 Super VL")
    if phase == "Research":
        st.sidebar.info("Research data is isolated from existing practice scans.")

    objects = load_objects(paths)
    captures = read_csv(paths["captures"], CAPTURE_COLUMNS)
    evaluations = read_csv(paths["trials"], TRIAL_COLUMNS)
    reviews = load_reviews(paths)
    scored = scored_latest(evaluations, objects, reviews)

    if page == "Collection":
        st.subheader("Collect photographs")
        a, b, c, d = st.columns(4)
        a.metric("Objects", len(objects))
        b.metric("Trials", captures.trial_id.nunique() if not captures.empty else 0)
        c.metric("Photographs", len(captures))
        d.metric("Evaluations", len(scored))
        if phase == "Research" and not objects:
            st.info("Clean Research workspace. Your earlier testing photos and results are safe in Testing.")
            test_objects = load_objects(data_paths(PHASE_ROOTS["Testing"]))
            if test_objects and st.button("Copy object definitions from Testing (no photos or results)"):
                atomic_json(paths["objects"], test_objects)
                st.rerun()
        with st.expander("Add an object", expanded=not bool(objects)):
            with st.form("add_object", clear_on_submit=True):
                name = st.text_input("Object name", placeholder="Samsung Galaxy Buds3 Pro")
                truth = st.text_input("Verified correct identity", placeholder="Samsung Galaxy Buds3 Pro charging case")
                aliases = st.text_input("Accepted full-identity synonyms (optional, comma-separated)")
                notes = st.text_input("Notes (optional)")
                submit = st.form_submit_button("Add object", type="primary")
            if submit:
                object_id = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
                if not object_id or not truth.strip():
                    st.error("Enter an object name and verified correct identity.")
                elif object_id in objects:
                    st.error("An object with this name already exists.")
                else:
                    objects[object_id] = dict(ground_truth=truth.strip(),
                                              aliases=[x.strip() for x in aliases.split(",") if x.strip()],
                                              notes=notes, created=utc_now())
                    atomic_json(paths["objects"], objects)
                    st.rerun()
        if objects:
            st.divider()
            object_id = st.selectbox("Object", list(objects),
                                     format_func=lambda oid: objects[oid]["ground_truth"])
            cc, dd = st.columns(2)
            with cc:
                budget = st.slider("Photographs per method", 2, 5, 5,
                                   disabled=(phase == "Research"),
                                   help="Research always uses the preregistered 5-photo budget.")
            with dd:
                camera_idx = st.number_input("Camera index", 0, 10, 3, step=1)
            st.caption("Five scanning strategies (uniform, random, plus one independently guided scan "
                       "per model) · shared 0° start · original photos kept. Recognition runs after capture, "
                       "while angle-selection requests happen during the three guided scans.")
            st.caption("At 5 views: 25 photos and 45 recognition evaluations per object; "
                       "three guided selectors make 12 additional calls total. "
                       "For 20 objects: ~380 calls per model including viewpoint selection, before retries.")
            st.warning("Secure the object, confirm external power, check the camera, and physically "
                       "align the turntable to 0°. Motor angles are commanded, not encoder-measured.")
            acknowledged = st.checkbox("Hardware assembled and safely aligned; object and camera will stay fixed")
            col_a, col_b = st.columns(2)
            run_all = col_a.button("Capture + evaluate", type="primary", disabled=not acknowledged)
            run_capture = col_b.button("Capture only", disabled=not acknowledged)
            if run_all or run_capture:
                info = st.empty()
                before = set(captures.trial_id)
                try:
                    batch_id = capture_batch(paths, phase, object_id, budget, int(camera_idx), info)
                    st.success(f"All five methods captured. Batch {batch_id}; images saved to disk.")
                    if run_all:
                        now = read_csv(paths["captures"], CAPTURE_COLUMNS)
                        fresh_trials = sorted(set(now.trial_id) - before)
                        counts, n = evaluate_saved(paths, phase, MODEL_ORDER, False, fresh_trials, info)
                        st.success(f"Model evaluation finished: {n} checkpoints; "
                                   f"{counts.get('ok',0)} structured, {counts.get('unstructured',0)} "
                                   f"unstructured, {counts.get('error',0)} errors. "
                                   "Failures can be retried without recapturing.")
                except Exception as exc:
                    st.error(f"Operation interrupted: {type(exc).__name__}: {exc}")
                    st.info("Previously saved photographs and model results remain intact. "
                            "Inspect the platform before scanning again. Use Evaluate & retry for saved photos.")
                st.caption("Refresh the page to load the newly saved records.")
        st.caption("In Research mode, only new scans enter the final CSV. Testing data is kept separately.")

    elif page == "Evaluate & retry":
        st.subheader("Evaluate saved photographs")
        st.write("Run recognition independently from hardware. Failures are logged; "
                 "you can retry without retaking any pictures.")
        if captures.empty:
            st.info("No photos captured yet.")
        else:
            selected = st.multiselect("Models", MODEL_ORDER, default=MODEL_ORDER)
            available_trials = list(capture_groups(captures))
            trials_choice = st.multiselect("Trials (leave empty for all)", available_trials)
            retry_errors = st.checkbox("Retry failed/blocked requests", value=True)
            force_repeat = st.checkbox(
                "Re-evaluate successful results with current prompt (extra API calls)",
                value=False, disabled=(phase == "Research"),
                help="Testing only. Does not overwrite earlier responses. Select specific trials to limit usage."
            )
            count = len(eligible_checkpoints(
                captures[captures.trial_id.isin(trials_choice)] if trials_choice else captures,
                evaluations, selected, retry_errors, force_repeat)) if selected else 0
            st.metric("Requests to attempt", count)
            if st.button("Evaluate / resume", type="primary", disabled=not count):
                progress = st.empty()
                counts, total = evaluate_saved(paths, phase, selected, retry_errors,
                                               trials_choice, progress, force=force_repeat)
                st.success(f"Saved {total} evaluations: {dict(counts)}. Refresh for updated status.")
            st.caption("Uniform and random photos are evaluated by all three models. "
                       "Each separately AI-guided scan is evaluated only by its guiding model. "
                       "Each checkpoint is stateless and uses only this trial's photos 1…k. "
                       "Gemini's free account shows 500 requests/day at 15/min; NVIDIA shows up to 40/min. "
                       "Selector calls ALSO count toward these quotas. Previously successful evaluations are not repeated.")

    elif page == "Results & graphs":
        st.subheader("Results")
        if scored.empty:
            st.info("No model results in this workspace. Capture and evaluate an object first.")
        else:
            choice_objects = st.multiselect("Objects", list(objects), default=list(objects),
                                            format_func=lambda oid: objects[oid]["ground_truth"])
            shown = scored[scored.object_id.isin(choice_objects)].copy()
            if shown.empty:
                st.info("Select an object to see its results.")
            else:
                shown["view_number"] = pd.to_numeric(shown.view_number, errors="coerce")
                status_counts = Counter(shown.evaluation_status)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Successful", status_counts.get("ok", 0))
                c2.metric("Unstructured", status_counts.get("unstructured", 0))
                c3.metric("Failed / blocked", status_counts.get("error", 0) + status_counts.get("blocked", 0))
                c4.metric("Human-reviewed", shown.reviewed_score.notna().sum())
                offsets = st.checkbox("Make overlapping graph lines visible", value=True,
                                      help="Tiny visual offsets for equal-value lines only. All true scores remain unchanged.")
                t1, t2, t3 = st.tabs(["Recognition", "Confidence", "Observations"])
                with t1:
                    st.markdown("**Draft name-overlap score (exploratory only)**")
                    render_graph(shown, "provisional_score", "Draft score (0–1)", offsets)
                    st.caption("This is NOT measured accuracy: word overlap can mistake correct categories "
                               "for incorrect names or vice versa. Use the exported raw labels for your "
                               "final human-assisted scoring.")
                    if shown.reviewed_score.notna().any():
                        st.markdown("**Human-reviewed recognition score by model and viewpoint method**")
                        render_graph(shown.dropna(subset=["reviewed_score"]),
                                     "reviewed_score", "Reviewed score (0–1)", offsets)
                    else:
                        st.info("No human-reviewed scores yet. The final study's accuracy graphs "
                                "should use reviewed labels, not these draft scores.")
                with t2:
                    st.markdown("**Self-reported confidence (not accuracy)**")
                    shown["confidence_reported"] = pd.to_numeric(shown.confidence_reported, errors="coerce")
                    render_graph(shown.dropna(subset=["confidence_reported"]),
                                 "confidence_reported", "Confidence (0–1)", offsets)
                    st.caption("Confidence is model-reported and uncalibrated. Do not equate it with accuracy.")
                with t3:
                    cols = ["object_id", "method", "model", "view_number", "angle", "prediction",
                            "category", "brand", "product_family", "product_model", "evaluation_status", "provisional_score",
                            "reviewed_score", "confidence_reported", "evaluation_seconds"]
                    st.dataframe(shown[cols], use_container_width=True, hide_index=True)
                st.download_button("Download current results CSV", export_csv(shown),
                                   "sightline_results.csv", "text/csv", type="primary")

    elif page == "Review predictions":
        st.subheader("Review model predictions")
        if scored.empty:
            st.info("No predictions available yet.")
        else:
            options = scored[scored.evaluation_status.isin(["ok", "unstructured"])].reset_index(drop=True)
            if options.empty:
                st.info("All evaluations failed. Retry them first.")
            else:
                i = st.selectbox("Choose evaluation", list(range(len(options))),
                                 format_func=lambda ix: f"{options.iloc[ix]['model']} · {options.iloc[ix]['object_id']} "
                                 f"· {options.iloc[ix]['method']} · {options.iloc[ix]['view_number']} photos")
                record = options.iloc[i]
                st.write("**Ground truth:**", record.ground_truth)
                st.write("**Model prediction:**", record.prediction)
                st.write("**Category:**", record.category or "Unidentified")
                st.write("**Brand:**", record.brand or "Unidentified")
                st.write("**Product family:**", record.product_family or "Unidentified")
                st.write("**Product/model:**", record.product_model or "Unidentified")
                st.write("**Raw model response:**")
                st.code(record.response_raw, language="json")
                key = review_key(record)
                old_key = f"{record.trial_id}::{record.view_number}::{record.model}"
                previous = reviews.get(key, reviews.get(old_key, {})
                                       if str(record.recognition_prompt_version) != PROMPT_VERSION
                                       else reviews.get(key, {}))
                if isinstance(previous, dict):
                    existing = previous.get("score")
                else:
                    existing = previous
                labels = {
                    "Not reviewed": None,
                    "Incorrect / different object (0)": 0.0,
                    "Correct category only (0.25)": 0.25,
                    "Correct brand or descriptive subtype (0.5)": 0.5,
                    "Correct product family, exact model uncertain (0.75)": 0.75,
                    "Correct exact identity (1)": 1.0,
                }
                initial = next((k for k, v in labels.items() if v == existing), "Not reviewed")
                with st.form("review_one"):
                    choice = st.radio("Human assessment", list(labels),
                                      index=list(labels).index(initial))
                    note = st.text_input("Evidence / caveat (optional)",
                                         value=previous.get("note", "") if isinstance(previous, dict) else "")
                    submit = st.form_submit_button("Save assessment", type="primary")
                if submit:
                    if labels[choice] is None:
                        reviews.pop(key, None)
                    else:
                        reviews[key] = {"score": labels[choice], "note": note,
                                        "reviewed_at": utc_now(), "rubric": "category-brand-family-exact-v2"}
                    atomic_json(paths["reviews"], reviews)
                    st.success("Saved. The results charts will reflect your reviewed score.")
                    st.rerun()
                st.caption("The correct identity is never sent to any recognition model. "
                           "Assessment is a separate post-processing step.")

    elif page == "Data & exports":
        st.subheader("Data, backups and exports")
        st.caption(f"Current workspace: {data_dir} — all image files and evaluation attempts remain on disk.")
        c1, c2 = st.columns(2)
        c1.download_button("Export latest results + ground truth (.csv)", export_csv(scored),
                           f"sightline_{phase.lower()}_results.csv", "text/csv", type="primary")
        c2.download_button("Export complete API attempt history (.csv)", export_csv(evaluations),
                           f"sightline_{phase.lower()}_attempts.csv", "text/csv")
        c3, c4 = st.columns(2)
        c3.download_button("Export capture manifest (.csv)", export_csv(captures),
                           f"sightline_{phase.lower()}_captures.csv", "text/csv")
        c4.download_button("Export object definitions (.json)",
                           json.dumps(objects, indent=2).encode("utf-8"),
                           f"sightline_{phase.lower()}_objects.json", "application/json")
        selector_history = read_csv(paths["selectors"], SELECTOR_COLUMNS)
        st.download_button("Export all viewpoint-selection API attempts (.csv)",
                           export_csv(selector_history),
                           f"sightline_{phase.lower()}_selector_calls.csv", "text/csv")
        attempt_history = read_csv(paths["evaluation_attempts"], EVALUATION_ATTEMPT_COLUMNS)
        st.download_button("Export recognition API attempt log (.csv)",
                           export_csv(attempt_history),
                           f"sightline_{phase.lower()}_evaluation_attempts.csv", "text/csv")
        st.caption("The results CSV includes full raw responses, known identities, image paths, "
                   "model IDs, status, time, token usage and review scores. Image files remain locally "
                   "under the workspace's images/ directory.")
        if not evaluations.empty or not selector_history.empty:
            st.markdown("**Model API usage (evaluation + viewpoint selection)**")
            usage_rows = []
            for model in MODEL_ORDER:
                ev = evaluations[evaluations["model"].eq(model)].copy()
                sel = selector_history[selector_history["model"].eq(model)].copy()
                ev_calls = pd.to_numeric(ev.get("request_attempts", pd.Series(dtype=str)),
                                         errors="coerce").fillna(0).sum()
                sel_calls = len(sel)  # one selector_calls.csv row for each API attempt
                ev_tokens = pd.to_numeric(ev.get("total_tokens", pd.Series(dtype=str)),
                                          errors="coerce").fillna(0).sum()
                sel_tokens = pd.to_numeric(sel.get("total_tokens", pd.Series(dtype=str)),
                                           errors="coerce").fillna(0).sum()
                usage_rows.append(dict(
                    Model=model, Recognition_API_calls=int(ev_calls),
                    Selection_API_calls=int(sel_calls),
                    Total_recorded_calls=int(ev_calls) + int(sel_calls),
                    Total_recorded_tokens=int(ev_tokens + sel_tokens),
                    Recognition_successes=int(ev.evaluation_status.isin(["ok", "unstructured"]).sum())
                ))
            st.dataframe(pd.DataFrame(usage_rows), hide_index=True, use_container_width=True)
            st.caption("Counts use saved request-attempt logs, including retry attempts when recorded. "
                       "They exclude calls made from separate scripts and any undocumented provider-side "
                       "retries. Token reporting may be incomplete for failures or legacy data. "
                       "Consult provider dashboards for the official account quota.")
        st.divider()
        st.markdown("**Manage testing records**")
        if not captures.empty:
            all_trials = sorted(captures.trial_id.unique())
            selected_trial = st.selectbox("Trial", all_trials)
            confirm = st.checkbox("Permanently remove this trial and its images (not reversible)")
            if st.button("Delete selected trial", disabled=not confirm):
                remove_trials(paths, [selected_trial])
                st.rerun()
        if objects:
            with st.expander("Edit an object's verified identity"):
                oid = st.selectbox("Object to edit", list(objects))
                with st.form("edit_truth"):
                    new_truth = st.text_input("Ground truth", value=objects[oid]["ground_truth"])
                    synonyms = st.text_input("Accepted full synonyms", value=", ".join(objects[oid].get("aliases", [])))
                    if st.form_submit_button("Save identity") and new_truth.strip():
                        objects[oid]["ground_truth"] = new_truth.strip()
                        objects[oid]["aliases"] = [x.strip() for x in synonyms.split(",") if x.strip()]
                        atomic_json(paths["objects"], objects)
                        st.rerun()
            with st.expander("Delete an object and all its trials"):
                oid_del = st.selectbox("Object to delete", list(objects), key="delete_object")
                confirmation = st.checkbox("I understand this permanently deletes the object, all trials, and photos")
                if st.button("Delete object and data", disabled=not confirmation):
                    ids = list(captures[captures.object_id.eq(oid_del)].trial_id.unique())
                    remove_trials(paths, ids)
                    objects.pop(oid_del, None)
                    atomic_json(paths["objects"], objects)
                    st.rerun()

    else:
        st.subheader("Methodology & safeguards")
        st.markdown("""
**Fixed recognition models:** Local Qwen3-VL-4B, Google Gemini 3.5 Flash Lite, and NVIDIA Nemotron 3.5 Super VL.

**Platform description:** The shared v5 prompts explicitly describe the large matte-black,
subtly speckled 150 mm 3D-printed disc. Because the disc can tilt slightly due to
mechanical tolerances, its edge and surface are never identifying evidence. This is
an observed hardware limitation, not a measured tilt or proof of constant wobble.
Gemma, Cohere, SenseNova, IteraCompute and Gemini 3.8 Flash are **not** used.

**Five scanning strategies:** uniform, seeded random, Qwen-guided, Gemini-guided and NVIDIA-guided.
All begin at operator-confirmed 0°; the stepper has no absolute encoder. Each guiding model
sees *all its own photos captured so far* and chooses the next angle from the same permitted
5-degree grid, with no correct identity or recognition outputs. Unlike v3, the other two AI
models do not guide another model's scan. This creates a within-model comparison:
uniform versus random versus that model's own guided scan.

**Cross-model transfer limitation:** The present dashboard evaluates each guided scan ONLY
with its guiding model. It supports comparisons across models, but does not yet test whether
Qwen-selected images improve Gemini/NVIDIA recognition, or vice versa. Such transfer
experiments require evaluating other models' guided trials as well, with additional API calls.

**Fair comparison:** Uniform and random scans are evaluated by all three models. Each model
also evaluates its *own* guided scan, making 45 evaluator checkpoints per object at five views.
The three selectors add 12 requests per object; 20 objects require ~380 requests PER model
(including the selector) before retries. This fits Gemini's stated 500/day only if the daily
quota has not already been used; account/provider availability still governs. Each model gets
identical image preprocessing, though different guided methods deliberately yield different
camera views. Selection and evaluation requests are stateless and independent.

Old "ai_guided" trials from earlier dashboard versions remain labeled "Legacy Qwen-guided"
and are *not* treated as Gemini- or NVIDIA-guided scans.

**Isolation:** recognition evaluation happens after all captures; guided viewpoint requests
necessarily happen DURING capture. Each request uses this trial's images 1…k, no ground truth,
previous evaluator answers, other methods, or identity labels. The evaluator's view choices
and explanations are never passed to its independent recognition requests. Original captures are never resized on disk;
only in-memory submission copies are normalized to at most 2560 × 1440, JPEG quality 90.

**Resilience:** every captured photo and every selector/evaluator attempt is retained
(including each individual evaluator retry in evaluation_attempts.csv). Failed
*recognition* calls can be retried from saved photos without repeating hardware. Guided-selection
calls occur during motor control: a failure uses marked fallback ONLY in Testing; Research stops
that trial and preserves partial photos instead of silently pretending coverage is AI guidance. Bounded retry/backoff, explicit API timeout, and NVIDIA pacing are enabled.
Every call's status, duration, token usage if available, errors and raw response are logged.
Requests may still fail during provider outages; no software can guarantee provider uptime.

**Prompt revisions:** In Testing, the optional re-evaluate control can submit existing
saved image prefixes again using the CURRENT prompt. Prior raw responses remain in
trials.csv and each new response gets its own review key. This costs additional API calls.
Research does not permit automatic re-evaluation of successful checkpoints.

**Evaluation:** charts distinguish *provisional lexical matches* from human-reviewed labels.
The draft 0/0.5/1 rubric is a keyword heuristic, not verified accuracy. Human review can assign
0 (incorrect), 0.25 (category), 0.5 (brand/subtype), 0.75 (product family) or 1 (exact identity).
Final scores should be assigned/audited using the exported CSV and a documented rubric before reporting.
Errors are excluded from accuracy denominators, while failure counts are presented separately.

**Study hygiene:** Testing uses `study_data/`. Research uses `final_study_data/` and never
silently includes Testing rows. Photograph budget for Research is fixed to five. Keep lighting,
background, framing and object placement consistent; repeat random trials when resources permit.
Research mode rotates the five-method order across objects to reduce sequence bias, though full
counterbalancing and repeated randomized conditions are still future improvements.
Human scoring and manual motor alignment remain limitations. Verify object ground truth from
packaging or reliable product records, not just AI guesses.

**Cloud privacy:** images are transmitted to Google or NVIDIA when those models are selected.
Do not include sensitive images or secret labels in images. The API keys are read from environment variables.
""")


if __name__ == "__main__":
    main()
