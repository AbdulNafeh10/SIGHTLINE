"""Read-only inventory of local experiment data; never runs inference or hardware."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MODELS = ["Qwen local", "Gemini cloud", "NVIDIA Nemotron"]
OWNERS = dict(qwen_guided=MODELS[0], gemini_guided=MODELS[1], nvidia_guided=MODELS[2])
METHODS = ["uniform", "random", *OWNERS]


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inventory(root):
    output = {"snapshot_date": datetime.now(timezone.utc).date().isoformat(), "workspaces": {}}
    for name in ["study_data", "final_study_data"]:
        base = root / name
        objects = json.loads((base / "objects.json").read_text(encoding="utf-8-sig"))
        captures = read_rows(base / "captures.csv")
        evaluations = read_rows(base / "trials.csv")
        selectors = read_rows(base / "selector_calls.csv")
        attempts = read_rows(base / "evaluation_attempts.csv")
        groups = defaultdict(list)
        for row in captures:
            groups[row["trial_id"]].append(row)
        batch_methods = defaultdict(set)
        complete = 0
        expected = set()
        for trial, rows in groups.items():
            views = {int(r["view_number"]) for r in rows}
            if views == set(range(1, 6)):
                complete += 1
                row = rows[0]
                batch_methods[(row["object_id"], row.get("batch_id", ""))].add(row["method"])
            for n in range(1, max(views) + 1):
                if n not in views:
                    break
                method = rows[0]["method"]
                models = [OWNERS[method]] if method in OWNERS else MODELS
                expected.update((trial, str(n), m) for m in models)
        latest = {(r["trial_id"], r["view_number"], r.get("model") or MODELS[0]): r for r in evaluations}
        statuses = Counter((r.get("evaluation_status") or ("ok" if r.get("prediction") else "error")) for r in latest.values())
        missing = bad_hash = 0
        for row in captures:
            # Read relocated copies without rewriting any manifest.
            recorded = row["image_path"].replace("\\", "/")
            suffix = recorded.split("images/", 1)[-1]
            path = base / "images" / suffix
            if not path.is_file():
                missing += 1
            elif row.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                bad_hash += 1
        completed_batches = sum(set(METHODS) <= methods for methods in batch_methods.values())
        output["workspaces"][name] = dict(
            registered_objects=len(objects), registered_ids_with_captures=len({r["object_id"] for r in captures}),
            capture_rows=len(captures), retained_images=len(list((base / "images").rglob("*.jpg"))),
            captured_trials=len(groups), five_view_trials=complete, five_strategy_capture_batches=completed_batches,
            evaluation_rows=len(evaluations), unique_evaluation_keys=len(latest), latest_statuses=dict(statuses),
            never_attempted_checkpoints=len(expected - set(latest)), expected_checkpoints=len(expected),
            selector_attempts=len(selectors), selector_statuses=dict(Counter(r["status"] for r in selectors)),
            recognition_attempt_rows=len(attempts), manual_review_file_present=(base / "reviews.json").exists(),
            missing_images=missing, checksum_mismatches=bad_hash,
            duplicate_capture_keys=len(captures)-len({(r["trial_id"],r["view_number"]) for r in captures}),
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(base.glob("*")) if p.suffix in (".csv", ".json", ".jsonl")},
        )
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("root", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(inventory(args.root), indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)
