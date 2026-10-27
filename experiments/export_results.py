"""Export a small public audit bundle from the retained local experiment runs.

This is an archival operation, not an experiment or model call. Full documents,
prompts, host snapshots, model files and runtime logs are deliberately omitted.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
BATCHES = {
    "condition_v1": "runs/condition_eval/2026-10-27_test_v1",
    "retrieval_v2_new12": "runs/retrieval_eval_v2/2026-10-27_holdout_v2_reset",
    "retrieval_v2_regression20": "runs/retrieval_eval_v2/2026-10-27_regression_v2_reset",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export(name, relative_source):
    source, target = ROOT / relative_source, DEST / name
    target.mkdir(parents=True, exist_ok=True)
    manifest = read_json(source / "manifest.json")
    gold = {x["id"]: x for x in read_json(source / "gold.json")["items"]}
    questions = read_json(source / "questions.json")
    review_key = read_json(source / "review_key.json")
    reviews = {review_key[x["review_id"]]: x for x in read_json(source / "slot_review.json")}
    predictions = {x["key"]: x for x in map(json.loads, (source / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    with (source / "reviewed_answer_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
        scores = list(csv.DictReader(handle))
    public_questions = []
    for question in questions:
        reference = gold[question["id"]]
        public_questions.append({
            **question,
            "reference_answers": reference["reference_answers"],
            "required_slot_ids": [x["id"] for x in reference["required_slots"]],
        })
    write_json(target / "questions.json", public_questions)
    records = []
    for score in scores:
        prediction, review = predictions[score["key"]], reviews[score["key"]]
        records.append({
            "key": score["key"], "question_id": score["question_id"], "group": score["group"],
            "status": score["status"], "attempts": prediction["attempts"],
            "reused_from": prediction.get("reused_from"),
            "raw_answer": prediction["raw_response"].get("response", ""),
            "done_reason": prediction["raw_response"].get("done_reason"),
            "expected_input_tokens": prediction.get("expected_input_tokens"),
            "measured_input_tokens": prediction["raw_response"].get("prompt_eval_count"),
            "output_tokens": prediction["raw_response"].get("eval_count"),
            "automatic_em": float(score["automatic_em"]),
            "automatic_token_f1": float(score["automatic_token_f1"]),
            "reviewed_em": float(score["em"]), "reviewed_token_f1": float(score["token_f1"]),
            "review_override_reason": score["review_override_reason"],
            "review": {k: review[k] for k in (
                "review_id", "review_type", "tp", "fp", "fn", "critical_error", "acceptable",
                "refusal", "abstain_conflict", "fp_reasons", "notes") if k in review},
            "slot_assessments": [
                {k: slot[k] for k in ("slot_id", "satisfied")}
                for slot in review["slot_assessments"]
            ],
        })
    (target / "records.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    for filename in ("summary.json", "paired_differences.csv", "results_table.csv"):
        shutil.copyfile(source / filename, target / filename)
    source_files = ["manifest.json", "summary.json", "predictions.jsonl", "reviewed_answer_scores.csv",
                    "automatic_scores.csv", "slot_review.json", "review_key.json", "questions.json",
                    "gold.json", "corpus.json", "inputs.jsonl", "attempts.jsonl",
                    "paired_differences.csv", "results_table.csv"]
    attempts = list(map(json.loads, (source / "attempts.jsonl").read_text(encoding="utf-8").splitlines()))
    provenance = {
        "export_version": 1, "source_run": relative_source, "split": manifest["split"],
        "prepared_at": manifest["prepared_at"], "question_count": manifest["n_questions"],
        "logical_positions": manifest["planned_positions"],
        "actual_attempts": len(attempts),
        "reused_positions": sum(bool(x["reused_from"]) for x in records),
        "model": manifest["model"], "protocol": manifest["protocol"],
        "retrieval": manifest.get("retrieval_config", manifest.get("embedding")),
        "corpus_sha256": sha(source / "corpus.json"),
        "frozen_scoring_source_sha256": manifest["source_hashes"]["condition_metrics.py"],
        "frozen_scoring_source_lf_sha256": hashlib.sha256(
            (source / "source_snapshot/condition_metrics.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "source_file_sha256": {f: sha(source / f) for f in source_files},
        "human_review_completed": False,
        "review_note": "Source-grounded AI-assisted gold and method-masked AI initial review with visible-group coordination; not independent human validation.",
        "omitted": ["full third-party documents", "full prompts", "source snapshots", "host snapshots",
                    "service logs", "model weights", "unselected initial failed batches"],
        "scope_note": "This bundle reproduces reported metric calculations, not exact historical retrieval or generation. Full local runs remain archived.",
    }
    write_json(target / "provenance.json", provenance)
    write_json(target / "checksums.json", {
        p.name: sha(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != "checksums.json"
    })
    print(f"{name}: {len(records)} positions exported")


if __name__ == "__main__":
    for batch, source in BATCHES.items():
        export(batch, source)
