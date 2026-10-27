"""Recompute published EM, Token-F1 and AI-label Slot-F1 using only stdlib.

Run: python experiments/verify_results.py
No model, network, corpus cache, third-party packages or private runs required.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from condition_metrics import score_answer


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def verify(batch):
    provenance = read_json(batch / "provenance.json")
    for name, expected in read_json(batch / "checksums.json").items():
        actual = hashlib.sha256((batch / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Changed artifact: {batch.name}/{name}")
    scorer_hash = hashlib.sha256(
        (HERE.parent / "condition_metrics.py").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    if scorer_hash != provenance["frozen_scoring_source_lf_sha256"]:
        raise ValueError("condition_metrics.py differs from the frozen scoring implementation")
    questions = {x["id"]: x for x in read_json(batch / "questions.json")}
    records = list(map(json.loads, (batch / "records.jsonl").read_text(encoding="utf-8").splitlines()))
    if len(records) != provenance["logical_positions"]:
        raise ValueError("Missing logical positions")
    keys = {r["key"] for r in records}
    if len(keys) != len(records):
        raise ValueError("Duplicate logical positions")
    by_key = {r["key"]: r for r in records}
    for record in records:
        if record["reused_from"]:
            original = by_key[record["reused_from"]]
            if (original["raw_answer"] != record["raw_answer"]
                    or original["question_id"] != record["question_id"]):
                raise ValueError("Invalid reused answer link")
    groups = defaultdict(list)
    for record in records:
        q = questions[record["question_id"]]
        scores = score_answer(record["raw_answer"], q["reference_answers"], q["answer_type"])
        close(scores["em"], record["automatic_em"], record["key"] + " automatic EM")
        close(scores["token_f1"], record["automatic_token_f1"], record["key"] + " automatic F1")
        review = record["review"]
        if review["review_type"] != "ai_assisted":
            raise ValueError("Review provenance changed")
        for key in ("tp", "fp", "fn"):
            if type(review[key]) is not int or review[key] < 0:
                raise ValueError("Invalid slot counts")
        if review["tp"] + review["fn"] != len(q["required_slot_ids"]):
            raise ValueError("Slot denominator mismatch")
        if {s["slot_id"] for s in record["slot_assessments"]} != set(q["required_slot_ids"]):
            raise ValueError("Slot IDs mismatch")
        if sum(s["satisfied"] for s in record["slot_assessments"]) != review["tp"]:
            raise ValueError("Slot assessments disagree with counts")
        reviewed_em, reviewed_f1 = scores["em"], scores["token_f1"]
        if review.get("refusal") or review.get("abstain_conflict"):
            reviewed_em = reviewed_f1 = 0
        close(reviewed_em, record["reviewed_em"], record["key"] + " reviewed EM")
        close(reviewed_f1, record["reviewed_token_f1"], record["key"] + " reviewed F1")
        groups[record["group"]].append(record)
    summary = read_json(batch / "summary.json")
    for group in sorted(groups):
        rows = groups[group]
        if len(rows) != len(questions) or {r["question_id"] for r in rows} != set(questions):
            raise ValueError("Incomplete group denominator")
        n = len(rows)
        em = sum(r["reviewed_em"] for r in rows) / n
        f1 = sum(r["reviewed_token_f1"] for r in rows) / n
        tp, fp, fn = (sum(r["review"][k] for r in rows) for k in ("tp", "fp", "fn"))
        slot_f1 = 2 * tp / (2 * tp + fp + fn)
        expected_answer = summary["reviewed_answer_groups"][group]
        expected_slots = summary["slot_reviews"][group]
        close(em, expected_answer["em"], group + " EM")
        close(f1, expected_answer["token_f1"], group + " F1")
        close(slot_f1, expected_slots["slot_f1"], group + " Slot-F1")
        for value, key in zip((tp, fp, fn), ("tp", "fp", "fn")):
            close(value, expected_slots[key], group + " " + key)
        print(f"{batch.name:26} {group} N={n:2} EM={em:7.2%} F1={f1:7.2%} Slot-F1={slot_f1:7.2%} TP/FP/FN={tp}/{fp}/{fn}")
    return len(records)


if __name__ == "__main__":
    total = sum(verify(HERE / name) for name in (
        "condition_v1", "retrieval_v2_new12", "retrieval_v2_regression20"))
    print(f"Verified {total} logical positions. Slot labels are AI-assisted; human review remains incomplete.")
