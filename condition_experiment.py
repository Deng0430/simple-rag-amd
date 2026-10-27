"""Prepare, freeze, run and score the small paired condition experiment."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
from statistics import mean, median
import subprocess
import time

from condition_context import load_experiment_corpus, build_contexts, RULE_VERSION
from condition_model import QwenTokenizer, ExperimentModel, make_prompt, sha256_file
from condition_metrics import score_answer, aggregate_review
from retrievers import create_retriever
from rag import _ollama_request, DEFAULT_OLLAMA_URL

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "condition_eval"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def public_question(item):
    return {key: item[key] for key in ("id", "scenario_id", "category", "question", "answer_type")}


def retrieval_query(question):
    # Output-format instructions are not part of the information need.
    return question.split("简短答案", 1)[0].strip()


def validate_gold(questions, gold):
    """Validate scoring annotations without passing them to retrieval or packing."""
    ids = [q["id"] for q in questions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Questions must be nonempty with unique IDs.")
    items = gold.get("items") if isinstance(gold, dict) else None
    if not isinstance(items, list) or any(not isinstance(q, dict) for q in items):
        raise ValueError("Gold must contain an items list.")
    gold_ids = [q.get("id") for q in items]
    if any(not isinstance(qid, str) or not qid for qid in gold_ids) or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Gold IDs must be nonempty and unique.")
    if set(gold_ids) != set(ids):
        raise ValueError("Gold IDs must exactly match the planned question IDs.")
    question_types = {q["id"]: q["answer_type"] for q in questions}
    for item in items:
        qid = item["id"]
        answer_type = item.get("answer_type")
        if answer_type not in ("text", "command", "path") or answer_type != question_types[qid]:
            raise ValueError(f"Gold answer_type mismatch: {qid}.")
        refs = item.get("reference_answers")
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError(f"Gold reference_answers must be nonempty strings: {qid}.")
        slots = item.get("required_slots")
        if not isinstance(slots, list) or not 1 <= len(slots) <= 3:
            raise ValueError(f"Gold requires 1–3 condition slots: {qid}.")
        slot_ids = []
        for slot in slots:
            if not isinstance(slot, dict) or not isinstance(slot.get("id"), str) or not slot["id"].strip():
                raise ValueError(f"Invalid gold slot ID: {qid}.")
            slot_ids.append(slot["id"])
            if not isinstance(slot.get("description"), str) or not slot["description"].strip():
                raise ValueError(f"Missing gold slot description: {qid}.")
            for field in ("acceptable", "evidence_ids"):
                values = slot.get(field)
                if not isinstance(values, list) or not values or any(
                        not isinstance(value, str) or not value.strip() for value in values):
                    raise ValueError(f"Invalid gold slot {field}: {qid}.")
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(f"Duplicate gold slot IDs: {qid}.")
    return {q["id"]: q for q in items}


def unique_records(rows, label, allowed_keys=None):
    """Fail closed instead of silently overwriting repeated resume records."""
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise ValueError(f"Invalid key in {label}.")
        key = row["key"]
        if key in indexed:
            raise ValueError(f"Duplicate key in {label}: {key}.")
        if allowed_keys is not None and key not in allowed_keys:
            raise ValueError(f"Unknown key in {label}: {key}.")
        indexed[key] = row
    return indexed


def validate_schedule(schedule, inputs):
    if (not isinstance(schedule, list) or any(not isinstance(k, str) for k in schedule)
            or len(schedule) != len(set(schedule)) or set(schedule) != set(inputs)):
        raise ValueError("Schedule must contain each input key exactly once.")


def nullable_mean(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def verify_scoring_sources(manifest):
    hashes = {}
    for relative in ("condition_metrics.py", "condition_experiment.py"):
        expected = manifest["source_hashes"].get(relative)
        actual = sha256_file(ROOT / relative)
        if expected is None or actual != expected:
            raise ValueError("Scoring source changed since preparation: " + relative)
        hashes[relative] = actual
    return hashes


def apply_review_overrides(rows, reviews, review_key):
    """Keep automatic scores intact; separately apply documented refusal review."""
    by_key = {review_key[r["review_id"]]: r for r in reviews}
    reviewed = []
    for row in rows:
        review = by_key[row["key"]]
        conflict = review.get("abstain_conflict")
        if conflict is not None and type(conflict) is not bool:
            raise ValueError("Reviewed abstain_conflict must be a boolean or null.")
        reasons = []
        if review.get("refusal") is True:
            reasons.append("reviewed_refusal")
        if conflict is True:
            reasons.append("reviewed_abstain_conflict")
        result = dict(row, automatic_em=row["em"], automatic_token_f1=row["token_f1"],
                      review_type=review["review_type"],
                      review_override_reason=";".join(reasons))
        if reasons:
            result["em"] = result["token_f1"] = 0.0
        reviewed.append(result)
    return reviewed


def capture_host():
    result = {"recorded_at": now(), "python": platform.python_version(), "platform": platform.platform()}
    try:
        result["nvidia_smi_snapshot"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi_error"] = str(exc)
    try:
        result["ollama_ps_snapshot"] = _ollama_request(DEFAULT_OLLAMA_URL, "/api/ps", timeout=5)
    except RuntimeError as exc:
        result["ollama_ps_error"] = str(exc)
    result["memory_note"] = "Point-in-time allocation, not peak memory. Independent Ollama process, not PyTorch."
    return result


def prepare(args):
    folder = Path(args.run_dir)
    if (folder / "manifest.json").exists():
        raise ValueError("Frozen run already exists. Use run/analyze, or choose another run directory.")
    folder.mkdir(parents=True, exist_ok=True)
    source = DATA / ("development.json" if args.split == "development" else "test_questions.json")
    original = read_json(source)
    questions = [public_question(q) for q in original["questions"]]
    gold = ({"items": original["questions"], "metadata": original.get("metadata", {})}
            if args.split == "development" else read_json(DATA / "gold.json"))
    validate_gold(questions, gold)
    corpus = load_experiment_corpus()
    tokenizer = QwenTokenizer()
    retriever = create_retriever(corpus["passages"], "dense", min_score=0.0)
    records = []
    for q in questions:
        start = time.perf_counter()
        query = retrieval_query(q["question"])
        seeds = retriever.search(query, 3)
        retrieval_seconds = time.perf_counter() - start
        contexts = build_contexts(seeds, corpus["passages"], tokenizer.count,
                                  lambda ps: make_prompt(q["question"], ps))
        for group, context in contexts.items():
            records.append(dict(context, question=public_question(q), question_id=q["id"],
                                retrieval_query=query, retrieval_seconds=retrieval_seconds,
                                key=q["id"] + ":" + group))
    rng = random.Random(42)
    ordered = list(questions)
    rng.shuffle(ordered)
    schedule = []
    for q in ordered:
        groups = ["A", "B", "C"]
        rng.shuffle(groups)
        schedule.extend(q["id"] + ":" + group for group in groups)
    write_jsonl(folder / "inputs.jsonl", records)
    write_json(folder / "questions.json", questions)
    write_json(folder / "corpus.json", corpus)
    write_json(folder / "schedule.json", schedule)
    # Gold is frozen for scoring; public_question is the only input used above.
    write_json(folder / "gold.json", gold)
    model = ExperimentModel(tokenizer)
    source_paths = list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("test_condition*.py"))
    source_paths += [ROOT / "docs/project-usage.md", ROOT / "data/source_manifest.json", ROOT / "requirements.txt"]
    source_paths += list(DATA.glob("*.json")) + list(DATA.glob("*.md"))
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT).as_posix()
        destination = folder / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[relative] = sha256_file(path)
    split_code_sections = set(corpus["audit"]["source_code_split_sections"])
    affected = sorted({p["id"] for r in records for p in r["passages"] if p["section_id"] in split_code_sections})
    manifest = {
        "prepared_at": now(), "split": args.split, "n_questions": len(questions),
        "planned_positions": len(schedule), "model": model.metadata,
        "protocol": {"top_k": 3, "extension_budget_tokens": 512, "max_new_tokens": 512,
                     "retrieval_query_rule": "question before the first 简短答案; full original question retained for generation",
                     "rule_version": RULE_VERSION, "run_order_seed": 42,
                     "review_type": "ai_assisted", "human_review_completed": False,
                     "annotation_note": "Source-grounded AI-assisted gold and group-masked AI review; not human labels.",
                     "long_code_note": "Original corpus chunks are preserved, including previously split long source blocks; no new splitting.",
                     "potential_split_code_context_ids": affected},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "source_hashes": source_hashes, "tokenizer": tokenizer.manifest,
        "embedding": {"model": retriever.model_id, "revision": retriever.artifact.get("revision"),
                      "corpus_passages": len(corpus["passages"])},
        "file_hashes": {name: sha256_file(folder / name) for name in
                        ("inputs.jsonl", "questions.json", "corpus.json", "schedule.json", "gold.json")},
    }
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "host_before.json", capture_host())
    different = sum(not next(r for r in records if r["key"] == q["id"] + ":B")["bc_identical"] for q in questions)
    print(f"Prepared {len(questions)} tasks, {len(schedule)} positions; B/C different on {different} tasks.", flush=True)


def verify_frozen(folder):
    manifest = read_json(folder / "manifest.json")
    for name, expected in manifest["file_hashes"].items():
        if sha256_file(folder / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    for relative, expected in manifest["source_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError("Source changed since preparation: " + relative)
    return manifest


def run(args):
    folder = Path(args.run_dir)
    manifest = verify_frozen(folder)
    inputs = unique_records(read_jsonl(folder / "inputs.jsonl"), "inputs")
    schedule = read_json(folder / "schedule.json")
    validate_schedule(schedule, inputs)
    already = unique_records(read_jsonl(folder / "predictions.jsonl"), "predictions", set(inputs))
    model = ExperimentModel(QwenTokenizer())
    if model.metadata != manifest["model"]:
        raise ValueError("Model or runtime configuration changed since preparation.")
    checks = model.validate_tokenizer()
    append_jsonl(folder / "warmups.jsonl", {"at": now(), "checks": checks})
    for position, key in enumerate(schedule, 1):
        if key in already:
            continue
        item = inputs[key]
        result = {"key": key, "question_id": item["question_id"], "group": item["group"],
                  "position": position, "recorded_at": now(), "input_status": item["status"],
                  "attempts": 0}
        if item["status"] != "ready":
            result.update(status=item["status"], raw_response=None, request_seconds=None)
        else:
            for attempt in (1, 2):
                result["attempts"] = attempt
                start = time.perf_counter()
                try:
                    generated = model.generate(item["prompt"], 512)
                    append_jsonl(folder / "attempts.jsonl", {"key": key, "attempt": attempt,
                                                           "recorded_at": now(), **generated})
                    result.update(generated)
                    response = generated["raw_response"]
                    result["status"] = ("tokenizer_mismatch" if not generated["token_count_match"]
                                        else "ok" if response.get("done") is True else "incomplete_response")
                    break
                except RuntimeError as exc:
                    attempt_record = {"key": key, "attempt": attempt, "error": str(exc),
                                      "recorded_at": now(), "request_seconds": time.perf_counter() - start}
                    append_jsonl(folder / "attempts.jsonl", attempt_record)
                    result.update(status="runtime_error", raw_response=None, error=str(exc),
                                  request_seconds=attempt_record["request_seconds"])
                    if attempt == 1:
                        time.sleep(1)
        append_jsonl(folder / "predictions.jsonl", result)
        seconds = result.get("request_seconds")
        print(f"{position}/{len(schedule)} {key}: {result['status']} "
              + (f"{seconds:.3f}s" if seconds is not None else "no generation"), flush=True)
        if result["status"] == "tokenizer_mismatch":
            raise RuntimeError("Token count changed; run paused for diagnosis. Raw response was saved.")
    write_json(folder / "host_after.json", capture_host())
    print(str(folder.resolve()), flush=True)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    folder = Path(args.run_dir)
    manifest = read_json(folder / "manifest.json")
    scoring_hashes = verify_scoring_sources(manifest)
    for name, expected in manifest["file_hashes"].items():
        if sha256_file(folder / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    inputs = unique_records(read_jsonl(folder / "inputs.jsonl"), "inputs")
    validate_schedule(read_json(folder / "schedule.json"), inputs)
    predictions = unique_records(read_jsonl(folder / "predictions.jsonl"), "predictions", set(inputs))
    if set(predictions) != set(inputs):
        raise ValueError("Run is incomplete; do not silently change evaluation denominators.")
    gold = validate_gold(read_json(folder / "questions.json"), read_json(folder / "gold.json"))
    rows, blind = [], []
    for key, item in inputs.items():
        pred, reference = predictions[key], gold[item["question_id"]]
        response = pred.get("raw_response") or {}
        raw = response.get("response", "")
        scored = score_answer(raw, reference["reference_answers"], reference["answer_type"])
        row = {"key": key, "question_id": item["question_id"], "group": item["group"],
               "question": item["question"]["question"], "status": pred["status"],
               "em": scored["em"], "token_f1": scored["token_f1"],
               "parse_ok": scored["parse_ok"], "abstain": scored["abstain"],
               "short_answer": scored["short_answer"], "raw_answer": raw,
               "flags": ";".join(scored["flags"]),
               "done_reason": response.get("done_reason"),
               "input_tokens": response.get("prompt_eval_count"), "output_tokens": response.get("eval_count"),
               "request_seconds": pred.get("request_seconds"),
               "load_seconds": response.get("load_duration", 0) / 1e9 if response else None,
               "extension_tokens": item["extension_tokens"], "extra_ids": ";".join(item["extra_ids"]),
               "context_flags": ";".join(item["flags"])}
        if pred["status"] != "ok":
            row["em"] = row["token_f1"] = 0.0
        rows.append(row)
        blind.append({"key": key, "question_id": item["question_id"], "question": row["question"],
                      "raw_answer": raw, "parsed": scored["parsed"],
                      **{k: reference.get(k, []) for k in ("reference_answers", "required_slots", "allowed_extras",
                                                          "critical_errors", "relevant_evidence")}})
    groups = {}
    for group in "ABC":
        items = [r for r in rows if r["group"] == group]
        times = [r["request_seconds"] for r in items if r["request_seconds"] is not None]
        groups[group] = {
            "n": len(items), "em": mean(r["em"] for r in items),
            "token_f1": mean(r["token_f1"] for r in items),
            "format_failures": sum(not r["parse_ok"] for r in items),
            "self_reported_abstentions": sum(r["abstain"] is True for r in items),
            "truncated_outputs": sum(r["done_reason"] == "length" for r in items),
            "runtime_failures": sum(r["status"] == "runtime_error" for r in items),
            "non_ok_positions": sum(r["status"] != "ok" for r in items),
            "input_budget_exceeded": sum(r["status"] == "input_budget_exceeded" for r in items),
            "no_evidence": sum(r["status"] == "no_evidence" for r in items),
            "status_counts": dict(Counter(r["status"] for r in items)),
            "mean_input_tokens": nullable_mean(r["input_tokens"] for r in items),
            "mean_output_tokens": nullable_mean(r["output_tokens"] for r in items),
            "mean_request_seconds": mean(times) if times else None,
            "median_request_seconds": median(times) if times else None,
        }
    ids = list(gold)
    length_pairs = []
    for qid in ids:
        b, c = inputs[qid + ":B"], inputs[qid + ":C"]
        gap = abs(b["input_tokens"] - c["input_tokens"]) / max(b["input_tokens"], c["input_tokens"])
        length_pairs.append({"question_id": qid, "bc_identical": b["bc_identical"],
                             "relative_input_gap": gap, "near_length_matched": gap <= .05 and not b["bc_identical"],
                             "b_extension_tokens": b["extension_tokens"], "c_extension_tokens": c["extension_tokens"],
                             "no_neighbor": "no_neighbor" in b["flags"]})
    score_by_key = {r["key"]: r for r in rows}
    paired = {}
    for metric in ("em", "token_f1"):
        deltas = [score_by_key[q + ":B"][metric] - score_by_key[q + ":C"][metric] for q in ids]
        paired[metric] = {"mean_delta_B_minus_C": mean(deltas), "improved": sum(d > 1e-12 for d in deltas),
                          "worsened": sum(d < -1e-12 for d in deltas), "tied": sum(abs(d) <= 1e-12 for d in deltas)}
    summary = {"manifest_sha256": sha256_file(folder / "manifest.json"), "groups": groups,
               "scoring_source_hashes": scoring_hashes,
               "paired_B_minus_C": paired, "length_pairs": length_pairs,
               "slot_review_status": "pending_ai_assisted_review", "human_review_completed": False}
    write_csv(folder / "automatic_scores.csv", rows)
    write_json(folder / "automatic_summary.json", summary)
    # Stable masked order, separate key file; reviewers do not need the key file.
    random.Random(20261027).shuffle(blind)
    review_key = {}
    for i, item in enumerate(blind, 1):
        review_id = f"R{i:03}"
        review_key[review_id] = item.pop("key")
        item["review_id"] = review_id
    write_json(folder / "masked_review.json", blind)
    write_json(folder / "review_key.json", review_key)
    review_path = folder / "slot_review.json"
    if review_path.exists():
        reviews = read_json(review_path)
        if {r["review_id"] for r in reviews} != set(review_key) or len(reviews) != len(review_key):
            raise ValueError("Review records do not cover every planned position exactly once.")
        reviewed_groups = {}
        for group in "ABC":
            subset = [r for r in reviews if review_key[r["review_id"]].endswith(":" + group)]
            reviewed_groups[group] = aggregate_review(subset)
        reviewed_answers = apply_review_overrides(rows, reviews, review_key)
        write_csv(folder / "reviewed_answer_scores.csv", reviewed_answers)
        reviewed_answer_groups = {}
        for group in "ABC":
            items = [r for r in reviewed_answers if r["group"] == group]
            reviewed_answer_groups[group] = {
                "n": len(items), "em": mean(r["em"] for r in items),
                "token_f1": mean(r["token_f1"] for r in items),
                "refusal_override_count": sum("reviewed_refusal" in r["review_override_reason"] for r in items),
                "abstain_conflict_override_count": sum("reviewed_abstain_conflict" in r["review_override_reason"]
                                                       for r in items),
            }
        reviewed_by_key = {r["key"]: r for r in reviewed_answers}
        reviewed_paired = {}
        for metric in ("em", "token_f1"):
            deltas = [reviewed_by_key[q + ":B"][metric] - reviewed_by_key[q + ":C"][metric] for q in ids]
            reviewed_paired[metric] = {
                "mean_delta_B_minus_C": mean(deltas),
                "improved": sum(d > 1e-12 for d in deltas),
                "worsened": sum(d < -1e-12 for d in deltas),
                "tied": sum(abs(d) <= 1e-12 for d in deltas),
            }
        summary["slot_review_status"] = "ai_assisted_review_complete"
        summary["slot_reviews"] = reviewed_groups
        summary["reviewed_answer_groups"] = reviewed_answer_groups
        summary["reviewed_answer_paired_B_minus_C"] = reviewed_paired
        summary["answer_review_note"] = (
            "Automatic lexical scores are preserved in automatic_scores.csv; documented semantic refusal "
            "or abstain conflicts set EM/F1 to zero only in reviewed_answer_scores.csv.")
        write_json(folder / "summary.json", summary)
    print(json.dumps({"groups": groups, "paired_B_minus_C": paired,
                      "bc_different": sum(not x["bc_identical"] for x in length_pairs),
                      "near_length_matched": sum(x["near_length_matched"] for x in length_pairs)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "analyze"):
        child = sub.add_parser(command)
        child.add_argument("--run-dir", required=True)
        if command == "prepare":
            child.add_argument("--split", choices=["development", "test"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
