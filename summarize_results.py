"""Summarize exported model runs; answer grades come from a separate human review."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def summarize(rows):
    if not rows or any(row["mode"] != "model" for row in rows):
        raise ValueError("Expected nonempty model results, not retrieval previews.")
    # Historical CSV files predate the retriever field and used TF-IDF only.
    methods = {row.get("retriever") or "tfidf" for row in rows}
    if len(methods) != 1:
        raise ValueError("Mixed retrievers must be summarized separately; filter the CSV by retriever first.")
    groups = {}
    for k in sorted({int(row["top_k"]) for row in rows}):
        group = [row for row in rows if int(row["top_k"]) == k]
        graded = [row for row in group if row["manual_quality"] in ("correct", "partial", "incorrect")]
        correct = sum(row["manual_quality"] == "correct" for row in graded)
        groups[str(k)] = {
            "count": len(group),
            "mean_total_seconds": mean(float(row["total_seconds"]) for row in group),
            "median_total_seconds": median(float(row["total_seconds"]) for row in group),
            "mean_generation_seconds": mean(float(row["generation_seconds"]) for row in group),
            "mean_retrieval_seconds": mean(float(row["retrieval_seconds"]) for row in group),
            "sum_total_seconds": sum(float(row["total_seconds"]) for row in group),
            "mean_input_tokens": mean(int(row["input_tokens"]) for row in group),
            "mean_output_tokens": mean(int(row["output_tokens"]) for row in group),
            "max_torch_peak_allocated_gib": max(float(row["torch_peak_allocated_gib"]) for row in group),
            "reference_passage_hit_count": sum(row["reference_passage_retrieved"] == "True" for row in group),
            "graded_count": len(graded),
            "correct_count": correct,
            "reviewed_correctness": correct / len(graded) if graded else None,
        }
    result = {"retriever": next(iter(methods)), "groups": groups}
    if "3" in groups and "6" in groups:
        small, large = groups["3"], groups["6"]
        result["reductions_relative_to_k6_percent"] = {
            "mean_input_tokens": 100 * (1 - small["mean_input_tokens"] / large["mean_input_tokens"]),
            "mean_total_seconds": 100 * (1 - small["mean_total_seconds"] / large["mean_total_seconds"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to an evaluation CSV.")
    parser.add_argument("--output")
    args = parser.parse_args()
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as source:
        result = summarize(list(csv.DictReader(source)))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")


if __name__ == "__main__":
    main()
