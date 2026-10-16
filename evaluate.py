"""Run the fixed small comparison and save real measurements incrementally."""

import argparse
import json
from pathlib import Path

from rag import DEFAULT_DATA, DEFAULT_MODEL, Generator, Retriever, environment, export_csv, load_data, run_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--top-k", type=int, nargs="+", default=[3],
                        help="Pass 3 6 for an optional paired comparison")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--preview", action="store_true", help="Retrieval only; does not generate answers")
    parser.add_argument("--output", default="runs/aup_demo.csv")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if any(k < 1 for k in args.top_k):
        parser.error("--top-k must be positive")
    data = load_data(args.data)
    retriever = Retriever(data["passages"], **data.get("retriever", {}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = None if args.preview else Generator(args.model)
    output.with_suffix(".environment.json").write_text(
        json.dumps(dict(environment(), knowledge_base=data.get("id", Path(args.data).stem),
                        data_file=Path(args.data).name), ensure_ascii=False, indent=2),
        encoding="utf-8", newline="\n",
    )
    results = []
    for question in data["questions"][:args.limit]:
        for k in args.top_k:
            result = run_query(retriever, question["question"], k, generator,
                               profile=data.get("prompt_profile", "general"),
                               max_new_tokens=data.get("max_new_tokens", 128),
                               knowledge_base=data.get("id", Path(args.data).stem))
            result.update(
                question_id=question["id"], reference_answers=" | ".join(question["answers"]),
                reference_passage_retrieved=any(
                    p["id"] == question["passage_id"] for p in result["passages"]
                ),
                manual_quality="",
            )
            results.append(result)
            output.write_bytes(export_csv(results))
            print(f"{len(results)} runs saved: k={k}, {result['total_seconds']:.3f}s", flush=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
