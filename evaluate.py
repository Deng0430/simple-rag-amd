"""Run the fixed small comparison and save real measurements incrementally."""

import argparse
import json
from pathlib import Path

from rag import DEFAULT_MODEL, Generator, environment, export_csv, load_data, run_query
from knowledge import load_knowledge
from retrievers import create_retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", help="Optional legacy JSON dataset; otherwise uses official documents")
    parser.add_argument("--retriever", choices=["tfidf", "bm25", "dense", "all"], default="tfidf")
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
    if args.retriever == "all" and not args.preview:
        parser.error("--retriever all is a retrieval-only comparison; add --preview")
    data = load_data(args.data) if args.data else load_knowledge()
    methods = ["tfidf", "bm25", "dense"] if args.retriever == "all" else [args.retriever]
    retrievers = {name: create_retriever(data["passages"], name,
                  analyzer=data.get("retriever", {}).get("analyzer", "word" if args.data else "char")) for name in methods}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = None if args.preview else Generator(args.model)
    output.with_suffix(".environment.json").write_text(
        json.dumps(dict(environment(), knowledge_base=data.get("id", "custom"),
                        data_file=Path(args.data).name if args.data else "official documents + project usage",
                        retrievers=methods), ensure_ascii=False, indent=2),
        encoding="utf-8", newline="\n",
    )
    results = []
    for question in data["questions"][:args.limit]:
        for name, k in [(name, k) for name in methods for k in args.top_k]:
            retriever = retrievers[name]
            result = run_query(retriever, question["question"], k, generator,
                               profile=data.get("prompt_profile", "general"),
                               max_new_tokens=data.get("max_new_tokens", 128),
                               knowledge_base=data.get("id", "custom"))
            result.update(
                question_id=question["id"], reference_answers=" | ".join(question.get("answers", [])),
                reference_passage_retrieved=any(
                    p["id"] == question.get("passage_id") for p in result["passages"]
                ) if "passage_id" in question else None,
                manual_quality="",
            )
            results.append(result)
            output.write_bytes(export_csv(results))
            print(f"{len(results)} runs saved: {name}, k={k}, {result['total_seconds']:.3f}s", flush=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
