"""Create a deterministic, attributed SQuAD subset for a small demo."""

import hashlib
import json
import random
import urllib.request
from pathlib import Path

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
TARGET = Path(__file__).parent / "data" / "squad_demo.json"


def prepare():
    with urllib.request.urlopen(URL, timeout=60) as response:
        articles = json.load(response)["data"]
    rng = random.Random(42)
    passages, questions, seen = [], [], set()

    def add(article, paragraph):
        text = paragraph["context"]
        pid = hashlib.sha256(text.encode()).hexdigest()[:16]
        if pid not in seen:
            passages.append({"id": pid, "title": article["title"], "text": text})
            seen.add(pid)
        return pid

    for article in rng.sample(articles, 10):
        paragraph = rng.choice([p for p in article["paragraphs"] if p["qas"]])
        question = rng.choice([q for q in paragraph["qas"] if q.get("answers")])
        pid = add(article, paragraph)
        questions.append({
            "id": question["id"], "question": question["question"],
            "answers": list(dict.fromkeys(a["text"] for a in question["answers"])),
            "passage_id": pid,
        })
    candidates = [(a, p) for a in articles for p in a["paragraphs"]]
    rng.shuffle(candidates)
    for article, paragraph in candidates:
        if len(passages) >= 30:
            break
        add(article, paragraph)
    rng.shuffle(passages)
    data = {
        "source": URL, "license": "CC-BY-SA-4.0", "seed": 42,
        "description": "10 questions and 30 original paragraphs, sampled from SQuAD 1.1 dev.",
        "passages": passages, "questions": questions,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(questions)} questions / {len(passages)} passages: {TARGET}")


if __name__ == "__main__":
    prepare()
