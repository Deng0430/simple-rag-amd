import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge import chunk_section, load_knowledge
from prepare_knowledge import extract_sections
from rag import export_csv, run_query
from retrievers import BM25Retriever, tokenize
from summarize_results import summarize


class RetrievalAndCorpusTests(unittest.TestCase):
    def test_bm25_finds_api_errors_and_rejects_no_keyword_overlap(self):
        passages = [
            dict(id="hip", title="HIP", text="AMD GPU uses torch.cuda with ROCm."),
            dict(id="git", title="Git", text="git pull fetches changes."),
            dict(id="web", title="Web", text="Streamlit listens on port 8501."),
            dict(id="venv", title="Environment", text="Create a Python virtualenv."),
            dict(id="cache", title="Cache", text="Download files with huggingface hub."),
        ]
        retriever = BM25Retriever(passages)
        small = retriever.search("torch.cuda", 1)
        self.assertEqual(small[0]["id"], "hip")
        self.assertEqual(retriever.search("unmentionedword", 3), [])
        self.assertIn("torch.cuda", tokenize("为什么 torch.cuda 可用？"))
        self.assertIn("cuda", tokenize("torch.cuda"))
        with self.assertRaises(ValueError):
            retriever.search(" ", 3)
        with self.assertRaises(ValueError):
            retriever.search("torch.cuda", 0)
        retriever.min_score = small[0]["score"] + 1
        self.assertEqual(retriever.search("torch.cuda", 3), [])

    def test_retrieval_export_identifies_algorithm_and_scores_without_tokens(self):
        passages = [dict(id=str(i), title="Source", text=text) for i, text in enumerate([
            "virtualenv ensurepip", "git status", "GPU ROCm", "Streamlit proxy", "model cache",
        ])]
        result = run_query(BM25Retriever(passages), "ensurepip", 3)
        row = next(csv.DictReader(io.StringIO(export_csv([result]).decode("utf-8-sig"))))
        self.assertEqual(row["retriever"], "bm25")
        self.assertEqual(row["passage_ids"], "0")
        self.assertGreater(float(row["passage_scores"]), 0)
        self.assertEqual(row["input_tokens"], "")
        self.assertEqual(row["answer"], "")

    def test_document_extraction_preserves_highlighted_command_lines(self):
        html = '<nav>Navigation must not be indexed</nav><main><h1 id="setup">Setup</h1>'
        html += '<p>' + "Use this documented setup procedure. " * 6 + '</p>'
        html += '<pre><code><span>python3</span> <span>-m</span> venv .venv\n  echo done\n</code></pre></main>'
        sections = extract_sections(html)
        text = "\n".join(item[2] for item in sections)
        self.assertIn("python3 -m venv .venv\n  echo done", text)
        self.assertNotIn("Navigation must not be indexed", text)
        self.assertEqual(sections[0][1], "setup")

    def test_long_code_chunks_keep_fences_and_content(self):
        code = "\n".join(f"print({n})" for n in range(80))
        chunks = chunk_section("```python\n" + code + "\n```", limit=150)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 150 and chunk.count("```") == 2 for chunk in chunks))
        self.assertTrue(all(f"print({n})" in "\n".join(chunks) for n in range(80)))

    def test_archived_faq_passages_are_excluded_from_active_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder) / "corpus.json"
            corpus.write_text(json.dumps(dict(documents=[], passages=[dict(
                id="doc-test", title="Official original", text="Source text",
            )], complete_aup=False)), encoding="utf-8")
            with patch("knowledge.CORPUS", corpus):
                data = load_knowledge()
        self.assertIn("doc-test", {p["id"] for p in data["passages"]})
        self.assertFalse(any(p["id"].startswith("aup-") for p in data["passages"]))
        self.assertTrue(all(set(q) == {"id", "question"} for q in data["questions"]))

    def test_generation_summary_does_not_mix_retrieval_algorithms(self):
        with self.assertRaisesRegex(ValueError, "Mixed retrievers"):
            summarize([dict(mode="model", retriever=method) for method in ("tfidf", "dense")])


if __name__ == "__main__":
    unittest.main()
