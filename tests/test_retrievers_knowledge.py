import csv
import copy
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



class CsvExportTests(unittest.TestCase):
    def test_evidence_round_trip_preserves_citation_order_text_and_all_sources(self):
        passages = [
            dict(id="first", title='指南：Python, "环境"', score=0.123456789,
                 scope="当前项目目录", text='第一行, 含逗号和引号 "原文"\r\n```python\nprint("你好")\n```',
                 sources=[
                     dict(title="官方说明", url="https://example.org/guide?a=1&b=2#venv",
                          kind="官方文档", checked_on="2026-11-17"),
                     dict(title="项目说明", url="https://example.org/project", kind="项目说明"),
                 ], document_id="guide", rrf_contributions={"dense": 0.01, "bm25": 0.02}),
            dict(id="second", title="补充说明", score=0.02, text="证据原文\n" * 100,
                 sources=[dict(title="另一来源", url="https://example.org/second")]),
        ]
        result = dict(question='应该用哪个命令, 为什么？', answer='先检查版本。\n\n[P1]',
                      raw_answer='{"short_answer":"先检查版本"}', strategy="X", model="test-model",
                      input_tokens=100, output_tokens=20, total_seconds=1.23,
                      recorded_at="2026-11-17T08:00:00+00:00", passages=passages)
        original = copy.deepcopy(result)
        exported = export_csv([result])
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(exported.decode("utf-8-sig"), newline="")))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key, value in result.items():
            if key != "passages":
                self.assertEqual(row[key], str(value))
        self.assertEqual(row["passage_ids"], "first;second")
        self.assertEqual(row["passage_scores"], "0.123457;0.020000")
        snapshots = json.loads(row["passages_json"])
        self.assertEqual([p["citation_id"] for p in snapshots], ["P1", "P2"])
        for source, snapshot in zip(passages, snapshots):
            self.assertEqual({k: v for k, v in snapshot.items() if k != "citation_id"}, source)
            self.assertIn(source["text"], row["evidence_text"])
            self.assertIn(source["title"], row["evidence_text"])
            for reference in source["sources"]:
                self.assertIn(reference["url"], row["evidence_text"])
        self.assertLess(row["evidence_text"].index("[P1]"), row["evidence_text"].index("[P2]"))
        self.assertEqual(result, original)

    def test_preview_and_no_evidence_keep_blank_answers_and_tokens(self):
        preview = dict(question="检查环境", mode="retrieval_preview", answer=None,
                       input_tokens=None, output_tokens=None, passages=[
                           dict(id="minimal", title="Guide", text="Use python3", score=1.0)])
        empty = dict(preview, question="没有相关资料", passages=[], status="no_evidence")
        rows = list(csv.DictReader(io.StringIO(export_csv([preview, empty]).decode("utf-8-sig"))))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["input_tokens"], "")
            self.assertEqual(row["output_tokens"], "")
        self.assertEqual(json.loads(rows[0]["passages_json"])[0]["text"], "Use python3")
        self.assertEqual(rows[1]["passages_json"], "[]")
        self.assertEqual(rows[1]["evidence_text"], "")
        self.assertEqual(rows[1]["passage_ids"], "")

    def test_failed_and_refused_answers_keep_raw_output_and_evidence(self):
        passage = dict(id="source", title="Guide", text="Documented content.", score=0.5)
        failed = dict(question="test", answer=None, status="format_error", raw_answer='{"short_answer":"截断',
                      output_truncated=True, passages=[passage])
        refused = dict(question="test", answer="资料不足", status="abstained",
                       raw_answer='{"abstain":true}', passages=[passage])
        rows = list(csv.DictReader(io.StringIO(export_csv([failed, refused]).decode("utf-8-sig"))))
        for original, row in zip((failed, refused), rows):
            self.assertEqual(row["raw_answer"], original["raw_answer"])
            self.assertEqual(row["status"], original["status"])
            self.assertEqual(json.loads(row["passages_json"])[0]["text"], passage["text"])
        self.assertEqual(rows[0]["answer"], "")
        self.assertEqual(rows[1]["answer"], "资料不足")


if __name__ == "__main__":
    unittest.main()
