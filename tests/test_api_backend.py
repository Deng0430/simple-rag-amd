import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from api_backend import ApiError, api_endpoint
from hybrid_retrieval import reciprocal_rank_fusion
from hybrid_runtime import PlanRuntime, run_plan_query
from rag import create_generator, export_csv, run_query


API_KEY = "test-only-credential"
PASSAGE = dict(id="guide", title="Python guide", text="Run python3 --version.",
               scope="AUP Code Server", score=1.0, sources=[])


def reply(content, *, usage=None, finish="stop"):
    data = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = usage
    return io.BytesIO(json.dumps(data).encode("utf-8"))


def valid_answer(citations=("P1",)):
    return json.dumps(dict(short_answer="python3 --version", explanation="Check Python first.",
                           citations=list(citations), abstain=False))


class ApiBackendTests(unittest.TestCase):
    def test_endpoint_allows_known_hosts_and_normalizes_the_path(self):
        self.assertEqual(api_endpoint("https://openrouter.ai/api/v1/"),
                         "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(api_endpoint("https://api.openai.com/v1/chat/completions"),
                         "https://api.openai.com/v1/chat/completions")
        with patch.dict(os.environ, {"RAG_API_ALLOWED_HOSTS": "api.example.test"}):
            self.assertEqual(api_endpoint("https://api.example.test/custom"),
                             "https://api.example.test/custom/chat/completions")
            for url in ("http://api.openai.com/v1", "https://127.0.0.1/v1",
                        "https://user:pass@api.openai.com/v1", "https://api.openai.com/v1?key=secret",
                        "https://api.openai.com.evil.example/v1", "https://api.openai.com:8443/v1",
                        "https://api.openai.com/v1#x", "https://api.example.test/\r\nx"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    api_endpoint(url)

    def test_bm25_path_sends_grounded_prompt_and_exports_usage_without_key(self):
        generator = create_generator("api", "demo-model", "https://api.openai.com/v1", api_key=API_KEY)
        retriever = SimpleNamespace(
            name="bm25", score_kind="BM25", min_score=0, passages=[PASSAGE],
            search=lambda question, k: [PASSAGE],
        )
        with patch("api_backend._opener.open", return_value=reply(
            valid_answer(), usage={"prompt_tokens": 20, "completion_tokens": 8}
        )) as call:
            result = run_query(retriever, "How do I check Python?", 3, generator)
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer " + API_KEY)
        self.assertEqual(payload["max_completion_tokens"], 128)
        self.assertNotIn(API_KEY.encode(), request.data)
        self.assertIn("[P1] Python guide", payload["messages"][1]["content"])
        self.assertEqual(result["status"], "matched")
        self.assertTrue(result["answer"].endswith("[P1]"))
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (20, 8))
        self.assertEqual(result["api_host"], "api.openai.com")
        csv = export_csv([result]).decode("utf-8-sig")
        self.assertIn("Run python3 --version.", csv)
        self.assertNotIn(API_KEY, csv)

    def test_hybrid_path_records_the_reference_budget_and_validates_citations(self):
        documents = [
            {**PASSAGE, "id": f"guide-{n}", "title": f"Guide {n}", "score": 1.0/n}
            for n in range(1, 5)
        ]
        rankings = {"dense": documents, "bm25": documents,
                    "fused": reciprocal_rank_fusion(documents, documents, documents),
                    "timings": {"total_seconds": .001}}
        retriever = SimpleNamespace(passages=documents, dense=SimpleNamespace(), search_all=lambda q: rankings)
        with patch("hybrid_runtime.QwenTokenizer", return_value=SimpleNamespace(count=len)):
            runtime = PlanRuntime("api", "demo", "https://api.deepseek.com/v1", api_key=API_KEY)
        with patch("api_backend._opener.open", return_value=reply(valid_answer(("P1", "P99")))) as call:
            result = run_plan_query(retriever, "Check Python?", "X", runtime, generate=True)
        request_body = json.loads(call.call_args.args[0].data)
        self.assertEqual(request_body["max_tokens"], 512)
        self.assertIn("[P4] Guide 4", request_body["messages"][1]["content"])
        self.assertIn("estimate", result["budget_tokenizer"])
        self.assertEqual(result["status"], "citation_warning")
        self.assertEqual(result["invalid_citations"], ["P99"])
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["output_tokens"])

    def test_no_evidence_skips_the_paid_request(self):
        generator = create_generator("api", "demo", "https://api.openai.com/v1", api_key=API_KEY)
        retriever = SimpleNamespace(name="bm25", score_kind="BM25", min_score=0,
                                    passages=[PASSAGE], search=lambda question, k: [])
        with patch("api_backend._opener.open") as call:
            result = run_query(retriever, "unrelated", 3, generator)
        self.assertEqual(result["status"], "no_evidence")
        call.assert_not_called()

    def test_http_errors_do_not_show_service_response_or_retry(self):
        generator = create_generator("api", "demo", "https://api.openai.com/v1", api_key=API_KEY)
        for status, hint in ((307,"重定向"), (401,"密钥"), (429,"额度")):
            error = HTTPError(generator.endpoint, status, "server", {}, io.BytesIO(API_KEY.encode()))
            with self.subTest(status=status), patch("api_backend._opener.open", side_effect=error) as call:
                with self.assertRaisesRegex(ApiError, hint) as caught:
                    generator.answer("test", [PASSAGE])
                self.assertNotIn(API_KEY, str(caught.exception))
                call.assert_called_once()

    def test_partial_and_invalid_answers_are_not_successful(self):
        generator = create_generator("api", "demo", "https://api.openai.com/v1", api_key=API_KEY)
        with patch("api_backend._opener.open", return_value=reply('{"short_answer":"' + API_KEY, finish="length")):
            result = generator.answer("test", [PASSAGE])
        self.assertEqual(result["status"], "format_error")
        self.assertTrue(result["output_truncated"])
        self.assertIsNone(result["answer"])
        self.assertNotIn(API_KEY, result["raw_answer"])


if __name__ == "__main__":
    unittest.main()
