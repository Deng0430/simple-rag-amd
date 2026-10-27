import io
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from rag import (DEFAULT_LOCAL_MODEL, OllamaGenerator, _ollama_request,
                 create_generator, ollama_status)
from summarize_results import summarize


def chat_response(**changes):
    result = {
        "done": True, "done_reason": "stop", "message": {"content": "Use python3. [P1]"},
        "prompt_eval_count": 42, "eval_count": 8, "total_duration": 900_000_000,
        "load_duration": 100_000_000, "prompt_eval_duration": 200_000_000,
        "eval_duration": 500_000_000,
    }
    result.update(changes)
    return result


class FakeOllama:
    def __init__(self):
        self.calls = []
        self.response = chat_response()

    def __call__(self, base_url, path, payload=None, **kwargs):
        self.calls.append((path, payload, kwargs))
        if path == "/api/tags":
            return {"models": [{"name": DEFAULT_LOCAL_MODEL, "digest": "sha256:test",
                                "details": {"quantization_level": "Q4_K_M"}}]}
        if path == "/api/show":
            return {"details": {"quantization_level": "Q4_K_M", "parameter_size": "3.1B"}}
        if path == "/api/version":
            return {"version": "test-version"}
        if path == "/api/chat":
            return self.response
        raise AssertionError("Unexpected endpoint: " + path)


class LocalGeneratorTests(unittest.TestCase):
    def test_local_factory_uses_installed_model_without_transformers_or_download(self):
        server = FakeOllama()
        with patch("rag._ollama_request", side_effect=server), patch("rag.Generator") as remote:
            model = create_generator("ollama")
            result = model.answer("Which command?", [{"title": "Python", "text": "Use python3."}])
        remote.assert_not_called()
        self.assertEqual(result["answer"], "Use python3. [P1]")
        self.assertEqual(result["input_tokens"], 42)
        self.assertEqual(result["output_tokens"], 8)
        self.assertEqual(result["load_seconds"], 0.1)
        self.assertEqual(result["server_eval_seconds"], 0.5)
        self.assertIsNone(result["torch_peak_allocated_gib"])
        self.assertEqual(result["backend"], "ollama")
        self.assertEqual(result["model_digest"], "sha256:test")
        self.assertEqual(result["model_quantization"], "Q4_K_M")
        self.assertGreaterEqual(model.metadata()["warmup_seconds"], 0)
        chats = [payload for path, payload, _ in server.calls if path == "/api/chat"]
        self.assertEqual(len(chats), 2)  # Warm-up and the real answer.
        self.assertEqual(chats[-1]["options"], {"temperature": 0, "seed": 42,
                                               "num_ctx": 8192, "num_predict": 128})
        self.assertFalse(chats[-1]["stream"])
        self.assertIn("[P1] Python\nUse python3.", chats[-1]["messages"][-1]["content"])

    def test_missing_model_does_not_attempt_chat_or_download(self):
        server = FakeOllama()
        with patch("rag._ollama_request", side_effect=server):
            with self.assertRaisesRegex(RuntimeError, "未找到已安装模型"):
                OllamaGenerator("missing:3b")
        self.assertEqual([path for path, _, _ in server.calls], ["/api/tags"])

    def test_status_failure_is_read_only_and_does_not_disable_retrieval(self):
        with patch("rag._ollama_request", side_effect=RuntimeError("offline")) as request:
            status = ollama_status()
        self.assertFalse(status["available"])
        self.assertEqual(status["models"], [])
        self.assertEqual(status["error"], "offline")
        self.assertEqual(request.call_args.args[1], "/api/tags")
        self.assertEqual(request.call_args.kwargs["timeout"], 2)

    def test_unfinished_missing_counts_and_empty_answers_are_not_successes(self):
        server = FakeOllama()
        with patch("rag._ollama_request", side_effect=server):
            model = OllamaGenerator()
            for broken in (chat_response(done=False), chat_response(prompt_eval_count=None),
                           chat_response(message={"content": ""})):
                with self.subTest(response=broken):
                    server.response = broken
                    with self.assertRaises(RuntimeError):
                        model.answer("Question", [{"title": "Source", "text": "Fact"}])

    def test_context_and_output_limits_fail_before_request(self):
        server = FakeOllama()
        with patch("rag._ollama_request", side_effect=server):
            with self.assertRaises(ValueError):
                OllamaGenerator(num_ctx=0)
            model = OllamaGenerator(num_ctx=4096)
            count = len(server.calls)
            with self.assertRaisesRegex(ValueError, "未自动截断"):
                model.answer("问" * 2000, [{"title": "Source", "text": "Fact"}])
            for limit in (0, -1, 4097, True):
                with self.assertRaises(ValueError):
                    model.answer("Question", [], max_new_tokens=limit)
            self.assertEqual(len(server.calls), count)

    def test_length_stop_is_recorded_as_truncated_not_hidden(self):
        server = FakeOllama()
        with patch("rag._ollama_request", side_effect=server):
            model = OllamaGenerator()
            server.response = chat_response(done_reason="length")
            result = model.answer("Question", [{"title": "Source", "text": "Fact"}])
        self.assertTrue(result["output_truncated"])
        self.assertEqual(result["done_reason"], "length")

    def test_http_and_connection_errors_are_actionable(self):
        errors = [URLError("refused"), HTTPError("http://localhost", 404, "Not Found", {},
                                                io.BytesIO(b'{"error":"model not found"}'))]
        for error in errors:
            with self.subTest(error=error), patch("rag.urlopen", side_effect=error):
                with self.assertRaises(RuntimeError):
                    _ollama_request("http://127.0.0.1:11434", "/api/chat", {})

    def test_existing_backend_remains_default(self):
        with patch("rag.Generator") as remote:
            create_generator()
        remote.assert_called_once_with("Qwen/Qwen3-4B-Instruct-2507")

    def test_summary_accepts_unmeasured_memory_and_rejects_mixed_models(self):
        row = dict(mode="model", retriever="dense", backend="ollama", model="qwen2.5:3b",
                   model_digest="sha256:test", model_quantization="Q4_K_M", top_k="3",
                   manual_quality="", total_seconds="1", generation_seconds="0.9",
                   retrieval_seconds="0.1", input_tokens="42", output_tokens="8",
                   torch_peak_allocated_gib="", reference_passage_retrieved="")
        result = summarize([row])
        self.assertIsNone(result["groups"]["3"]["max_torch_peak_allocated_gib"])
        self.assertEqual(result["backend"], "ollama")
        for change in (dict(model="another"), dict(backend="transformers"),
                       dict(model_quantization="Q8_0"), dict(num_ctx="8192"),
                       dict(model_digest="new-weights")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Mixed models"):
                summarize([row, dict(row, **change)])


if __name__ == "__main__":
    unittest.main()
