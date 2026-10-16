import csv
import io
import unittest
from unittest.mock import Mock

from rag import Retriever, export_csv, load_data, make_messages, run_query


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.retriever = Retriever(cls.data["passages"], **cls.data.get("retriever", {}))

    def test_reference_passages_are_present_and_k_changes_context(self):
        for q in self.data["questions"]:
            self.assertIn(q["passage_id"], {p["id"] for p in self.data["passages"]})
            small = self.retriever.search(q["question"], 3)
            large = self.retriever.search(q["question"], 6)
            self.assertIn(q["passage_id"], {p["id"] for p in small})
            self.assertLessEqual(len(small), 3)
            self.assertEqual([p["id"] for p in small], [p["id"] for p in large[:3]])

    def test_chinese_paraphrases_find_the_relevant_guide(self):
        cases = {
            "系统找不到 python 这个命令，要怎么办": "aup-python",
            "venv 创建失败，说 ensurepip 不可用": "aup-venv",
            "服务运行在远端，本机访问 localhost:8501 失败": "aup-proxy",
            "文件放在哪里，服务器关闭后还能留着？": "aup-storage",
            "关掉浏览器标签页会停止服务器吗？": "aup-session",
            "克隆时 destination path already exists，要删除文件夹吗？": "aup-clone-exists",
            "按启动快捷键没反应，怎么找到那个 task？": "aup-task",
            "依赖下载 ReadTimeoutError 卡住了": "aup-pip-timeout",
            "8501 提示 Address already in use，需要换端口吗？": "aup-port-busy",
            "权重缓存路径能放到 HF_HOME 指定目录吗？": "aup-model-cache",
            "HIP out of memory，empty_cache 能释放所有显存吗？": "aup-model-oom",
            "网页 502 Bad Gateway，是进程停止了吗？": "aup-proxy-failure",
            "runs 中的 CSV 为什么不在 GitHub 上？": "aup-backup",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertIn(expected, {p["id"] for p in self.retriever.search(question, 3)})

    def test_unrelated_question_does_not_call_the_model(self):
        generator = Mock(model_id="test-only")
        result = run_query(self.retriever, "今天北京的天气怎么样？", 3, generator)
        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["passages"], [])
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["input_tokens"])
        generator.answer.assert_not_called()

    def test_squad_questions_retrieve_their_reference_passages(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        data = load_data(root / "data/squad_demo.json")
        retriever = Retriever(data["passages"])
        for question in data["questions"]:
            small = retriever.search(question["question"], 3)
            large = retriever.search(question["question"], 6)
            self.assertEqual(small[0]["id"], question["passage_id"])
            self.assertEqual([p["id"] for p in small], [p["id"] for p in large[:3]])

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("  ", 3)

    def test_preview_does_not_claim_generated_answer_or_tokens(self):
        result = run_query(self.retriever, self.data["questions"][0]["question"], 3)
        self.assertEqual(result["mode"], "retrieval_preview")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["input_tokens"])
        self.assertGreaterEqual(result["retrieval_seconds"], 0)
        rows = list(csv.DictReader(io.StringIO(export_csv([result]).decode("utf-8-sig"))))
        self.assertEqual(rows[0]["answer"], "")
        self.assertEqual(rows[0]["question"], result["question"])

    def test_prompt_uses_only_question_and_retrieved_text(self):
        messages = make_messages("A question?", [{"title": "Source", "text": "A fact."}])
        self.assertIn("[P1] Source\nA fact.", messages[-1]["content"])
        self.assertNotIn("reference_answers", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
