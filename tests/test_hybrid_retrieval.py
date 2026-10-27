import unittest

from condition_context import render_passages
from hybrid_retrieval import HybridRetriever, build_contexts, reciprocal_rank_fusion


def passage(pid, text="short text", doc="guide", section="section"):
    return {"id": pid, "title": "Guide", "text": text, "scope": "",
            "document_id": doc, "section_id": section}


def scored(p, score=1.0):
    return dict(p, score=score)


def search_result(dense, bm25, corpus):
    return {"dense": dense, "bm25": bm25,
            "fused": reciprocal_rank_fusion(dense, bm25, corpus)}


def prompt(items):
    return "SYSTEM\n" + render_passages(items) + "\nQUESTION"


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_uses_rank_not_incompatible_score_magnitudes(self):
        a, b, c = data = [passage(x) for x in "abc"]
        result = reciprocal_rank_fusion([scored(a, 0.9), scored(b, 0.8)],
                                        [scored(b, 1000), scored(c, 999)], data)
        self.assertEqual([p["id"] for p in result], ["b", "a", "c"])
        self.assertAlmostEqual(result[0]["rrf_score"], 1/62 + 1/61)
        self.assertEqual(result[0]["dense_rank"], 2)
        self.assertEqual(result[0]["bm25_rank"], 1)
        self.assertAlmostEqual(result[1]["rrf_contributions"]["bm25"], 0)

    def test_exact_ties_use_corpus_order_and_candidates_are_union_only(self):
        a, b, unused = data = [passage(x) for x in ["a", "b", "unused"]]
        fused = reciprocal_rank_fusion([scored(b)], [scored(a)], data)
        self.assertEqual([p["id"] for p in fused], ["a", "b"])
        self.assertNotIn("unused", [p["id"] for p in fused])

    def test_unknown_duplicates_changed_text_and_nonfinite_scores_are_rejected(self):
        data = [passage("a")]
        for ranked, message in [([data[0], data[0]], "unique"),
                                ([passage("unknown")], "Unknown"),
                                ([dict(data[0], text="injected")], "differs"),
                                ([scored(data[0], float("nan"))], "Nonfinite")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                reciprocal_rank_fusion(ranked, [], data)

    def test_search_all_uses_same_query_fixed_top20_and_existing_components(self):
        data = [passage(x) for x in "ab"]
        class Fake:
            model_id = "fake"
            artifact = {"revision": "fixed"}
            def __init__(self, rows): self.rows, self.calls = rows, []
            def search(self, q, k): self.calls.append((q, k)); return self.rows
        dense, bm25 = Fake([scored(data[0])]), Fake([scored(data[1])])
        retriever = HybridRetriever(data, dense=dense, bm25=bm25)
        got = retriever.search_all("generic query")
        self.assertEqual(dense.calls, [("generic query", 20)])
        self.assertEqual(bm25.calls, dense.calls)
        self.assertEqual(got["config"]["rrf_k"], 60)
        self.assertEqual(len(got["fused"]), 2)
        self.assertGreaterEqual(got["timings"]["total_seconds"], 0)

    def test_top3_and_global_expansion_keep_rank_order_across_sections(self):
        # Corpus ordering is deliberately different from ranking ordering.
        data = [passage(str(i), doc=f"d{i}", section=f"s{i}") for i in range(7)]
        ranks = [scored(data[i]) for i in [4, 2, 5, 6, 0, 3, 1]]
        sr = search_result(ranks, ranks, data)
        result = build_contexts(sr, data, len, prompt, extension_budget=1000)
        self.assertEqual(result["H"]["seed_ids"], ["4", "2", "5"])
        self.assertEqual(result["X"]["seed_ids"], result["H"]["seed_ids"])
        self.assertEqual(result["X"]["extra_ids"], ["6", "0", "3", "1"])
        self.assertEqual(result["X"]["passages"][:3], result["H"]["passages"])
        self.assertEqual(result["D"]["extra_ids"], [])
        self.assertFalse(result["X"]["hx_identical"])

    def test_long_candidate_skipped_without_truncating_and_later_candidate_can_fit(self):
        data = [passage(str(i)) for i in range(5)]
        data[3]["text"] = "x" * 1000
        ranks = [scored(p) for p in data]
        result = build_contexts(search_result(ranks, ranks, data), data, len, prompt,
                                extension_budget=40)
        self.assertEqual(result["X"]["extra_ids"], ["4"])
        self.assertEqual(result["X"]["budget_skipped"][0]["id"], "3")
        self.assertEqual(result["X"]["passages"][-1]["text"], data[4]["text"])
        self.assertLessEqual(result["X"]["extension_tokens"], 40)

    def test_window_limit_includes_full_prompt_and_output_reserve(self):
        data = [passage(str(i)) for i in range(4)]
        ranks = [scored(p) for p in data]
        sr = search_result(ranks, ranks, data)
        seed_tokens = len(prompt(data[:3]))
        exact = build_contexts(sr, data, len, prompt, num_ctx=seed_tokens+10,
                               max_new_tokens=10)
        self.assertEqual(exact["X"]["status"], "ready")
        self.assertEqual(exact["X"]["extra_ids"], [])
        self.assertIn("context_window", exact["X"]["budget_skipped"][0]["reasons"])
        over = build_contexts(sr, data, len, prompt, num_ctx=seed_tokens+9,
                              max_new_tokens=10)
        self.assertTrue(all(r["status"] == "input_budget_exceeded" for r in over.values()))
        self.assertEqual(over["X"]["passages"], over["H"]["passages"])

    def test_empty_search_does_not_fill_from_corpus_or_invent_neighbors(self):
        data = [passage("unused")]
        result = build_contexts(search_result([], [], data), data, len, prompt)
        for r in result.values():
            self.assertEqual(r["status"], "no_evidence")
            self.assertEqual(r["passages"], [])
            self.assertEqual(r["input_tokens"], len(prompt([])))

    def test_tampered_fused_rank_and_excess_candidate_count_fail_closed(self):
        data = [passage(str(i)) for i in range(22)]
        ranks = [scored(p) for p in data[:4]]
        sr = search_result(ranks, ranks, data)
        sr["fused"] = list(reversed(sr["fused"]))
        with self.assertRaisesRegex(ValueError, "differs from fixed RRF"):
            build_contexts(sr, data, len, prompt)
        long = [scored(p) for p in data]
        with self.assertRaisesRegex(ValueError, "top-20"):
            build_contexts(search_result(long, long, data), data, len, prompt)


if __name__ == "__main__":
    unittest.main()
