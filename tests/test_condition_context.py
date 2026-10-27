import unittest

from condition_context import build_contexts, has_condition_cue, render_passages


def passage(n, text="ordinary content", *, section="s", document="d", order=0):
    return dict(id=f"{document}-{section}-{n}", title="Guide", text=text, scope="",
                document_id=document, document_order=order, section_id=section,
                section_index=0 if section == "s" else 1, chunk_index=n)


def prompt(items):
    return "SYSTEM\n" + render_passages(items) + "\nQUESTION"


class ConditionPackingTests(unittest.TestCase):
    def test_pool_is_local_unique_and_excludes_all_seeds(self):
        data = [passage(i) for i in range(9)]
        data += [passage(4, section="other"), passage(4, document="other", order=1)]
        seeds = [data[4], data[5], data[8]]
        packed = build_contexts(seeds, data, len, prompt, extension_budget=1000)
        expected = {"d-s-2", "d-s-3", "d-s-6", "d-s-7"}
        for group in packed.values():
            self.assertEqual(set(group["candidate_ids"]), expected)
            self.assertEqual(group["seed_ids"], [p["id"] for p in seeds])
            self.assertEqual(group["passages"][:3], seeds)
        self.assertEqual(len(packed["B"]["candidate_relations"]["d-s-6"]), 3)

    def test_b_prioritizes_cues_c_uses_distance_but_both_keep_source_order(self):
        data = [passage(0, "Before using this, prepare it."),
                passage(1, "Some ordinary nearby text here."), passage(2, "Seed content.")]
        # Exactly one candidate fits. B chooses the condition at distance 2;
        # C chooses the ordinary nearer candidate at distance 1.
        budget = max(len(render_passages([p], 2)) for p in data[:2])
        packed = build_contexts([data[2]], data, len, prompt, extension_budget=budget)
        self.assertEqual(packed["B"]["extra_ids"], [data[0]["id"]])
        self.assertEqual(packed["C"]["extra_ids"], [data[1]["id"]])
        self.assertNotIn("identical_context", packed["B"]["flags"])
        self.assertLessEqual(packed["B"]["extension_tokens"], budget)
        self.assertLessEqual(packed["C"]["extension_tokens"], budget)

    def test_oversize_candidate_is_skipped_and_later_complete_candidate_is_used(self):
        data = [passage(0, "if " + "x" * 1000), passage(1, "small"), passage(2, "seed")]
        packed = build_contexts([data[2]], data, len, prompt, extension_budget=60)
        self.assertEqual(packed["B"]["extra_ids"], [data[1]["id"]])
        self.assertEqual(packed["B"]["budget_skipped"][0]["id"], data[0]["id"])
        self.assertEqual(packed["B"]["passages"][-1]["text"], "small")

    def test_expansion_is_rendered_in_source_order_after_selection(self):
        data = [passage(0, "ordinary"), passage(1, "seed"), passage(2, "Must prepare this.")]
        packed = build_contexts([data[1]], data, len, prompt, extension_budget=200)
        # B selects the later condition first but renders additions in source order.
        self.assertEqual(packed["B"]["extra_ids"], [data[0]["id"], data[2]["id"]])
        self.assertTrue(packed["B"]["prompt"].index("[P2]") < packed["B"]["prompt"].index("[P3]"))
        self.assertEqual(packed["B"]["passages"][0], data[1])

    def test_no_neighbors_produce_identical_groups_without_inventing_text(self):
        data = [passage(0, "Only one passage.")]
        packed = build_contexts(data, data, len, prompt)
        self.assertEqual(len({p["prompt"] for p in packed.values()}), 1)
        for item in packed.values():
            self.assertEqual(item["extra_ids"], [])
            self.assertEqual(item["extension_tokens"], 0)
            self.assertIn("no_neighbor", item["flags"])
            self.assertIn("identical_context", item["flags"])

    def test_context_limit_includes_full_prompt_and_output_and_preserves_seeds(self):
        data = [passage(0, "seed"), passage(1, "if extra")]
        seed_tokens = len(prompt(data[:1]))
        packed = build_contexts(data[:1], data, len, prompt, extension_budget=100,
                                num_ctx=seed_tokens + 10, max_new_tokens=10)
        for group in packed.values():
            self.assertEqual(group["status"], "ready")
            self.assertEqual(group["extra_ids"], [])
        blocked = build_contexts(data[:1], data, len, prompt, num_ctx=seed_tokens + 9,
                                 max_new_tokens=10)
        for group in blocked.values():
            self.assertEqual(group["status"], "input_budget_exceeded")
            self.assertEqual(group["passages"], data[:1])

    def test_english_words_are_bounded_and_chinese_cues_are_literal(self):
        self.assertFalse(has_condition_cue(passage(0, "different requirement onlyness")))
        self.assertTrue(has_condition_cue(passage(0, "ONLY use it if installed.")))
        self.assertTrue(has_condition_cue(passage(0, "启动之前需要安装。")))

    def test_content_mismatch_and_duplicate_seeds_fail_closed(self):
        data = [passage(0)]
        with self.assertRaisesRegex(ValueError, "content differs"):
            build_contexts([dict(data[0], text="replaced by a gold answer")], data, len, prompt)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_contexts(data * 2, data, len, prompt)

    def test_empty_seeds_are_not_silently_filled(self):
        packed = build_contexts([], [passage(0)], len, prompt)
        for group in packed.values():
            self.assertEqual(group["status"], "no_evidence")
            self.assertEqual(group["passages"], [])


if __name__ == "__main__":
    unittest.main()
