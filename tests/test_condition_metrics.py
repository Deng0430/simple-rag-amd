import json
import unittest

from condition_metrics import (aggregate_review, normalize_answer, parse_answer,
                               score_answer, token_f1, tokenize_answer)


def response(answer="python3", **overrides):
    fields = dict(short_answer=answer, explanation="Use the documented environment.",
                  citations=["P1"], abstain=False)
    fields.update(overrides)
    return json.dumps(fields, ensure_ascii=False)


class ConditionMetricTests(unittest.TestCase):
    def test_multiple_references_take_best_exact_and_overlap(self):
        result = score_answer(response("python3 --version"),
                              ["python --version", "python3 --version"], "command")
        self.assertEqual(result["em"], 1)
        self.assertEqual(result["token_f1"], 1)
        self.assertTrue(result["parse_ok"])

    def test_chinese_tokens_and_repeated_operators_preserve_multiplicity(self):
        self.assertEqual(tokenize_answer("进入 /Home/A && python3 -m venv"),
                         ["进", "入", "/", "Home", "/", "A", "&", "&", "python3", "-", "m", "venv"])
        score = score_answer(response("cd /tmp & run"), ["cd /tmp && run"], "command")
        self.assertEqual(score["em"], 0)
        self.assertLess(score["token_f1"], 1)
        self.assertAlmostEqual(token_f1("a a", "a"), 2 / 3)

    def test_case_path_slash_quotes_and_parameters_are_not_erased(self):
        for wrong, expected in [("/Home/User", "/home/user"), ("home/user", "/home/user"),
                                ('echo "A"', "echo A"), ("run -f", "run --f")]:
            with self.subTest(wrong=wrong):
                score = score_answer(response(wrong), [expected], "command")
                self.assertEqual(score["em"], 0)
                self.assertLess(score["token_f1"], 1)

    def test_only_text_em_collapses_internal_whitespace(self):
        self.assertEqual(normalize_answer(" A  B\r\nC ", "text"), "A B C")
        self.assertEqual(normalize_answer(" A  B\r\nC ", "command"), "A  B\nC")
        self.assertEqual(score_answer(response("a  b"), ["a b"], "command")["em"], 0)
        self.assertEqual(score_answer(response("a  b"), ["a b"], "text")["em"], 1)
        # Ignoring spaces in F1 is disclosed; it cannot validate command syntax.
        self.assertEqual(token_f1("a  b", "a b", "command"), 1)

    def test_one_whole_code_fence_is_allowed_but_prose_is_not_extracted(self):
        raw = "```json\n" + response("```bash\npython3\n```") + "\n```"
        self.assertTrue(parse_answer(raw)["parse_ok"])
        self.assertEqual(score_answer(raw, ["python3"], "command")["em"], 1)
        self.assertEqual(score_answer("```json\n" + response() + "\n```", ["python3"])["em"], 1)
        self.assertEqual(score_answer(response("```bash\npython3\n```"), ["python3"], "command")["em"], 1)
        self.assertFalse(parse_answer("Here is the answer: " + response())["parse_ok"])
        self.assertFalse(parse_answer(response() + response())["parse_ok"])

    def test_json_schema_types_and_duplicate_keys_fail(self):
        cases = ["[]", "null", response(abstain=0), response(citations="P1"),
                 response(citations=[1]), response(short_answer=3),
                 response(extra="unexpected"), '{"short_answer":"a","short_answer":"b"}']
        for raw in cases:
            with self.subTest(raw=raw):
                result = score_answer(raw, ["python3"])
                self.assertFalse(result["parse_ok"])
                self.assertEqual(result["em"], 0)
                self.assertEqual(result["token_f1"], 0)
                self.assertEqual(result["raw"], raw)

    def test_refusal_and_refusal_conflict_are_zero_without_discarding_raw(self):
        refused = score_answer(response("", abstain=True), ["python3"])
        self.assertTrue(refused["parse_ok"])
        self.assertFalse(refused["abstain_conflict"])
        self.assertEqual(refused["token_f1"], 0)
        raw = response("python3", abstain=True)
        conflict = score_answer(raw, ["python3"])
        self.assertTrue(conflict["abstain_conflict"])
        self.assertEqual(conflict["em"], 0)
        self.assertEqual(conflict["raw"], raw)

    def test_invalid_references_and_answer_types_are_rejected(self):
        for references in ([], [""], [None], "python3"):
            with self.subTest(references=references), self.assertRaises(ValueError):
                score_answer(response(), references)
        with self.assertRaises(ValueError):
            normalize_answer("a", "unknown")
        self.assertEqual(token_f1("", ""), 0)
        self.assertEqual(token_f1("a", "b"), 0)

    def test_slot_aggregation_requires_explicit_review_not_model_claims(self):
        rows = [dict(review_type="ai_assisted", tp=1, fp=1, fn=1, critical_error=True),
                dict(review_type="ai_assisted", tp=2, fp=0, fn=0, critical_error=False)]
        result = aggregate_review(rows)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (3, 1, 1))
        self.assertEqual(result["slot_f1"], 0.75)
        self.assertEqual(result["critical_error_rate"], 0.5)
        self.assertIsNone(result["acceptable_rate"])
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            aggregate_review([rows[0], dict(rows[1], review_type="human")])
        with self.assertRaises(ValueError):
            aggregate_review([dict(tp=1, fp=0, fn=0)])
        with self.assertRaises(ValueError):
            aggregate_review([dict(rows[0], tp=True)])

    def test_partial_review_is_not_treated_as_zero_errors(self):
        result = aggregate_review([
            dict(review_type="human", tp=0, fp=0, fn=2, critical_error=False),
            dict(review_type="human", tp=1, fp=0, fn=0),
        ])
        self.assertEqual(result["critical_error_reviewed_count"], 1)
        self.assertIsNone(result["critical_error_rate"])
        self.assertEqual(result["slot_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
