import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from condition_experiment import (apply_review_overrides, nullable_mean, unique_records, validate_gold,
                                 validate_schedule, verify_scoring_sources)
from condition_model import sha256_file


def fixture():
    questions = [{"id": "q1", "answer_type": "command"}]
    gold = {"items": [{"id": "q1", "answer_type": "command",
                       "reference_answers": ["python3"],
                       "required_slots": [{"id": "python3", "description": "Python 3 command",
                                           "acceptable": ["python3"], "evidence_ids": ["p1"]}]}]}
    return questions, gold


class ConditionRunnerTests(unittest.TestCase):
    def test_gold_covers_the_exact_task_set_without_duplicates(self):
        questions, gold = fixture()
        self.assertEqual(set(validate_gold(questions, gold)), {"q1"})
        duplicate = copy.deepcopy(gold)
        duplicate["items"] *= 2
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_gold(questions, duplicate)
        missing = {"items": []}
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_gold(questions, missing)
        extra = copy.deepcopy(gold)
        extra["items"].append(dict(extra["items"][0], id="q2"))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_gold(questions, extra)

    def test_empty_answers_or_invalid_slot_annotations_fail_before_model_use(self):
        questions, gold = fixture()
        for field, replacement in (("reference_answers", []), ("reference_answers", [""]),
                                   ("required_slots", []), ("answer_type", "unknown")):
            changed = copy.deepcopy(gold)
            changed["items"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_gold(questions, changed)
        for field, replacement in (("acceptable", []), ("evidence_ids", []), ("description", "")):
            changed = copy.deepcopy(gold)
            changed["items"][0]["required_slots"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_gold(questions, changed)

    def test_resume_keys_cannot_be_overwritten_or_imported_from_another_run(self):
        records = [{"key": "q1:A", "status": "ok"}]
        self.assertEqual(set(unique_records(records, "predictions", {"q1:A"})), {"q1:A"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            unique_records(records * 2, "predictions", {"q1:A"})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            unique_records(records, "predictions", {"q2:A"})

    def test_schedule_requires_one_position_per_input(self):
        inputs = {"q1:A": {}, "q1:B": {}, "q1:C": {}}
        validate_schedule(["q1:C", "q1:A", "q1:B"], inputs)
        for schedule in (["q1:A", "q1:A", "q1:C"], ["q1:A"], ["q1:A", "q1:B", "q2:C"]):
            with self.assertRaises(ValueError):
                validate_schedule(schedule, inputs)

    def test_missing_token_counts_remain_unmeasured(self):
        self.assertIsNone(nullable_mean([]))
        self.assertIsNone(nullable_mean([None, None]))
        self.assertEqual(nullable_mean([None, 2, 4]), 3)

    def test_analysis_rejects_scorer_changes_after_freezing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("condition_metrics.py", "condition_experiment.py"):
                (root / name).write_text("frozen code", encoding="utf-8")
            hashes = {p.name: sha256_file(p) for p in root.iterdir()}
            with patch("condition_experiment.ROOT", root):
                self.assertEqual(verify_scoring_sources({"source_hashes": hashes}), hashes)
                (root / "condition_metrics.py").write_text("new scoring rule", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Scoring source changed"):
                    verify_scoring_sources({"source_hashes": hashes})

    def test_semantic_refusal_and_conflict_only_override_separate_reviewed_scores(self):
        automatic = [{"key": "q1:A", "em": 1, "token_f1": 1.0},
                     {"key": "q1:B", "em": 1, "token_f1": 1.0},
                     {"key": "q1:C", "em": 1, "token_f1": 1.0}]
        reviews = [{"review_id": "R1", "review_type": "ai_assisted", "refusal": True},
                   {"review_id": "R2", "review_type": "ai_assisted", "refusal": False,
                    "abstain_conflict": True},
                   {"review_id": "R3", "review_type": "ai_assisted", "refusal": False}]
        key = {"R1": "q1:A", "R2": "q1:B", "R3": "q1:C"}
        reviewed = apply_review_overrides(automatic, reviews, key)
        self.assertEqual([r["em"] for r in reviewed], [0, 0, 1])
        self.assertEqual([r["token_f1"] for r in reviewed], [0, 0, 1.0])
        self.assertEqual([r["em"] for r in automatic], [1, 1, 1])
        self.assertEqual(reviewed[0]["automatic_em"], 1)


if __name__ == "__main__":
    unittest.main()
