"""Deterministic answer metrics and aggregation of explicitly supplied reviews.

The token F1 here is a course-specific Chinese/command overlap metric, not the
original English SQuAD scorer. Slot labels must come from a documented review;
this module never infers semantic correctness from a model's self-report.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Iterable


METRIC_VERSION = "condition-metrics-v1"
ANSWER_TYPES = {"text", "command", "path"}
ANSWER_FIELDS = {"short_answer", "explanation", "citations", "abstain"}


def _strip_outer_fence(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Strip one whole-response code fence, never extract a convenient substring.
    match = re.fullmatch(r"```[^\n`]*\n(.*?)\n```", value, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


def normalize_answer(value: str, answer_type: str = "text") -> str:
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"Unsupported answer_type: {answer_type}")
    if not isinstance(value, str):
        raise TypeError("Answers must be strings.")
    value = _strip_outer_fence(value)
    return " ".join(value.split()) if answer_type == "text" else value


def tokenize_answer(value: str, answer_type: str = "text") -> list[str]:
    """ASCII word runs; individual Han and other non-whitespace characters.

    All punctuation, case, path separators and repeated operators are retained.
    As a bag-of-tokens metric this still cannot determine shell semantics.
    """
    return re.findall(r"[A-Za-z0-9_]+|[^\s]", normalize_answer(value, answer_type))


def token_f1(prediction: str, reference: str, answer_type: str = "text") -> float:
    predicted = tokenize_answer(prediction, answer_type)
    expected = tokenize_answer(reference, answer_type)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    return 2.0 * overlap / (len(predicted) + len(expected))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def parse_answer(raw: str) -> dict:
    """Validate one complete JSON response without repairing or regenerating it.

    ``parse_ok`` means JSON and the four field types are valid. Explicit refusal
    conflicts are separately flagged. Semantic refusal/action contradictions
    require a documented review of the preserved raw response.
    """
    result = {
        "raw": raw, "parsed": None, "parse_ok": False, "format_error": None,
        "abstain_conflict": False, "short_answer": "", "abstain": None,
        "flags": [],
    }
    if not isinstance(raw, str):
        result.update(format_error="raw_not_string", flags=["format_error"])
        return result
    try:
        parsed = json.loads(_strip_outer_fence(raw), object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        result.update(format_error=f"invalid_json: {exc}", flags=["format_error"])
        return result
    result["parsed"] = parsed
    if not isinstance(parsed, dict):
        result.update(format_error="response_not_object", flags=["format_error"])
        return result
    if set(parsed) != ANSWER_FIELDS:
        missing, extra = sorted(ANSWER_FIELDS - set(parsed)), sorted(set(parsed) - ANSWER_FIELDS)
        result.update(format_error=f"schema_fields: missing={missing}, extra={extra}",
                      flags=["format_error"])
        return result
    valid = (isinstance(parsed["short_answer"], str)
             and isinstance(parsed["explanation"], str)
             and isinstance(parsed["citations"], list)
             and all(isinstance(item, str) for item in parsed["citations"])
             and type(parsed["abstain"]) is bool)
    if not valid:
        result.update(format_error="schema_field_types", flags=["format_error"])
        return result
    result.update(parse_ok=True, short_answer=parsed["short_answer"], abstain=parsed["abstain"])
    if parsed["abstain"]:
        result["flags"].append("abstain")
        if parsed["short_answer"].strip():
            result["abstain_conflict"] = True
            result["flags"].append("abstain_conflict")
    elif not parsed["short_answer"].strip():
        result["flags"].append("empty_answer")
    return result


def score_answer(raw: str, reference_answers: Iterable[str], answer_type: str = "text") -> dict:
    if isinstance(reference_answers, (str, bytes)):
        raise ValueError("reference_answers must be a collection of nonempty strings.")
    references = list(reference_answers)
    if not references or any(not isinstance(item, str) for item in references):
        raise ValueError("At least one nonempty reference answer is required.")
    normalized = [normalize_answer(item, answer_type) for item in references]
    if any(not item for item in normalized):
        raise ValueError("Empty references are unsupported in this answerable-only test set.")
    result = parse_answer(raw)
    result.update(em=0, token_f1=0.0, metric_version=METRIC_VERSION)
    if not result["parse_ok"] or result["abstain"] or result["abstain_conflict"]:
        return result
    prediction = normalize_answer(result["short_answer"], answer_type)
    if not prediction:
        return result
    result["em"] = int(prediction in normalized)
    result["token_f1"] = max(token_f1(result["short_answer"], reference, answer_type)
                              for reference in references)
    return result


def aggregate_review(rows: Iterable[dict]) -> dict:
    """Aggregate supplied TP/FP/FN labels, preserving their review provenance.

    Every row must declare ``review_type`` (``human`` or ``ai_assisted``) and
    nonnegative integer ``tp``, ``fp``, ``fn``. Optional boolean ``critical_error``,
    ``acceptable`` and ``refusal`` fields are aggregated only when fully reviewed.
    Partial labels remain visibly incomplete instead of becoming zero errors.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("At least one explicitly labelled review is required.")
    types = {row.get("review_type") for row in rows}
    if len(types) != 1 or not types.issubset({"human", "ai_assisted"}):
        raise ValueError("Review types must be explicit and cannot mix human with ai_assisted.")
    for row in rows:
        for key in ("tp", "fp", "fn"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Review {key} must be a nonnegative integer, not inferred text.")
        for key in ("critical_error", "acceptable", "refusal"):
            if row.get(key) is not None and type(row[key]) is not bool:
                raise ValueError(f"Review {key} must be a boolean or null.")
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    result = {
        "review_type": next(iter(types)), "reviewed_count": len(rows),
        "tp": tp, "fp": fp, "fn": fn,
        "slot_precision": tp / (tp + fp) if tp + fp else 0.0,
        "slot_recall": tp / (tp + fn) if tp + fn else None,
        "slot_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
    for key in ("critical_error", "acceptable", "refusal"):
        labelled = [row[key] for row in rows if row.get(key) is not None]
        result[key + "_reviewed_count"] = len(labelled)
        result[key + "_count"] = sum(labelled)
        result[key + "_rate"] = sum(labelled) / len(rows) if len(labelled) == len(rows) else None
    return result
