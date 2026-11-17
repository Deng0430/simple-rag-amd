"""Evidence rendering, query preparation and strict response validation."""

import json
import re

ANSWER_FIELDS = {"short_answer", "explanation", "citations", "abstain"}


def render_passages(passages: list[dict], start_index: int = 1) -> str:
    """One presentation function for seeds and additions, including all overhead."""
    return "\n\n".join(
        f"[P{i}] {p['title']}\n"
        + (f"适用范围：{p['scope']}\n" if p.get("scope") else "") + p["text"]
        for i, p in enumerate(passages, start_index)
    )


def retrieval_query(question):
    # Output-format instructions are not part of the information need.
    return question.split("简短答案", 1)[0].strip()


def response_status(record):
    raw = record.get("raw_response")
    if (not isinstance(raw, dict) or raw.get("done") is not True or not raw.get("model")
            or type(raw.get("prompt_eval_count")) is not int or type(raw.get("eval_count")) is not int):
        return "incomplete_response"
    return "ok" if record.get("token_count_match") is True else "tokenizer_mismatch"


def _strip_outer_fence(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Strip one whole-response code fence, never extract a convenient substring.
    match = re.fullmatch(r"```[^\n`]*\n(.*?)\n```", value, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


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
