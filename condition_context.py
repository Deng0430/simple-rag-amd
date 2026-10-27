"""Audited, local-only corpus positions and answer-blind A/B/C context packing.

No question annotations, reference answers, or generated answers are read here.
The callbacks supplied by the experiment runner define the exact prompt and its
tokenizer; this module never approximates tokenizer counts with character counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import knowledge
from prepare_knowledge import extract_sections


RULE_VERSION = "condition-neighbors-v1"
ENGLISH_CUES = (
    "if", "before", "only", "require", "requires", "required", "must",
    "prerequisite", "prerequisites",
)
CHINESE_CUES = ("如果", "之前", "仅", "需要", "必须", "前提", "确保")
ENGLISH_PATTERN = re.compile(r"\b(?:" + "|".join(ENGLISH_CUES) + r")\b", re.IGNORECASE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _section_metadata(document_id, document_order, section_index, title, anchor, chunk_index):
    return {
        "document_order": document_order,
        "section_id": f"{document_id}:section:{section_index}",
        "section_index": section_index,
        "section_title": title,
        "section_anchor": anchor,
        "chunk_index": chunk_index,
    }


def load_experiment_corpus() -> dict:
    """Reconstruct positions and verify every cached official ID and text.

    Reads existing cache files only. The current project guide is included as its
    own last document. It does not rewrite the application's cache or refresh any
    URL. Failure to reproduce a cached official passage aborts the experiment.
    """
    cached = json.loads(knowledge.CORPUS.read_text(encoding="utf-8"))
    manifest = json.loads(knowledge.MANIFEST.read_text(encoding="utf-8"))
    existing = {p["id"]: p for p in cached["passages"]}
    if len(existing) != len(cached["passages"]):
        raise ValueError("Cached corpus contains duplicate passage IDs.")
    source_order = {s["id"]: i for i, s in enumerate(manifest["sources"])}
    cached_documents = {d["id"]: d for d in cached["documents"]}
    if len(cached_documents) != len(cached["documents"]):
        raise ValueError("Cached corpus contains duplicate document IDs.")

    passages, documents, reconstructed_ids = [], [], set()
    sections_audit, split_code_sections, page_hashes = [], [], {}
    for source in manifest["sources"]:
        document_id = source["id"]
        if document_id not in cached_documents:
            continue
        path = knowledge.CORPUS.parent / "pages" / f"{document_id}.html"
        html = path.read_text(encoding="utf-8")
        page_hashes[document_id] = _sha256(html)
        sections = extract_sections(html)
        document_order = source_order[document_id]
        for section_index, (title, anchor, text) in enumerate(sections):
            chunks = knowledge.chunk_section(text)
            section_id = f"{document_id}:section:{section_index}"
            sections_audit.append({"document_id": document_id, "section_id": section_id,
                                   "section_title": title, "chunk_count": len(chunks)})
            # The original importer can split a very long fenced source block.
            # Surface that limitation without changing the frozen source text.
            if any(len(block[3:-3].strip()) > 992
                   for block in re.findall(r"```[\s\S]*?```", text)):
                split_code_sections.append(section_id)
            for chunk_index, chunk in enumerate(chunks):
                digest = _sha256(f"{document_id}:{section_index}:{chunk_index}")[:12]
                passage_id = "doc-" + digest
                old = existing.get(passage_id)
                if old is None or old["text"] != chunk or old["document_id"] != document_id:
                    raise ValueError(f"Cached HTML cannot reproduce official passage {passage_id}.")
                if old["title"] != source["title"] + " / " + title:
                    raise ValueError(f"Official passage title changed: {passage_id}.")
                passages.append(dict(old, **_section_metadata(
                    document_id, document_order, section_index, title, anchor, chunk_index)))
                reconstructed_ids.add(passage_id)
        documents.append(dict(cached_documents[document_id], document_order=document_order,
                              cached_html_sha256=page_hashes[document_id]))

    if reconstructed_ids != set(existing):
        missing = sorted(set(existing) - reconstructed_ids)
        raise ValueError(f"Unreconstructed official passages: {missing[:5]}.")
    official_count = len(passages)
    project_text = knowledge.PROJECT_GUIDE.read_text(encoding="utf-8")
    project_by_id = {p["id"]: p for p in knowledge.project_passages()}
    project_order = len(manifest["sources"])
    project_ids = set()
    for section_index, part in enumerate(re.split(r"(?m)^## ", project_text)[1:]):
        title, _, body = part.partition("\n")
        chunks = knowledge.chunk_section(body, 750)
        section_id = f"project-usage:section:{section_index}"
        sections_audit.append({"document_id": "project-usage", "section_id": section_id,
                               "section_title": title, "chunk_count": len(chunks)})
        if any(len(block[3:-3].strip()) > 742
               for block in re.findall(r"```[\s\S]*?```", body)):
            split_code_sections.append(section_id)
        for chunk_index, chunk in enumerate(chunks):
            passage_id = "project-" + _sha256(title + str(chunk_index + 1))[:10]
            old = project_by_id.get(passage_id)
            if old is None or old["text"] != chunk:
                raise ValueError(f"Project guide changed while positions were being rebuilt: {passage_id}.")
            passages.append(dict(old, **_section_metadata(
                "project-usage", project_order, section_index, title, "", chunk_index)))
            project_ids.add(passage_id)
    if project_ids != set(project_by_id):
        raise ValueError("Unreconstructed project passages or duplicate section titles.")
    if len({p["id"] for p in passages}) != len(passages):
        raise ValueError("Final corpus contains duplicate IDs; check project section titles.")
    documents.append({"id": "project-usage", "title": "本项目使用说明",
                      "document_order": project_order, "sha256": _sha256(project_text),
                      "chunk_count": len(project_ids)})

    counts = Counter(p["section_id"] for p in passages)
    by_document = {}
    for document in documents:
        items = [p for p in passages if p["document_id"] == document["id"]]
        by_document[document["id"]] = {
            "passages": len(items), "sections": len({p["section_id"] for p in items}),
            "passages_with_same_section_neighbor": sum(counts[p["section_id"]] > 1 for p in items),
        }
    return {
        "passages": passages,
        "documents": documents,
        "audit": {
            "official_passages_verified": official_count,
            "project_passages": len(project_ids), "total_passages": len(passages),
            "sections": len(counts), "multi_chunk_sections": sum(n > 1 for n in counts.values()),
            "passages_with_same_section_neighbor": sum(n for n in counts.values() if n > 1),
            "by_document": by_document, "section_details": sections_audit,
            "source_code_split_sections": split_code_sections,
            "unbalanced_fence_ids": [p["id"] for p in passages if p["text"].count("```") % 2],
            "cached_corpus_sha256": hashlib.sha256(knowledge.CORPUS.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(knowledge.MANIFEST.read_bytes()).hexdigest(),
            "project_guide_sha256": hashlib.sha256(knowledge.PROJECT_GUIDE.read_bytes()).hexdigest(),
            "cached_page_text_hashes": page_hashes,
        },
    }


def render_passages(passages: list[dict], start_index: int = 1) -> str:
    """One presentation function for seeds and additions, including all overhead."""
    return "\n\n".join(
        f"[P{i}] {p['title']}\n"
        + (f"适用范围：{p['scope']}\n" if p.get("scope") else "") + p["text"]
        for i, p in enumerate(passages, start_index)
    )


def has_condition_cue(passage: dict) -> bool:
    text = passage["title"] + "\n" + passage["text"]
    return bool(ENGLISH_PATTERN.search(text)) or any(cue in text for cue in CHINESE_CUES)


def _source_key(passage: dict) -> tuple:
    return (passage["document_order"], passage["section_index"], passage["chunk_index"], passage["id"])


def _validated_seeds(seeds, passages):
    by_id = {p["id"]: p for p in passages}
    if len(by_id) != len(passages):
        raise ValueError("Corpus passage IDs must be unique.")
    if len({p["id"] for p in seeds}) != len(seeds):
        raise ValueError("Seed passage IDs must be unique.")
    validated = []
    for seed in seeds:
        if seed["id"] not in by_id:
            raise ValueError(f"Unknown seed passage: {seed['id']}.")
        canonical = by_id[seed["id"]]
        for key in ("text", "title", "scope"):
            if seed.get(key) != canonical.get(key):
                raise ValueError(f"Seed content differs from corpus: {seed['id']} ({key}).")
        # Preserve retrieval scores, but take structural positions from the corpus.
        validated.append(dict(seed, **canonical))
    return validated


def build_contexts(seeds: list[dict], passages: list[dict],
                   token_count: Callable[[str], int],
                   make_prompt: Callable[[list[dict]], str],
                   extension_budget: int = 512, num_ctx: int = 8192,
                   max_new_tokens: int = 512) -> dict:
    """Pack the three conditions without access to gold labels or model outputs."""
    for name, value in (("extension_budget", extension_budget), ("num_ctx", num_ctx),
                        ("max_new_tokens", max_new_tokens)):
        if type(value) is not int or value < (0 if name == "extension_budget" else 1):
            raise ValueError(f"Invalid {name}.")
    seeds = _validated_seeds(seeds, passages)
    seed_ids = [p["id"] for p in seeds]
    seed_set = set(seed_ids)
    section_lookup = defaultdict(list)
    for p in passages:
        section_lookup[(p["document_id"], p["section_id"])].append(p)

    pool, relations = {}, defaultdict(list)
    for rank, seed in enumerate(seeds):
        for candidate in section_lookup[(seed["document_id"], seed["section_id"])]:
            distance = abs(seed["chunk_index"] - candidate["chunk_index"])
            if candidate["id"] in seed_set or not 1 <= distance <= 2:
                continue
            pool[candidate["id"]] = candidate
            relations[candidate["id"]].append({"seed_id": seed["id"], "seed_rank": rank + 1,
                                                "distance": distance})

    def ordinary_key(p):
        best = min((r["distance"], r["seed_rank"]) for r in relations[p["id"]])
        return best + _source_key(p)

    ordinary = sorted(pool.values(), key=ordinary_key)
    prioritized = sorted(ordinary, key=lambda p: (not has_condition_cue(p),) + ordinary_key(p))
    candidate_ids = [p["id"] for p in ordinary]
    common_flags = []
    if not pool:
        common_flags.append("no_neighbor")
    if not any(has_condition_cue(p) for p in ordinary):
        common_flags.append("no_condition_cue")
    seed_prompt = make_prompt(seeds)
    seed_tokens = token_count(seed_prompt)
    base_status = "no_evidence" if not seeds else (
        "input_budget_exceeded" if seed_tokens + max_new_tokens > num_ctx else "ready")

    def record(label, extras, skipped):
        extras = sorted(extras, key=_source_key)
        combined = seeds + extras
        prompt = make_prompt(combined)
        rendered_extension = render_passages(extras, start_index=len(seeds) + 1)
        flags = list(common_flags)
        if skipped:
            flags.append("budget_skipped")
        return {
            "group": label, "status": base_status, "rule_version": RULE_VERSION,
            "passages": combined, "seed_ids": list(seed_ids),
            "extra_ids": [p["id"] for p in extras], "candidate_ids": list(candidate_ids),
            "candidate_relations": dict(relations),
            "candidate_condition_cues": {p["id"]: has_condition_cue(p) for p in ordinary},
            "prompt": prompt, "input_tokens": token_count(prompt),
            "extension_tokens": token_count(rendered_extension) if extras else 0,
            "extension_text": rendered_extension,
            "flags": flags, "budget_skipped": skipped,
            "extension_budget": extension_budget, "num_ctx": num_ctx,
            "max_new_tokens": max_new_tokens,
        }

    output = {"A": record("A", [], [])}
    for label, ordered in (("B", prioritized), ("C", ordinary)):
        chosen, skipped = [], []
        if base_status == "ready":
            for candidate in ordered:
                trial = sorted(chosen + [candidate], key=_source_key)
                extension = render_passages(trial, start_index=len(seeds) + 1)
                extension_tokens = token_count(extension)
                full_tokens = token_count(make_prompt(seeds + trial))
                reasons = []
                if extension_tokens > extension_budget:
                    reasons.append("extension_budget")
                if full_tokens + max_new_tokens > num_ctx:
                    reasons.append("context_window")
                if reasons:
                    skipped.append({"id": candidate["id"], "reasons": reasons,
                                    "trial_extension_tokens": extension_tokens,
                                    "trial_input_tokens": full_tokens})
                else:
                    chosen.append(candidate)
        output[label] = record(label, chosen, skipped)

    identical = output["B"]["prompt"] == output["C"]["prompt"]
    for item in output.values():
        item["bc_identical"] = identical
        if identical:
            item["flags"].append("identical_context")
    return output
