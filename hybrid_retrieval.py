"""Answer-blind dense/BM25 fusion and global evidence expansion.

This v2 experiment does not change the frozen v1 neighbor experiment. No gold
labels, question IDs, source-specific routing, condition keyword rules or model
answers are read here. The only retrieval input is the supplied query string.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from condition_context import render_passages
from retrievers import BM25Retriever, DenseRetriever


RULE_VERSION = "dense-bm25-rrf-global-v2"
TOP_K = 3
CANDIDATE_K = 20
RRF_K = 60
RRF_WEIGHTS = {"dense": 1.0, "bm25": 1.0}
EXTENSION_BUDGET = 512
NUM_CTX = 8192
MAX_NEW_TOKENS = 512


def _corpus_lookup(passages):
    by_id = {p["id"]: p for p in passages}
    if len(by_id) != len(passages):
        raise ValueError("Corpus passage IDs must be unique.")
    return by_id, {p["id"]: i for i, p in enumerate(passages)}


def _canonical_ranked(ranked, by_id):
    """Reject altered text; preserve scores/metadata on canonical passages."""
    seen, result = set(), []
    for p in ranked:
        pid = p["id"]
        if pid in seen:
            raise ValueError("Ranked passage IDs must be unique.")
        if pid not in by_id:
            raise ValueError(f"Unknown ranked passage: {pid}.")
        canonical = by_id[pid]
        for field in ("title", "text", "scope"):
            if p.get(field) != canonical.get(field):
                raise ValueError(f"Ranked content differs from corpus: {pid} ({field}).")
        seen.add(pid)
        # Metadata may be added by retrieval, but source fields stay canonical.
        result.append(dict(p, **canonical))
    return result


def reciprocal_rank_fusion(dense, bm25, passages):
    """Fixed equal-weight RRF: sum(1 / (60 + one-based rank)).

    Only passages present in either list are candidates. Scores from different
    retrieval systems are not normalized or compared; exact RRF ties use the
    original corpus order, not a gold-derived priority or source preference.
    """
    by_id, order = _corpus_lookup(passages)
    ranked = {"dense": _canonical_ranked(dense, by_id),
              "bm25": _canonical_ranked(bm25, by_id)}
    info = {}
    for source, items in ranked.items():
        for rank, item in enumerate(items, 1):
            raw_score = item.get("score")
            if raw_score is not None and not math.isfinite(float(raw_score)):
                raise ValueError("Nonfinite retrieval score.")
            slot = info.setdefault(item["id"], {"dense_rank": None, "bm25_rank": None,
                "dense_score": None, "bm25_score": None,
                "rrf_contributions": {"dense": 0.0, "bm25": 0.0}})
            slot[source + "_rank"] = rank
            slot[source + "_score"] = raw_score
            slot["rrf_contributions"][source] = RRF_WEIGHTS[source] / (RRF_K + rank)
    fused = []
    for pid, metadata in info.items():
        score = sum(metadata["rrf_contributions"].values())
        fused.append(dict(by_id[pid], **metadata, score=score, rrf_score=score,
                          corpus_order=order[pid]))
    fused.sort(key=lambda p: (-p["rrf_score"], p["corpus_order"]))
    for rank, p in enumerate(fused, 1):
        p["fused_rank"] = rank
    return fused


class HybridRetriever:
    """Reuse cached E5 and the project's BM25 tokenizer, without new weights."""

    def __init__(self, passages, *, dense=None, bm25=None):
        _corpus_lookup(passages)
        self.passages = passages
        self.dense = dense if dense is not None else DenseRetriever(passages)
        self.bm25 = bm25 if bm25 is not None else BM25Retriever(passages)
        self.config = {
            "rule_version": RULE_VERSION,
            "dense_model": getattr(self.dense, "model_id", None),
            "dense_artifact": getattr(self.dense, "artifact", None),
            "bm25_parameters": {"k1": 1.5, "b": 0.75, "tokenizer": "retrievers.tokenize"},
            "branch_top_k": CANDIDATE_K, "seed_top_k": TOP_K,
            "rrf_k": RRF_K, "rrf_weights": dict(RRF_WEIGHTS),
            "tie_break": "original corpus order",
        }

    def search_all(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty retrieval query is required.")
        start = time.perf_counter()
        dense = self.dense.search(question, CANDIDATE_K)
        after_dense = time.perf_counter()
        bm25 = self.bm25.search(question, CANDIDATE_K)
        after_bm25 = time.perf_counter()
        if len(dense) > CANDIDATE_K or len(bm25) > CANDIDATE_K:
            raise ValueError("Retriever returned more than the frozen candidate limit.")
        fused = reciprocal_rank_fusion(dense, bm25, self.passages)
        end = time.perf_counter()
        return {
            "dense": dense, "bm25": bm25, "fused": fused,
            "timings": {"dense_seconds": after_dense - start,
                        "bm25_seconds": after_bm25 - after_dense,
                        "fusion_seconds": end - after_bm25,
                        "total_seconds": end - start},
            "config": dict(self.config),
        }


def build_contexts(search_result: dict, passages: list[dict],
                   token_count: Callable[[str], int],
                   make_prompt: Callable[[list[dict]], str],
                   extension_budget: int = EXTENSION_BUDGET,
                   num_ctx: int = NUM_CTX,
                   max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """D=dense top3, H=RRF top3, X=H plus whole global ranked passages.

    Extensions retain fused rank order, including when an oversized candidate is
    skipped. The rendered extension includes title/scope/citation labels. The
    complete prompt is separately counted with its output reserve. D/H inputs
    can differ in length despite equal top-k; all lengths are recorded.
    """
    for name, value in (("extension_budget", extension_budget), ("num_ctx", num_ctx),
                        ("max_new_tokens", max_new_tokens)):
        if type(value) is not int or value < (0 if name == "extension_budget" else 1):
            raise ValueError(f"Invalid {name}.")
    by_id, _ = _corpus_lookup(passages)
    dense = _canonical_ranked(search_result["dense"], by_id)
    bm25 = _canonical_ranked(search_result["bm25"], by_id)
    # Recompute from ranks to prevent an externally edited fused list from
    # becoming a hidden intervention. The saved list must match exactly.
    fused = reciprocal_rank_fusion(dense, bm25, passages)
    supplied = _canonical_ranked(search_result["fused"], by_id)
    if [p["id"] for p in supplied] != [p["id"] for p in fused]:
        raise ValueError("Supplied fused ranking differs from fixed RRF.")
    if len(dense) > CANDIDATE_K or len(bm25) > CANDIDATE_K:
        raise ValueError("Candidate list exceeds the frozen top-20 limit.")

    dense_seeds, hybrid_seeds = dense[:TOP_K], fused[:TOP_K]
    candidates = fused[TOP_K:]
    provenance = [{k: p[k] for k in ("id", "fused_rank", "dense_rank", "bm25_rank",
                   "dense_score", "bm25_score", "rrf_score", "rrf_contributions", "corpus_order")}
                  for p in fused]

    def count(text):
        n = token_count(text)
        if type(n) is not int or n < 0:
            raise ValueError("Tokenizer must return a nonnegative integer.")
        return n

    def base_status(seeds):
        if not seeds:
            return "no_evidence"
        return "input_budget_exceeded" if count(make_prompt(seeds)) + max_new_tokens > num_ctx else "ready"

    def record(label, seeds, extras, skipped, status):
        combined = seeds + extras
        prompt = make_prompt(combined)
        extension = render_passages(extras, start_index=len(seeds) + 1)
        flags = []
        if len(seeds) < TOP_K:
            flags.append("fewer_than_top_k")
        if not candidates:
            flags.append("no_global_candidates")
        if skipped:
            flags.append("budget_skipped")
        return {
            "group": label, "rule_version": RULE_VERSION, "status": status,
            "passages": combined, "seed_ids": [p["id"] for p in seeds],
            "extra_ids": [p["id"] for p in extras],
            "candidate_ids": [p["id"] for p in candidates],
            "dense_top20_ids": [p["id"] for p in dense],
            "bm25_top20_ids": [p["id"] for p in bm25],
            "fusion_ranking": provenance,
            "prompt": prompt, "input_tokens": count(prompt),
            "extension_text": extension, "extension_tokens": count(extension) if extras else 0,
            "extension_budget": extension_budget, "num_ctx": num_ctx,
            "max_new_tokens": max_new_tokens, "flags": flags,
            "budget_skipped": skipped,
        }

    d_status, h_status = base_status(dense_seeds), base_status(hybrid_seeds)
    chosen, skipped = [], []
    if h_status == "ready":
        for p in candidates:
            trial = chosen + [p]
            ext_tokens = count(render_passages(trial, start_index=len(hybrid_seeds) + 1))
            full_tokens = count(make_prompt(hybrid_seeds + trial))
            reasons = []
            if ext_tokens > extension_budget:
                reasons.append("extension_budget")
            if full_tokens + max_new_tokens > num_ctx:
                reasons.append("context_window")
            if reasons:
                skipped.append({"id": p["id"], "reasons": reasons,
                                "trial_extension_tokens": ext_tokens,
                                "trial_input_tokens": full_tokens})
            else:
                chosen.append(p)
    result = {
        "D": record("D", dense_seeds, [], [], d_status),
        "H": record("H", hybrid_seeds, [], [], h_status),
        "X": record("X", hybrid_seeds, chosen, skipped, h_status),
    }
    identities = {"dh_identical": result["D"]["prompt"] == result["H"]["prompt"],
                  "hx_identical": result["H"]["prompt"] == result["X"]["prompt"],
                  "dx_identical": result["D"]["prompt"] == result["X"]["prompt"]}
    for item in result.values():
        item.update(identities)
    return result
