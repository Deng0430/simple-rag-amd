"""Interchangeable local retrievers. The dense encoder uses CPU ONNX, not torch."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
import time
from uuid import uuid4

import numpy as np

from rag import Retriever

LABELS = {"tfidf": "TF-IDF · 字面匹配", "bm25": "BM25 · 关键词检索", "dense": "E5 · 多语言稠密检索"}
DEFAULT_EMBEDDING = "intfloat/multilingual-e5-small"
CACHE_ROOT = Path(__file__).parent / ".cache" / "retrieval"
_REGISTER_LOCK = Lock()


def document_text(passage):
    return "\n".join([passage["title"], passage.get("keywords", ""), passage["text"]])


def ranked(passages, scores, k, min_score):
    if k < 1:
        raise ValueError("Top-k 必须大于零。")
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return [dict(passages[i], score=float(scores[i])) for i in order
            if np.isfinite(scores[i]) and scores[i] > min_score]


def tokenize(text):
    import jieba

    jieba.setLogLevel(40)
    tokens = []
    for part in re.findall(r"[a-zA-Z_][a-zA-Z_0-9.-]*|[0-9]+|[\u4e00-\u9fff]+", text.casefold()):
        if re.search(r"[\u4e00-\u9fff]", part):
            tokens.extend(word for word in jieba.lcut(part) if word.strip())
        else:
            tokens.append(part)
            # Retain the full API/error token and also its components.
            tokens.extend(word for word in re.split(r"[._-]+", part) if word and word != part)
    return tokens


class BM25Retriever:
    name = "bm25"
    score_kind = "BM25 分数"
    model_id = None

    def __init__(self, passages, *, min_score=0.0):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("BM25 需要 rank-bm25 和 jieba，请安装 requirements.txt。") from exc
        self.passages, self.min_score = passages, min_score
        self.index = BM25Okapi([tokenize(document_text(p)) for p in passages], k1=1.5, b=0.75)

    def search(self, question, k):
        if not question.strip():
            raise ValueError("请输入问题。")
        return ranked(self.passages, self.index.get_scores(tokenize(question)), k, self.min_score)


class DenseRetriever:
    name = "dense"
    score_kind = "E5 余弦相似度"

    def __init__(self, passages, *, min_score=0.0, model_id=DEFAULT_EMBEDDING, cache_dir=CACHE_ROOT):
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError as exc:
            raise RuntimeError("稠密检索需要 fastembed，请安装 requirements.txt。") from exc
        self.passages, self.min_score, self.model_id = passages, min_score, model_id
        with _REGISTER_LOCK:
            if model_id == DEFAULT_EMBEDDING and not any(m["model"] == model_id for m in TextEmbedding.list_supported_models()):
                TextEmbedding.add_custom_model(
                    model=model_id, pooling=PoolingType.MEAN, normalization=True, dim=384,
                    sources=ModelSource(hf="Xenova/multilingual-e5-small"),
                    model_file="onnx/model_quantized.onnx", license="mit", size_in_gb=0.12,
                    description="Multilingual E5 small, Xenova ONNX int8 export; query/passage prefixes required.",
                )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from embedding_model import ensure_embedding_model
        model_path, self.artifact = ensure_embedding_model(self.cache_dir / "models")
        # E5 requires these English prefixes even for Chinese text.
        texts = ["passage: " + document_text(p) for p in passages]
        fingerprint = hashlib.sha256(json.dumps(
            ["e5-prefix-v1", model_id, self.artifact["revision"], [(p["id"], t) for p, t in zip(passages, texts)]],
            ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest()
        self.cache_path = self.cache_dir / f"{fingerprint}.npz"
        self.encoder = TextEmbedding(model_name=model_id, cache_dir=str(self.cache_dir / "models"),
                                     threads=min(4, os.cpu_count() or 1),
                                     providers=["CPUExecutionProvider"], specific_model_path=str(model_path))
        self.cache_hit = False
        if self.cache_path.exists():
            try:
                with np.load(self.cache_path, allow_pickle=False) as stored:
                    matrix = stored["vectors"]
                if matrix.shape == (len(passages), 384) and np.isfinite(matrix).all():
                    self.matrix = matrix
                    self.cache_hit = True
            except (OSError, ValueError, KeyError):
                pass
        if not self.cache_hit:
            self.matrix = np.asarray(list(self.encoder.embed(texts, batch_size=16)), dtype=np.float32)
            self.matrix /= np.maximum(np.linalg.norm(self.matrix, axis=1, keepdims=True), 1e-12)
            temporary = self.cache_path.with_suffix(f".{uuid4().hex}.tmp.npz")
            np.savez_compressed(temporary, vectors=self.matrix)
            temporary.replace(self.cache_path)

    def search(self, question, k):
        if not question.strip():
            raise ValueError("请输入问题。")
        vector = np.asarray(next(iter(self.encoder.embed(["query: " + question]))), dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return ranked(self.passages, self.matrix @ vector, k, self.min_score)


def create_retriever(passages, method="tfidf", *, min_score=None, analyzer="char"):
    started = time.perf_counter()
    if method == "tfidf":
        result = Retriever(passages, analyzer=analyzer, min_score=0.04 if min_score is None else min_score)
    elif method == "bm25":
        result = BM25Retriever(passages, min_score=0.0 if min_score is None else min_score)
    elif method == "dense":
        result = DenseRetriever(passages, min_score=0.0 if min_score is None else min_score)
    else:
        raise ValueError(f"未知检索器：{method}")
    result.index_seconds = time.perf_counter() - started
    return result
