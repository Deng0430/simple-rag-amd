"""Web runtime for dense retrieval, rank fusion and budgeted evidence expansion."""
import hashlib
import time
from threading import Lock

from answer_protocol import render_passages, retrieval_query, parse_answer, response_status
from structured_model import QwenTokenizer, SYSTEM, make_prompt, IsolatedLocalModel
from hybrid_retrieval import build_contexts
from rag import DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_URL, Generator

STRATEGIES = {"X": "X · 融合检索＋证据补充（推荐）", "H": "H · 融合检索 Top-3",
              "D": "D · E5 Top-3", "custom": "自定义检索演示"}
RESULT_LABELS = {"plan_" + k: v for k, v in STRATEGIES.items() if k != "custom"}


class PlanRuntime:
    """Use fixed local prompt/schema/options; cloud models use their own tokenizer."""
    def __init__(self, backend=None, model_id=None, base_url=DEFAULT_OLLAMA_URL, num_ctx=8192):
        self.backend, self.model_id, self.num_ctx = backend, model_id, num_ctx
        self.lock = Lock()
        self.model = None
        if backend in (None, "ollama"):
            self.tokenizer = QwenTokenizer()
            self.count = self.tokenizer.count
            self.prompt = make_prompt
            if backend == "ollama":
                if model_id != DEFAULT_LOCAL_MODEL or num_ctx != 8192:
                    raise ValueError("此方案使用 qwen2.5:3b 和 8192 上下文；其他本地模型请使用自定义检索演示。")
                self.model = IsolatedLocalModel(self.tokenizer, base_url=base_url, num_ctx=num_ctx)
                self.model.validate_tokenizer()
        elif backend == "transformers":
            self.model = Generator(model_id)
            self.tokenizer = self.model.tokenizer
            self.count = lambda text: len(self.tokenizer.encode(text, add_special_tokens=False))
            self.prompt = self.cloud_prompt
        else:
            raise ValueError("不支持的生成后端。")

    def cloud_prompt(self, question, passages):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"资料：\n{render_passages(passages)}\n\n问题：{question}\n请按JSON格式作答。"}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def generate(self, context):
        # Unload/generate is one critical section across Streamlit sessions.
        with self.lock:
            if self.backend == "ollama":
                output = self.model.generate(context["prompt"], 512)
                if response_status(output) != "ok":
                    raise RuntimeError("模型响应未完成或输入 token 校验失败。")
                raw = output["raw_response"]
                return dict(raw_answer=raw["response"], input_tokens=raw["prompt_eval_count"],
                            output_tokens=raw["eval_count"], generation_seconds=output["request_seconds"],
                            done_reason=raw.get("done_reason"), load_seconds=raw.get("load_duration", 0)/1e9,
                            model_digest=self.model.metadata["model_digest"], model_quantization="Q4_K_M",
                            prompt_mode=self.model.metadata["prompt_mode"],
                            request_isolation=self.model.metadata["request_isolation"])
            if self.backend != "transformers":
                raise RuntimeError("检索预览未配置生成模型。")
            import torch
            started = time.perf_counter()
            inputs = self.tokenizer(context["prompt"], return_tensors="pt", add_special_tokens=False)
            n = inputs["input_ids"].shape[1]
            if n != context["input_tokens"] or n + 512 > self.num_ctx:
                raise RuntimeError("云端输入 token 预算校验失败。")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            pad_id = self.tokenizer.pad_token_id
            with torch.inference_mode():
                output = self.model.model.generate(**inputs, do_sample=False, max_new_tokens=512,
                    pad_token_id=pad_id if pad_id is not None else self.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            generated = output[0, n:]
            return dict(raw_answer=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                        input_tokens=n, output_tokens=len(generated),
                        generation_seconds=time.perf_counter()-started,
                        done_reason="length" if len(generated) == 512 else "stop",
                        prompt_mode="model chat template; same structured instructions; no constrained decoder",
                        cloud_quality_evaluated=False)


def display_answer(raw, passages):
    """Do not repair malformed JSON or turn a refusal into an actionable answer."""
    parsed = parse_answer(raw)
    if not parsed["parse_ok"]:
        return dict(answer=None, status="format_error", notice="模型回答格式不完整，请先查阅下方原文。", parse_ok=False)
    obj = parsed["parsed"]
    if obj["abstain"]:
        return dict(answer=obj["explanation"], status="abstained", notice="资料不足，模型未给出确定操作建议。", parse_ok=True)
    if not obj["short_answer"].strip():
        return dict(answer=None, status="empty_answer", notice="模型未给出直接答案，请核对原文。", parse_ok=True)
    valid = {f"P{i}" for i in range(1, len(passages)+1)}
    citations = obj["citations"]
    invalid = [c for c in citations if c not in valid]
    answer = obj["short_answer"] + "\n\n" + obj["explanation"]
    refs = list(dict.fromkeys(c for c in citations if c in valid))
    if refs:
        answer += "\n\n" + " ".join(f"[{c}]" for c in refs)
    return dict(answer=answer, status="citation_warning" if invalid or not refs else "matched",
                notice="部分引用编号无效或缺失，请核对原文。" if invalid or not refs else "",
                parse_ok=True, invalid_citations=invalid)


def run_plan_query(retriever, question, strategy, runtime, *, generate=False, knowledge_base="unspecified"):
    if strategy not in ("D", "H", "X"):
        raise ValueError("未知证据方案。")
    started = time.perf_counter()
    rankings = retriever.search_all(retrieval_query(question))
    context = build_contexts(rankings, retriever.passages, runtime.count,
                             lambda ps: runtime.prompt(question, ps), num_ctx=runtime.num_ctx)[strategy]
    ps = context["passages"]
    dense = retriever.dense
    result = dict(question=question.strip(), top_k=3, mode="model" if generate else "retrieval_preview",
        model=runtime.model_id if generate else None, backend=runtime.backend,
        retriever="plan_"+strategy, strategy=strategy,
        score_kind="E5 余弦相似度" if strategy=="D" else "RRF 融合分数", min_score=0.0,
        answer=None, input_tokens=None, output_tokens=None, generation_seconds=None,
        torch_peak_allocated_gib=None, passages=ps, retrieved_count=len(ps),
        retrieval_seconds=rankings["timings"]["total_seconds"], knowledge_base=knowledge_base,
        retriever_model=getattr(dense, "model_id", None),
        index_seconds=getattr(dense, "index_seconds", None), index_cache_hit=getattr(dense, "cache_hit", None),
        index_dimensions=getattr(dense, "matrix", []).shape[1] if hasattr(dense, "matrix") else None,
        indexed_passages=len(retriever.passages),
        status=context["status"], notice="", seed_count=len(context["seed_ids"]),
        seed_ids=";".join(context["seed_ids"]), extra_ids=";".join(context["extra_ids"]),
        extension_tokens=context["extension_tokens"], extension_budget=512,
        planned_input_tokens=context["input_tokens"],
        prompt_sha256=hashlib.sha256(context["prompt"].encode()).hexdigest(),
        rule_version=context["rule_version"],
        budget_tokenizer="Qwen2.5 fixed tokenizer" if runtime.backend != "transformers" else runtime.model_id)
    if context["status"] != "ready":
        result["notice"] = "没有可用资料。" if not ps else "资料超过上下文预算，请缩短问题。"
    elif generate:
        result.update(runtime.generate(context))
        result.update(display_answer(result["raw_answer"], ps))
        result["output_truncated"] = result.get("done_reason") == "length"
        if result["output_truncated"]:
            result["notice"] = (result["notice"] + " 回答达到输出上限，可能不完整。").strip()
    result["total_seconds"] = time.perf_counter()-started
    return result
