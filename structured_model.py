"""Pinned local Qwen tokenizer, ChatML and structured Ollama inference."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

from answer_protocol import SYSTEM, structured_messages
from rag import DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_URL, _ollama_request, ollama_status

TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"
TOKENIZER_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_DIGEST = "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b"
CACHE = Path(__file__).parent / ".cache" / "generation" / "qwen25-tokenizer"
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "short_answer": {"type": "string"},
        "explanation": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
    },
    "required": ["short_answer", "explanation", "citations", "abstain"],
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QwenTokenizer:
    def __init__(self, cache=CACHE):
        self.folder = Path(cache)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            target = self.folder / name
            if not target.exists():
                url = f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    body = response.read()
                json.loads(body)
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(body)
                temporary.replace(target)
        self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))
        self.manifest = {
            "repository": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
            "files": {p.name: sha256_file(p) for p in self.folder.glob("*.json")
                      if p.name != "manifest.json"},
            "prompt_mode": "explicit_chatml_api_generate_raw",
        }
        manifest_path = self.folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
                raise ValueError("Tokenizer manifest changed; verify the local tokenizer cache.")
        manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def make_prompt(question, passages):
    user = structured_messages(question, passages)[1]["content"]
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


class StructuredLocalModel:
    def __init__(self, tokenizer, base_url=DEFAULT_OLLAMA_URL, num_ctx=8192):
        self.base_url, self.num_ctx, self.tokenizer = base_url, num_ctx, tokenizer
        status = ollama_status(base_url)
        installed = next((m for m in status["models"] if m["name"] == DEFAULT_LOCAL_MODEL), None)
        if not installed or installed["digest"] != MODEL_DIGEST:
            raise RuntimeError("The frozen qwen2.5:3b model is not available or its digest changed.")
        shown = _ollama_request(base_url, "/api/show", {"model": DEFAULT_LOCAL_MODEL})
        self.metadata = {
            "backend": "ollama", "model": DEFAULT_LOCAL_MODEL, "model_digest": MODEL_DIGEST,
            "details": shown.get("details"),
            "ollama_version": _ollama_request(base_url, "/api/version")["version"],
            "num_ctx": num_ctx, "temperature": 0, "seed": 42,
            "prompt_mode": "api/generate raw=True, explicit ChatML, fixed JSON schema",
            "tokenizer": tokenizer.manifest,
            "ollama_default_template_sha256": hashlib.sha256(shown.get("template", "").encode()).hexdigest(),
        }

    def generate(self, prompt, max_new_tokens=512):
        expected = self.tokenizer.count(prompt)
        if expected + max_new_tokens > self.num_ctx:
            raise ValueError("input_budget_exceeded")
        start = time.perf_counter()
        response = _ollama_request(self.base_url, "/api/generate", {
            "model": DEFAULT_LOCAL_MODEL, "prompt": prompt, "raw": True,
            "format": SCHEMA, "stream": False, "keep_alive": "20m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": self.num_ctx,
                        "num_predict": max_new_tokens},
        }, timeout=180)
        return {"raw_response": response, "request_seconds": time.perf_counter() - start,
                "expected_input_tokens": expected,
                "token_count_match": response.get("prompt_eval_count") == expected}

    def validate_tokenizer(self):
        checks = []
        for question in ("仅回复OK。", "说明命令 python3 -m venv .venv；不要执行。", "路径 /home/jovyan 和条件 if CUDA is available。"):
            prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                      f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
            record = self.generate(prompt, max_new_tokens=2)
            checks.append(record)
            if not record["token_count_match"]:
                raise RuntimeError(f"Tokenizer mismatch: expected {record['expected_input_tokens']}, "
                                   f"received {record['raw_response'].get('prompt_eval_count')}")
        return checks


class IsolatedLocalModel(StructuredLocalModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metadata['request_isolation']='unload qwen2.5:3b before each unique generation, same weights/prompt/schema/options'

    def generate(self,prompt,max_new_tokens=512):
        start=time.perf_counter()
        reset=_ollama_request(self.base_url,'/api/generate',
                             {'model':DEFAULT_LOCAL_MODEL,'keep_alive':0,'stream':False},timeout=60)
        if reset.get('done') is not True:
            raise RuntimeError('Model unload did not complete; refusing a potentially contaminated request.')
        reset_seconds=time.perf_counter()-start
        result=super().generate(prompt,max_new_tokens)
        result['generation_request_seconds']=result['request_seconds']
        result['reset_seconds']=reset_seconds
        result['reset_response']=reset
        result['request_seconds']=time.perf_counter()-start
        return result
