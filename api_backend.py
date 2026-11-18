"""External generation through an OpenAI-compatible Chat Completions endpoint."""

import http.client
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from answer_protocol import display_answer, structured_messages


PUBLIC_API_HOSTS = frozenset({
    "api.openai.com",
    "api.deepseek.com",
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "api.moonshot.cn",
    "api.siliconflow.cn",
    "api.siliconflow.com",
    "openrouter.ai",
})
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class ApiError(RuntimeError):
    """An API failure whose message can be shown to the current user."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A provider redirect must never forward the Bearer credential.
        return None


_opener = build_opener(NoRedirect())


def api_endpoint(base_url: str) -> str:
    """Allow public providers and hosts explicitly trusted by the server operator."""
    base_url = base_url.strip()
    invalid = "请填写服务商文档中的 HTTPS API 地址，例如 https://api.openai.com/v1。"
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError:
        raise ValueError(invalid) from None
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None
            or parts.query or parts.fragment or port not in (None, 443)
            or "\\" in base_url or any(ord(ch) <= 32 or ord(ch) == 127 for ch in base_url)):
        raise ValueError(invalid)
    extra_hosts = {
        host.strip().lower() for host in os.getenv("RAG_API_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }
    if parts.hostname not in PUBLIC_API_HOSTS | extra_hosts:
        raise ValueError("该 API 域名尚未启用。自定义服务可由部署者通过 RAG_API_ALLOWED_HOSTS 添加。")
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit(("https", parts.netloc, path, "", ""))


def _http_error(status: int) -> str:
    if 300 <= status < 400:
        return "API 地址返回了重定向，请填写服务商提供的最终 HTTPS 地址。"
    hints = {
        400: "请检查模型名称、上下文长度和 Chat Completions 兼容性。",
        401: "请核对 API 密钥。",
        403: "请核对密钥的服务权限和模型权限。",
        404: "请核对 API 地址和模型名称。",
        429: "请检查额度或速率限制，稍后重试。",
    }
    return f"外部 API 请求失败（HTTP {status}）。" + hints.get(status, "请稍后重试或检查服务商状态。")


def _token_count(usage, key):
    count = usage.get(key) if isinstance(usage, dict) else None
    return count if type(count) is int and count >= 0 else None


class ApiGenerator:
    backend = "api"

    def __init__(self, model_id: str, base_url: str, api_key: str):
        self.endpoint = api_endpoint(base_url)
        self.model_id = model_id.strip()
        self._api_key = api_key.strip()
        if not self.model_id:
            raise ValueError("请填写服务商控制台中的模型 ID。")
        if not self._api_key or not all("!" <= ch <= "~" for ch in self._api_key):
            raise ValueError("请填写有效的 API 密钥。")
        self.host = urlsplit(self.endpoint).hostname

    def metadata(self) -> dict:
        return {"backend": self.backend, "model": self.model_id, "api_host": self.host}

    def complete(self, messages: list[dict], max_new_tokens: int) -> dict:
        limit_field = "max_completion_tokens" if self.host == "api.openai.com" else "max_tokens"
        body = {"model": self.model_id, "messages": messages, "stream": False,
                limit_field: max_new_tokens}
        request = Request(
            self.endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key,
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with _opener.open(request, timeout=90) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            exc.close()
            raise ApiError(_http_error(exc.code)) from None
        except (URLError, OSError, http.client.HTTPException):
            raise ApiError("无法连接外部 API 或请求超时，请检查网络和服务地址。") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ApiError("API 返回内容过大，请检查模型与服务地址。")
        try:
            result = json.loads(raw)
        except ValueError:
            raise ApiError("API 未返回有效的 JSON，请检查服务地址。") from None
        if not isinstance(result, dict) or result.get("error"):
            raise ApiError("API 返回了异常响应，请检查服务商控制台。")
        choices = result.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise ApiError("API 未返回 Chat Completions 消息，请检查接口兼容性。")
        finish = choice.get("finish_reason")
        if message.get("refusal") or finish == "content_filter":
            raise ApiError("外部模型拒绝了本次请求，可切换检索预览查阅资料。")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or finish in ("tool_calls", "function_call"):
            raise ApiError("API 未返回可用的文本答案，请选择支持文本问答的模型。")
        usage = result.get("usage")
        return dict(
            self.metadata(),
            raw_answer=content.strip().replace(self._api_key, "[API 密钥已隐藏]"),
            input_tokens=_token_count(usage, "prompt_tokens"),
            output_tokens=_token_count(usage, "completion_tokens"),
            generation_seconds=time.perf_counter() - started,
            done_reason=finish if finish in ("stop", "length") else None,
        )

    def answer(self, question: str, passages: list[dict], *, profile: str = "general",
               max_new_tokens: int = 128) -> dict:
        generated = self.complete(structured_messages(question, passages), max_new_tokens)
        result = dict(generated, **display_answer(generated["raw_answer"], passages),
                      output_truncated=generated["done_reason"] == "length")
        if result["output_truncated"]:
            result["notice"] = (result["notice"] + " 回答达到输出上限，可能不完整。").strip()
        return result
