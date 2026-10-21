"""An OpenAI-compatible chat backend, shaped like the transformers pipeline.

``MedGemmaEngine`` only ever does this::

    out = self.pipe(text=messages, max_new_tokens=N, do_sample=False)
    return out[0]["generated_text"][-1]["content"]

so anything that returns that list-of-one-dict can stand in for the in-process
``image-text-to-text`` pipeline. A server speaking ``POST /v1/chat/completions``
can: vLLM, llama.cpp's ``llama-server``, TGI, Ollama, LM Studio, SGLang.

Why bother: the notebook path loads 8 GB of bfloat16 weights and pays python
overhead on every token. A 4-bit/8-bit GGUF on llama.cpp, or vLLM with paged
attention, is several times faster on the same GPU and fits a smaller one.

    export PT_LLM_BASE_URL=http://127.0.0.1:8080/v1
    python colab_run.py "How much is a diagnostic mammogram at UPMC Presbyterian?"

``load_models()`` then skips MedGemma entirely and returns this shim, so the
retrieval stand-ins (embeddings, Qdrant) still load normally.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import config


def _flatten(content: Any) -> str:
    """Turn transformers-style content parts into one string for the server.

    The engine builds ``[{"type": "text", "text": prompt}]`` (the modality the
    image-text pipeline wants). OpenAI-compatible servers want a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                if part.get("type") == "text" or "text" in part:
                    parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    raise ValueError(
                        "this backend is text-only; the image-text pipeline is still "
                        "the only path that can read a photo"
                    )
        return "".join(parts)
    return str(content)


def to_openai_messages(messages: Any) -> List[Dict[str, str]]:
    """Normalise the engine's messages (or a bare string) to the OpenAI shape."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: List[Dict[str, str]] = []
    for msg in messages or []:
        if not isinstance(msg, Mapping):
            out.append({"role": "user", "content": _flatten(msg)})
            continue
        role = str(msg.get("role") or "user")
        # The transformers chat template seeds the assistant turn; the server
        # does that itself, and a trailing empty assistant turn confuses vLLM.
        if role == "assistant" and not msg.get("content"):
            continue
        out.append({"role": role, "content": _flatten(msg.get("content"))})
    if not out:
        raise ValueError("no messages to send")
    return out


class OpenAICompatPipe:
    """Drop-in for ``transformers.pipeline``'s call shape, over HTTP."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        base = (base_url or config.llm_base_url()).rstrip("/")
        if not base:
            raise ValueError("no backend URL: set PT_LLM_BASE_URL")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self.base_url = base
        self.model = model or config.llm_model_name()
        self.timeout = timeout or config.llm_timeout()
        self.api_key = api_key or config.llm_api_key()
        self.last_usage: Dict[str, Any] = {}
        self.calls = 0

    # -- the pipeline contract ------------------------------------------
    def __call__(
        self,
        text: Any = None,
        *args: Any,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        return_full_text: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        messages = to_openai_messages(text if text is not None else args[0])
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_new_tokens or config.max_new_tokens()),
            "temperature": 0.7 if do_sample else 0.0,
        }
        if do_sample:
            body["top_p"] = 0.95

        answered = self._post("/chat/completions", body)
        content = answered["choices"][0]["message"]["content"]
        self.last_usage = answered.get("usage") or {}
        self.calls += 1

        # What transformers returns for a chat pipeline: the history, with the
        # assistant turn appended. The engine reads [-1]["content"].
        return [{"generated_text": messages + [{"role": "assistant", "content": content}]}]

    # -- transport -------------------------------------------------------
    def _post(self, path: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body).encode()
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # surface the server's own words
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{exc.code} from {self.base_url}{path}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach {self.base_url}: {exc.reason}. Is the server running "
                f"(llama-server / vllm) and is PT_LLM_BASE_URL right?"
            ) from None

    def health(self) -> Dict[str, Any]:
        """``GET /models`` — enough to prove the backend is up before a run."""
        req = urllib.request.Request(f"{self.base_url}/models")
        if self.api_key:
            req.add_header("authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def timed(self, fn, *a, **kw):  # pragma: no cover - convenience for notebooks
        t0 = time.time()
        out = fn(*a, **kw)
        return out, time.time() - t0
