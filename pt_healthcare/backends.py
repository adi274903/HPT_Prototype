"""Two ways to run MedGemma inside the process, instead of through a server.

The pipeline only asks its model for one thing (``pt_healthcare/llm.py``):

    out = self.pipe(text=messages, max_new_tokens=N, do_sample=False)
    return out[0]["generated_text"][-1]["content"]

so anything that answers in that shape can be the generator. Both classes below
do it **in code** — the weights live in this Python process, no port is opened,
no ``llama-server`` or ``vllm serve`` process is started:

    llama.cpp   ``llama_cpp.Llama(...).create_chat_completion(...)``
    vLLM        ``vllm.LLM(...).chat(...)``

Colab, llama.cpp::

    !pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    from pt_healthcare.backends import LlamaCppPipe
    pipe, embed = load_models(backend="llamacpp", gguf="/content/medgemma-1.5-4b-it-Q4_K_M.gguf")

Colab, vLLM::

    !pip install -q vllm
    pipe, embed = load_models(backend="vllm")

Both imports are lazy, so this module imports cleanly with neither installed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

Messages = Union[str, Sequence[Mapping[str, Any]]]


def to_openai_messages(messages: Messages) -> List[Dict[str, str]]:
    """Flatten the engine's content parts into ``[{"role", "content"}]``.

    ``MedGemmaEngine`` passes ``[{"role": "user", "content": [{"type": "text",
    "text": "…"}]}]`` (the transformers chat-template shape). Both backends want
    plain strings, and an empty assistant seed is the server's job to add.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            parts: List[str] = []
            for part in content or []:
                kind = part.get("type") if isinstance(part, Mapping) else None
                if kind == "text":
                    parts.append(str(part.get("text", "")))
                elif kind in ("image_url", "image"):
                    raise ValueError(
                        "these backends are text-only; MedGemma would need the "
                        "transformers pipeline to look at an image"
                    )
            text = "".join(parts)
        if role == "assistant" and not text.strip():
            continue
        out.append({"role": role, "content": text})
    return out


def _shape(content: str) -> List[Dict[str, Any]]:
    """Wrap a completion in the transformers chat shape."""
    return [{"generated_text": [{"role": "assistant", "content": content}]}]


class LlamaCppPipe:
    """MedGemma through ``llama_cpp.Llama``, in this process.

    ``n_gpu_layers=-1`` offloads everything the card can hold; ``0`` runs on the
    CPU, which is the fallback when Colab gives you no GPU. ``use_mmap`` keeps
    reload cheap when the kernel restarts the cell.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        n_batch: int = 512,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        from llama_cpp import Llama  # lazy: only needed on this path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"no GGUF at {model_path}")

        self.model_path = model_path
        self.last_usage: Dict[str, int] = {}
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or (os.cpu_count() or 4),
            n_batch=n_batch,
            verbose=verbose,
            **kwargs,
        )

    def __call__(
        self,
        messages: Optional[Messages] = None,
        text: Optional[Messages] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        payload = messages if messages is not None else text
        out = self._llm.create_chat_completion(
            messages=to_openai_messages(payload or ""),
            max_tokens=max_new_tokens,
            temperature=0.7 if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
            **kwargs,
        )
        self.last_usage = out.get("usage") or {}
        return _shape(out["choices"][0]["message"]["content"] or "")

    def close(self) -> None:
        self._llm.close()


class VLLMPipe:
    """MedGemma through ``vllm.LLM``, in this process.

    Paged attention and CUDA graphs, no HTTP hop. ``max_model_len`` caps the KV
    cache: 8192 covers the categoriser prompt plus a full ``thought`` block.
    """

    def __init__(
        self,
        model: str = "google/medgemma-1.5-4b-it",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        from vllm import LLM  # lazy: only needed on this path

        self.model = model
        self.last_usage: Dict[str, int] = {}
        # ``llm`` is an instance someone else built — the app and the eval share
        # one so a 40 GB card holds one copy of the weights, not two.
        self._llm = llm if llm is not None else LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def __call__(
        self,
        messages: Optional[Messages] = None,
        text: Optional[Messages] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        from vllm import SamplingParams

        payload = messages if messages is not None else text
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7 if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
        )
        outputs = self._llm.chat(messages=to_openai_messages(payload or ""), sampling_params=params)
        completion = outputs[0]
        self.last_usage = {
            "prompt_tokens": len(completion.prompt_token_ids or []),
            "completion_tokens": len(completion.outputs[0].token_ids or []),
        }
        return _shape(completion.outputs[0].text or "")

    def close(self) -> None:
        """Free the cards. vLLM keeps its engine alive for the whole process."""
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:  # best effort — the kernel is usually going away anyway
            pass
        self._llm = None
