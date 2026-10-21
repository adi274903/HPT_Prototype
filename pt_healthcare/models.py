"""Model loading.

``load_models`` is carried over from notebook cell 3: it builds the MedGemma
``image-text-to-text`` pipeline (bfloat16 on CUDA) and the MedTE sentence
embedder. Imports are lazy so the rest of the package (pricing, parsing,
pipeline validation) can be imported on a machine without torch installed.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from . import config


def load_models(
    device: str = "cuda",
    dtype: Optional[Any] = None,
    backend: Optional[str] = None,
    gguf: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load and return ``(medgemma_pipe, sentence_transformer)``.

    Four generators, all answering the same call shape:

    ``transformers``  in-process bfloat16 weights (the original path, default)
    ``llamacpp``      in-process GGUF through ``llama_cpp.Llama`` — no server
    ``vllm``          in-process through ``vllm.LLM`` — no server
    ``remote``        an OpenAI-compatible server someone else runs

    Only the generator changes. The embedder is always loaded here: retrieval has
    no backend behind it.

    Parameters
    ----------
    device : str
        Device for the transformers pipeline, e.g. ``"cuda"`` or ``"cpu"``.
    dtype : torch.dtype | None
        Torch dtype for the generator. Defaults to ``torch.bfloat16``.
    backend : str | None
        Overrides ``PT_LLM_BACKEND``.
    gguf : str | None
        Overrides ``PT_LLM_GGUF`` for the ``llamacpp`` backend.
    """
    from sentence_transformers import SentenceTransformer

    choice = (backend or config.llm_backend()).strip().lower()

    if choice == "llamacpp":
        from .backends import LlamaCppPipe

        path = gguf or config.llm_gguf_path()
        if not path:
            raise ValueError("PT_LLM_BACKEND=llamacpp needs PT_LLM_GGUF=/path/to/model.gguf")
        pipe: Any = LlamaCppPipe(
            path,
            n_ctx=config.llm_max_model_len(),
            n_gpu_layers=config.llm_n_gpu_layers(),
        )
    elif choice == "vllm":
        from .backends import VLLMPipe

        pipe = VLLMPipe(
            model=config.medgemma_model_id(),
            max_model_len=config.llm_max_model_len(),
        )
    elif choice == "remote":
        from .api_pipe import OpenAICompatPipe

        pipe = OpenAICompatPipe(config.llm_base_url(), model=config.llm_model_name())
    elif choice == "transformers":
        import torch
        from transformers import pipeline

        if dtype is None:
            dtype = torch.bfloat16

        pipe = pipeline(
            "image-text-to-text",
            model=config.medgemma_model_id(),
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(
            f"unknown PT_LLM_BACKEND={choice!r}: use transformers, llamacpp, vllm or remote"
        )

    embed_model = SentenceTransformer(config.embedding_model_id())

    return pipe, embed_model
