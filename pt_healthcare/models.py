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
) -> Tuple[Any, Any]:
    """Load and return ``(medgemma_pipeline, sentence_transformer)``.

    If ``PT_LLM_BASE_URL`` is set, the generator is an
    :class:`~pt_healthcare.api_pipe.OpenAICompatPipe` pointing at that server
    (llama.cpp, vLLM, TGI, Ollama, LM Studio) instead of an in-process
    transformers pipeline. The embedder is still loaded locally either way —
    retrieval has no server behind it.

    Parameters
    ----------
    device : str
        Device passed to the transformers pipeline, e.g. ``"cuda"`` or ``"cpu"``.
        Ignored when a backend URL is set.
    dtype : torch.dtype | None
        Torch dtype for the generator. Defaults to ``torch.bfloat16``.
    """
    from sentence_transformers import SentenceTransformer

    if config.llm_base_url():
        from .api_pipe import OpenAICompatPipe

        pipe: Any = OpenAICompatPipe(config.llm_base_url(), model=config.llm_model_name())
    else:
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

    embed_model = SentenceTransformer(config.embedding_model_id())

    return pipe, embed_model
