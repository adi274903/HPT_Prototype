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

    Parameters
    ----------
    device : str
        Device passed to the transformers pipeline, e.g. ``"cuda"`` or ``"cpu"``.
    dtype : torch.dtype | None
        Torch dtype for the generator. Defaults to ``torch.bfloat16``.
    """
    import torch
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    if dtype is None:
        dtype = torch.bfloat16

    pipe = pipeline(
        "image-text-to-text",
        model=config.MEDGEMMA_MODEL_ID,
        dtype=dtype,
        device=device,
    )

    embed_model = SentenceTransformer(config.EMBEDDING_MODEL_ID)

    return pipe, embed_model
