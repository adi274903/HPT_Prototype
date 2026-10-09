"""Central configuration for the HPT prototype.

Every value can be overridden with an environment variable so the same code
runs unchanged in Colab, on a GPU box, or in CI.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

# --- Models -----------------------------------------------------------------
# Multimodal generator used for entity extraction, code decision and the final
# patient-facing answer.
MEDGEMMA_MODEL_ID: str = os.getenv("PT_MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it")

# Sentence embedder used to build the query vector for the Qdrant collections.
EMBEDDING_MODEL_ID: str = os.getenv(
    "PT_EMBED_MODEL",
    "MohammadKhodadad/MedTE-cl15-step-8000",
)

# Hugging Face token env vars (gated models). The first one that is set wins.
# HF_TOKEN / HUGGINGFACE_HUB_TOKEN are the names huggingface_hub reads by itself,
# so accepting them avoids the surprise of a .env that "works" for the library
# but not for our explicit login() call.
HF_TOKEN_ENV: str = "HUGGINGFACE_API_KEY"
HF_TOKEN_ENV_VARS: Sequence[str] = (
    "HUGGINGFACE_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def hf_token_from_env() -> Optional[str]:
    """First token found in the environment, or ``None``."""
    for name in HF_TOKEN_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None

# --- Retrieval --------------------------------------------------------------
# Number of candidate codes pulled from each Qdrant collection per query.
TOP_K: int = int(os.getenv("PT_TOP_K", "10"))

CPT_COLLECTION: str = os.getenv("PT_CPT_COLLECTION", "cpt_medte")
HCPCS_COLLECTION: str = os.getenv("PT_HCPCS_COLLECTION", "hcpcs_medte")

# --- Data paths -------------------------------------------------------------
# Local Qdrant store directory (contains `cpt_medte` / `hcpcs_medte`).
QDRANT_PATH: str = os.getenv("PT_QDRANT_PATH", "New_PT_DB")

# Cleaned hospital MRF (machine-readable file) CSV.
MRF_CSV_PATH: str = os.getenv("PT_MRF_CSV", "everyUPMCmrf_clean.csv")

# --- Decision prompt limits -------------------------------------------------
MAX_CANDIDATES_PER_FAMILY: int = int(os.getenv("PT_MAX_CANDIDATES", "8"))
CANDIDATE_TEXT_LIMIT: int = int(os.getenv("PT_CANDIDATE_TEXT_LIMIT", "120"))

# Generation limits.
MAX_NEW_TOKENS: int = int(os.getenv("PT_MAX_NEW_TOKENS", "4096"))
