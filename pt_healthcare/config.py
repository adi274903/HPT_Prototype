"""Central configuration for the HPT prototype.

Values come from the environment, and are read **at call time** through the
accessors at the bottom of this module. That matters in Colab: a notebook cell
usually runs somewhere other than this repo, so ``load_dotenv()`` there happens
*after* this module was imported. Reading at import time would freeze the
defaults and silently ignore the ``.env``; reading at call time does not.

The module-level constants (``TOP_K``, ``MRF_CSV_PATH``, …) still exist for
convenience, but they are a snapshot — take them at import, or call
``refresh()`` after loading a ``.env`` to bring them up to date.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------
#: Repository root (the directory holding ``pt_healthcare/``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env_file_candidates() -> List[str]:
    """Paths searched for a ``.env``, in order.

    ``PT_ENV_FILE`` (explicit override), then ``./.env`` — what a notebook's
    ``load_dotenv()`` picks up — then Colab's ``/content/.env``, then this repo's
    ``.env``. Duplicates are dropped.
    """
    candidates = [
        os.getenv("PT_ENV_FILE"),
        ".env",
        "/content/.env",
        os.path.join(_REPO_ROOT, ".env"),
    ]

    out: List[str] = []
    for path in candidates:
        if path and path not in out:
            out.append(path)
    return out


def load_dotenv_files(
    paths: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> List[str]:
    """Load every readable ``.env``; return the paths that were loaded.

    Uses ``override=False``, so a real export, a Colab secret, or a value set
    with ``os.environ[...]`` always beats the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv absent — rely on the real environment
        return []

    loaded: List[str] = []

    for path in (env_file_candidates() if paths is None else paths):
        try:
            if os.path.isfile(path):
                load_dotenv(path, override=False)
                loaded.append(path)
        except Exception:
            continue

    if verbose and loaded:
        print("Loaded .env: " + ", ".join(loaded))

    return loaded


# Load whatever is on disk now, so the constants below are right in the common
# case (repo checked out with its own .env). Anything loaded later is picked up
# by the accessors, or explicitly via refresh().
load_dotenv_files()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"
DEFAULT_EMBEDDING_MODEL = "MohammadKhodadad/MedTE-cl15-step-8000"
DEFAULT_TOP_K = 10
DEFAULT_CPT_COLLECTION = "cpt_medte"
DEFAULT_HCPCS_COLLECTION = "hcpcs_medte"
DEFAULT_QDRANT_PATH = "New_PT_DB"
DEFAULT_MRF_CSV = "everyUPMCmrf_clean.csv"
DEFAULT_MAX_CANDIDATES = 8
DEFAULT_CANDIDATE_TEXT_LIMIT = 120
DEFAULT_MAX_NEW_TOKENS = 4096

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


# ---------------------------------------------------------------------------
# Live accessors — read the environment on every call
# ---------------------------------------------------------------------------
def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None or not str(raw).strip():
        return default

    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def medgemma_model_id() -> str:
    return _get("PT_MEDGEMMA_MODEL", DEFAULT_MEDGEMMA_MODEL)


def embedding_model_id() -> str:
    return _get("PT_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)


def top_k() -> int:
    return _get_int("PT_TOP_K", DEFAULT_TOP_K)


def cpt_collection() -> str:
    return _get("PT_CPT_COLLECTION", DEFAULT_CPT_COLLECTION)


def hcpcs_collection() -> str:
    return _get("PT_HCPCS_COLLECTION", DEFAULT_HCPCS_COLLECTION)


def qdrant_dir() -> str:
    return _get("PT_QDRANT_PATH", DEFAULT_QDRANT_PATH)


def mrf_csv_path() -> str:
    return _get("PT_MRF_CSV", DEFAULT_MRF_CSV)


def max_candidates_per_family() -> int:
    return _get_int("PT_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES)


def candidate_text_limit() -> int:
    return _get_int("PT_CANDIDATE_TEXT_LIMIT", DEFAULT_CANDIDATE_TEXT_LIMIT)


def max_new_tokens() -> int:
    return _get_int("PT_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)


def hf_token_from_env() -> Optional[str]:
    """First Hugging Face token found in the environment, or ``None``."""
    for name in HF_TOKEN_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Module-level constants (a snapshot; see refresh())
# ---------------------------------------------------------------------------
MEDGEMMA_MODEL_ID: str = medgemma_model_id()
EMBEDDING_MODEL_ID: str = embedding_model_id()
TOP_K: int = top_k()
CPT_COLLECTION: str = cpt_collection()
HCPCS_COLLECTION: str = hcpcs_collection()
QDRANT_PATH: str = qdrant_dir()
MRF_CSV_PATH: str = mrf_csv_path()
MAX_CANDIDATES_PER_FAMILY: int = max_candidates_per_family()
CANDIDATE_TEXT_LIMIT: int = candidate_text_limit()
MAX_NEW_TOKENS: int = max_new_tokens()


def refresh(
    paths: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> None:
    """Re-read the environment (and optionally load a ``.env``) into the constants.

    Call this after ``load_dotenv()`` if you rely on the module constants rather
    than the accessors::

        from dotenv import load_dotenv
        load_dotenv()
        from pt_healthcare import config
        config.refresh()
    """
    if paths is not None:
        load_dotenv_files(paths, verbose=verbose)

    global MEDGEMMA_MODEL_ID, EMBEDDING_MODEL_ID, TOP_K, CPT_COLLECTION
    global HCPCS_COLLECTION, QDRANT_PATH, MRF_CSV_PATH
    global MAX_CANDIDATES_PER_FAMILY, CANDIDATE_TEXT_LIMIT, MAX_NEW_TOKENS

    MEDGEMMA_MODEL_ID = medgemma_model_id()
    EMBEDDING_MODEL_ID = embedding_model_id()
    TOP_K = top_k()
    CPT_COLLECTION = cpt_collection()
    HCPCS_COLLECTION = hcpcs_collection()
    QDRANT_PATH = qdrant_dir()
    MRF_CSV_PATH = mrf_csv_path()
    MAX_CANDIDATES_PER_FAMILY = max_candidates_per_family()
    CANDIDATE_TEXT_LIMIT = candidate_text_limit()
    MAX_NEW_TOKENS = max_new_tokens()
