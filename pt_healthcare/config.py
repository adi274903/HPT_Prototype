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
from typing import Dict, List, Optional, Sequence, Tuple

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

#: Seconds to wait for one completion from an OpenAI-compatible server. A 4B
#: thinking model that spends 400 tokens on <unused94>thought is slow unbatched.
DEFAULT_LLM_TIMEOUT = 600

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


#: Token budget per step. MedGemma is a thinking model: it spends the budget on
#: its ``<unused94>thought`` block *before* any JSON or prose exists, and that block
#: restates the input — the decision step walks the candidate list one code at a
#: time. A fixed per-step constant is therefore the wrong shape: 512 cut
#: "colonoscopy" mid-candidate, and 1536 cut it again on a longer candidate list.
#:
#: The budget is sized from the prompt instead:
#:
#:     budget = prompt_tokens * thought_factor + output_slack
#:
#: floored at the step's minimum (so a tiny prompt still gets room) and capped
#: by ``max_new_tokens_cap()`` so a pathological prompt cannot run away.
#: Tokens are estimated at 4 characters each, so this works before any
#: tokenizer is loaded. Each entry is ``(thought_factor, output_slack, floor)``.
#:
#: ``PT_MAX_NEW_TOKENS_<STEP>`` pins a step to a constant (the escape hatch if
#: the formula misbehaves on your model); ``PT_MAX_NEW_TOKENS`` raises the cap.
DEFAULT_STEP_TOKENS: Dict[str, Tuple[float, int, int]] = {
    "intent": (2.0, 256, 512),
    "categorize": (2.0, 384, 768),
    "decide": (3.0, 768, 1536),
    "answer": (1.5, 768, 1024),
}

#: Nothing generates past this many tokens unless ``PT_MAX_NEW_TOKENS`` says so.
DEFAULT_TOKEN_CAP = 4096


def max_new_tokens_cap() -> int:
    """The ceiling on any single generation."""
    return _get_int("PT_MAX_NEW_TOKENS", DEFAULT_TOKEN_CAP)


def max_new_tokens(step: str = "", prompt: str = "") -> int:
    """The generation budget — dynamic in the prompt, pinned by env if set.

    ``PT_MAX_NEW_TOKENS_<STEP>`` pins the step outright. Otherwise a prompt
    given here sizes the budget from its own length (see
    ``DEFAULT_STEP_TOKENS``); with no prompt, the step's floor is returned, so
    a call that cannot estimate keeps a sane bound. An unnamed step returns the
    cap, which keeps an existing ``PT_MAX_NEW_TOKENS`` ``.env`` behaving as
    before.
    """
    name = (step or "").strip().lower()

    if name:
        pinned = _get_int(f"PT_MAX_NEW_TOKENS_{name.upper()}", 0)
        if pinned:
            return pinned

    if name in DEFAULT_STEP_TOKENS:
        thought_factor, output_slack, floor = DEFAULT_STEP_TOKENS[name]

        if prompt:
            prompt_tokens = max(1, len(prompt)) // 4
            wanted = int(prompt_tokens * thought_factor) + output_slack
            return max(floor, min(wanted, max_new_tokens_cap()))

        return floor

    return max_new_tokens_cap()


def llm_backend() -> str:
    """Which generator ``load_models`` builds.

    ``transformers`` (default, in-process bfloat16), ``llamacpp`` (in-process GGUF
    through llama_cpp), ``vllm`` (in-process through vllm.LLM), or ``remote`` (an
    OpenAI-compatible server someone else is running).
    """
    name = _get("PT_LLM_BACKEND", "transformers").strip().lower()
    return name or "transformers"


def llm_gguf_path() -> str:
    """GGUF for the llamacpp backend, e.g. medgemma-1.5-4b-it-Q4_K_M.gguf."""
    return _get("PT_LLM_GGUF", "")


def llm_n_gpu_layers() -> int:
    """-1 offloads the whole model; 0 is CPU-only."""
    return _get_int("PT_LLM_N_GPU_LAYERS", -1)


def llm_max_model_len() -> int:
    """KV-cache ceiling for vLLM / n_ctx for llama.cpp."""
    return _get_int("PT_LLM_MAX_MODEL_LEN", 8192)


def llm_base_url() -> str:
    """OpenAI-compatible backend, e.g. ``http://127.0.0.1:8080/v1``.

    Empty means "load MedGemma in this process", the notebook path.
    """
    return _get("PT_LLM_BASE_URL", "").rstrip("/")


def llm_model_name() -> str:
    """Name the server should be serving; llama-server ignores it, vLLM checks it."""
    return _get("PT_LLM_MODEL", medgemma_model_id())


def llm_api_key() -> str:
    return _get("PT_LLM_API_KEY", "")


def llm_timeout() -> float:
    """Seconds to wait for one completion. A 4B thinking model is slow unbatched."""
    return float(_get_int("PT_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT))


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
