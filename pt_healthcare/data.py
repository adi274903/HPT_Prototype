"""Data access: Hugging Face auth, Qdrant client, MRF CSV, archive extraction.

Replaces the Colab-specific cells (``drive.mount`` + ``!cp`` + ``!tar``) with
plain Python so the pipeline runs anywhere the data is on disk.
"""

from __future__ import annotations

import os
import tarfile
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from . import config

#: Fallback collection names, used only if the environment says nothing.
DEFAULT_COLLECTIONS: Sequence[str] = ("cpt_medte", "hcpcs_medte")


def default_collections() -> Sequence[str]:
    """The collections to report on, resolved from the environment now."""
    return (config.cpt_collection(), config.hcpcs_collection())


def login_huggingface(token: Optional[str] = None) -> None:
    """Authenticate against Hugging Face for the gated MedGemma/MedTE models."""
    import huggingface_hub

    token = token or config.hf_token_from_env()

    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set one of "
            f"{', '.join(config.HF_TOKEN_ENV_VARS)} in your environment or .env "
            "(see .env.example)."
        )

    huggingface_hub.login(token)


def get_qdrant_client(path: Optional[str] = None, **kwargs: Any) -> Any:
    """Open a local Qdrant store (the on-disk snapshot directory).

    Defaults to ``PT_QDRANT_PATH`` at call time, so a ``.env`` loaded after
    import still applies.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(path=path or config.qdrant_dir(), **kwargs)


def collection_stats(
    client: Any,
    collections: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Return ``{collection_name: points_count}`` for the given collections."""
    stats: Dict[str, int] = {}

    for collection_name in collections or default_collections():
        info = client.get_collection(collection_name)
        stats[collection_name] = info.points_count

    return stats


def extract_archive(archive_path: str, dest_dir: str) -> None:
    """Extract a ``.tar.gz`` Qdrant snapshot (replaces the notebook's ``!tar``)."""
    os.makedirs(dest_dir, exist_ok=True)

    with tarfile.open(archive_path) as archive:
        archive.extractall(dest_dir)


def load_mrf_data(path: Optional[str] = None, low_memory: bool = False) -> pd.DataFrame:
    """Load the cleaned hospital MRF CSV.

    ``low_memory=False`` avoids the mixed-type ``DtypeWarning`` the prototype
    emitted and gives stable dtypes for the code/price columns. Defaults to
    ``PT_MRF_CSV`` at call time.
    """
    csv_path = path or config.mrf_csv_path()

    return pd.read_csv(csv_path, low_memory=low_memory)
