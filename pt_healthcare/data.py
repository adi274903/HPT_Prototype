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

DEFAULT_COLLECTIONS: Sequence[str] = (config.CPT_COLLECTION, config.HCPCS_COLLECTION)


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
    """Open a local Qdrant store (the on-disk snapshot directory)."""
    from qdrant_client import QdrantClient

    return QdrantClient(path=path or config.QDRANT_PATH, **kwargs)


def collection_stats(
    client: Any,
    collections: Iterable[str] = DEFAULT_COLLECTIONS,
) -> Dict[str, int]:
    """Return ``{collection_name: points_count}`` for the given collections."""
    stats: Dict[str, int] = {}

    for collection_name in collections:
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
    emitted and gives stable dtypes for the code/price columns.
    """
    csv_path = path or config.MRF_CSV_PATH

    return pd.read_csv(csv_path, low_memory=low_memory)
