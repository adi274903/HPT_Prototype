"""Semantic code retrieval over the Qdrant CPT/HCPCS collections.

Notebook cells 11-14, refactored from module-level globals into a
``CodeRetriever`` object that takes the embedder and Qdrant client by
injection (which also makes it testable without a GPU or a real store).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from .config import CPT_COLLECTION, HCPCS_COLLECTION, TOP_K
from .utils import clean_list

_EMPTY_COLUMNS = ["rank", "score", "code", "text"]


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


def build_medical_query(categorized: Mapping[str, Any]) -> str:
    """Create one semantic-search query from categorizer output.

    Example:
        {
            "medical": ["diagnostic mammogram", "breast imaging"],
            ...
        }
        -> "diagnostic mammogram breast imaging"
    """
    if not isinstance(categorized, dict):
        raise TypeError("`categorized` must be a dictionary.")

    medical_terms = clean_list(categorized.get("medical", []))

    return " ".join(medical_terms).strip()


class CodeRetriever:
    """Embed a query and search a Qdrant collection for candidate codes."""

    def __init__(
        self,
        embed_model: Any,
        client: Any,
        top_k: int = TOP_K,
    ) -> None:
        self.embed_model = embed_model
        self.client = client
        self.top_k = top_k

    # ------------------------------------------------------------------
    def retrieve_codes(
        self,
        query: str,
        collection_name: str,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Embed ``query``, search a Qdrant collection, and return candidates."""
        if not query or not str(query).strip():
            return _empty_candidates()

        query_vector = self.embed_model.encode(
            str(query).strip(),
            normalize_embeddings=True,
        ).tolist()

        limit = self.top_k if top_k is None else top_k

        hits = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

        return pd.DataFrame(
            [
                {
                    "rank": rank,
                    "score": float(hit.score),
                    "code": str(hit.payload.get("code", "")).strip(),
                    "text": str(hit.payload.get("text", "")).strip(),
                }
                for rank, hit in enumerate(hits, start=1)
            ]
        )

    # ------------------------------------------------------------------
    def retrieve_cpt(self, categorized: Mapping[str, Any], top_k: Optional[int] = None) -> pd.DataFrame:
        """Search the CPT collection using the categorizer's medical entities."""
        query = build_medical_query(categorized)

        if not query:
            return _empty_candidates()

        return self.retrieve_codes(
            query=query,
            collection_name=CPT_COLLECTION,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    def retrieve_hcpcs(self, categorized: Mapping[str, Any], top_k: Optional[int] = None) -> pd.DataFrame:
        """Search the HCPCS collection using the categorizer's medical entities."""
        query = build_medical_query(categorized)

        if not query:
            return _empty_candidates()

        return self.retrieve_codes(
            query=query,
            collection_name=HCPCS_COLLECTION,
            top_k=top_k,
        )
