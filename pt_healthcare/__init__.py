"""Healthcare Price Transparency (HPT) prototype.

A modular refactor of the ``Copy_of_Try_PT-3`` Colab notebook:

    entity extraction -> CPT/HCPCS retrieval -> code decision
    -> code validation -> MRF price filtering -> patient-facing answer.
"""

from .config import (
    CPT_COLLECTION,
    EMBEDDING_MODEL_ID,
    HCPCS_COLLECTION,
    MEDGEMMA_MODEL_ID,
    MRF_CSV_PATH,
    QDRANT_PATH,
    TOP_K,
)

__version__ = "0.1.0"

__all__ = [
    "CPT_COLLECTION",
    "EMBEDDING_MODEL_ID",
    "HCPCS_COLLECTION",
    "MEDGEMMA_MODEL_ID",
    "MRF_CSV_PATH",
    "QDRANT_PATH",
    "TOP_K",
    "__version__",
]
