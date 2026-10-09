"""Small normalization helpers shared across the pipeline.

These are lifted verbatim (semantics preserved) from the notebook's
``_clean_list``, ``_as_clean_list``, ``_normalize_series`` and ``_contains_any``
so that filtering behaves exactly as it did in the prototype.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

import pandas as pd

# Values that should be treated as "empty" after stringification.
_NULLISH = {"nan", "none", "null"}


def clean_list(values: Any) -> List[str]:
    """Normalize ``None``, a single string, or a list into a clean list of strings.

    Case is preserved (used for building semantic-search queries and for
    validating model-selected codes). Notebook cell 12.
    """
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    return [
        str(value).strip()
        for value in values
        if value is not None
        and str(value).strip()
        and str(value).strip().lower() not in _NULLISH
    ]


def as_clean_list(values: Any, uppercase: bool = False) -> List[str]:
    """Clean and de-duplicate a list-like input, preserving first-seen order.

    When ``uppercase`` is False the values are casefolded (case-insensitive
    matching); when True they are uppercased (code matching). Notebook cell 20.
    """
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    cleaned: List[str] = []

    for value in values:
        if value is None:
            continue

        value = str(value).strip()

        if not value or value.casefold() in _NULLISH:
            continue

        cleaned.append(value.upper() if uppercase else value.casefold())

    return list(dict.fromkeys(cleaned))


def normalize_series(series: pd.Series, uppercase: bool = False) -> pd.Series:
    """Standardize a DataFrame column for reliable text comparisons."""
    normalized = series.fillna("").astype(str).str.strip()

    return normalized.str.upper() if uppercase else normalized.str.casefold()


def contains_any(series: pd.Series, terms: Sequence[str]) -> pd.Series:
    """Case-insensitive substring matching against any of ``terms``.

    Example:
        term = "UPMC Presbyterian"
        dataset = "UPMC Presbyterian Shadyside"
        -> match

    Returns ``True`` for every row when no terms were supplied.
    """
    if not terms:
        return pd.Series(True, index=series.index)

    normalized = normalize_series(series)
    pattern = "|".join(re.escape(term) for term in terms)

    return normalized.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False,
    )
