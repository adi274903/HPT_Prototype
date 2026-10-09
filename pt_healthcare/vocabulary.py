"""The closed sets of hospitals and payers present in the MRF.

The categorizer is open-ended NER over free text, which is why it invents things:
on the query "What hospital should i go to for my colonoscopy if I have my
Higmark BCBS plan?" it spent a paragraph deciding whether to emit the literal
string "hospital" as a placeholder, and it returned "Higmark BCBS plan" — the
user's typo, verbatim.

Handing it the values that actually exist in the file turns that job into
slot-filling against a known vocabulary. It also matters downstream: the price
filter is a case-insensitive *substring* match, so a payer the model invented
matches no rows at all.
"""

from __future__ import annotations

from typing import Any, List, Optional

#: Values we never want to offer as an entity.
_PLACEHOLDERS = {"nan", "none", "null", "n/a", "-", ""}

#: Keep the prompt bounded; a real MRF has tens of these, not thousands.
DEFAULT_LIMIT = 300


def column_vocabulary(
    frame: Any,
    column: str,
    limit: int = DEFAULT_LIMIT,
) -> List[str]:
    """Sorted, de-duplicated values of ``column`` in an MRF DataFrame.

    Returns ``[]`` for a missing column or an empty/absent frame, so callers can
    treat "no vocabulary" as "leave the prompt as it was".
    """
    columns = getattr(frame, "columns", None)
    if columns is None or column not in columns:
        return []

    try:
        series = frame[column].dropna().astype(str).str.strip()
    except Exception:
        return []

    # Case-insensitive: "Upmc Altoona" and "upmc altoona" are one hospital, and
    # the first spelling seen is the one kept.
    seen: dict = {}
    for value in series.tolist():
        key = value.casefold()
        if not value or key in _PLACEHOLDERS:
            continue
        seen.setdefault(key, value)

    return sorted(seen.values(), key=str.casefold)[:limit]


def vocabulary_block(
    hospitals: Optional[List[str]] = None,
    insurers: Optional[List[str]] = None,
) -> str:
    """Prompt text listing the known hospitals and payers, or ``""`` if unknown.

    Left empty when both lists are empty, so a pipeline built without an MRF gets
    exactly the original prompt.
    """
    hospitals = list(hospitals or [])
    insurers = list(insurers or [])

    if not hospitals and not insurers:
        return ""

    lines = [
        "",
        "    Known entries in the published price file. When the query refers to one",
        "    of these, return the exact string from the list below — matching loosely,",
        "    since a misspelling, an abbreviation or a partial name still refers to an",
        "    entry. Do not invent a name, and never return the list itself: only what",
        "    the query actually refers to.",
        "",
    ]

    if hospitals:
        lines.append(f"    hospitals ({len(hospitals)}):")
        lines.extend(f"    - {name}" for name in hospitals)

    if insurers:
        lines.append(f"    insurers ({len(insurers)}):")
        lines.extend(f"    - {name}" for name in insurers)

    lines.append("")

    return "\n".join(lines)
