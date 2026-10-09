"""The closed sets of hospitals and payers present in the MRF.

The categorizer is open-ended NER over free text, which is why it invents things.
On the query "What hospital should i go to for my colonoscopy if I have my
Higmark BCBS plan?" it spent a paragraph deciding whether to emit the literal
string "hospital" as a placeholder, and it returned "Higmark BCBS plan" — the
user's typo, verbatim.

Handing it the values that actually exist in the file turns that job into
slot-filling against a known vocabulary. It also matters downstream: the price
filter is a case-insensitive *substring* match, so a value the model invented
matches no rows at all — or, worse, matches an arbitrary slice of the file and
presents the wrong payer's or facility's prices as the user's.

Two guards live here, matching two failure modes:

* :func:`column_vocabulary` / :func:`vocabulary_block` — tell the model what exists.
* :func:`drop_placeholders` — refuse bare category words even when the model emits
  them anyway. The prompt used to *ask* for these ("include generic references
  such as 'my hospital'"), which is how ``{"hospital": ["hospital"]}`` happened.
"""

from __future__ import annotations

import difflib
from typing import Any, Iterable, List, Optional, Sequence

#: Values we never want to offer as an entity.
_PLACEHOLDERS = {"nan", "none", "null", "n/a", "-", ""}

#: Keep the prompt bounded; a real MRF has tens of these, not thousands.
DEFAULT_LIMIT = 300

#: Bare category words. None of these is a facility name, and the filter is a
#: substring match, so a generic term either matches nothing (leaving no evidence
#: for the answer) or matches a meaningless slice of the file.
GENERIC_HOSPITALS = {
    "hospital",
    "hospitals",
    "the hospital",
    "a hospital",
    "any hospital",
    "my hospital",
    "local hospital",
    "nearby hospital",
    "hospital or clinic",
    "clinic",
    "clinics",
    "doctor",
    "doctors",
    "physician",
    "physicians",
    "provider",
    "providers",
    "facility",
    "facilities",
    "medical center",
    "health system",
    "urgent care",
    "emergency room",
    "emergency department",
    "er",
    "pharmacy",
    "pharmacies",
    "laboratory",
    "lab",
    "imaging center",
    "care center",
}

#: Same idea for payers. "medicare" and "medicaid" are deliberately absent: they
#: are real named values that appear in an MRF, not category words.
GENERIC_INSURERS = {
    "insurance",
    "insurances",
    "the insurance",
    "my insurance",
    "my insurance plan",
    "insurance company",
    "insurance plan",
    "insurance coverage",
    "health insurance",
    "my plan",
    "my health plan",
    "health plan",
    "insurer",
    "insurers",
    "payer",
    "payers",
    "coverage",
    "my coverage",
    "benefits",
}

_GENERIC_BY_KIND = {
    "hospital": GENERIC_HOSPITALS,
    "insurer": GENERIC_INSURERS,
}


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


def drop_placeholders(values: Optional[Iterable[str]], kind: str) -> List[str]:
    """Remove bare category words from a categorizer entity list.

    ``kind`` is ``"hospital"`` or ``"insurer"``. An unrecognised ``kind`` passes
    values straight through, so this can only ever remove generic terms.
    """
    generic = _GENERIC_BY_KIND.get(kind)

    kept: List[str] = []
    for value in values or []:
        if not value:
            continue
        if generic is not None and str(value).strip().casefold() in generic:
            continue
        kept.append(value)

    return kept


def resolve_to_vocabulary(
    values: Optional[Iterable[str]],
    vocabulary: Sequence[str],
    cutoff: float = 0.6,
    margin: float = 0.05,
) -> List[str]:
    """Snap near-miss names onto the real values in the price file.

    The filter is a substring match, so a near miss is not a near miss — it is a
    total miss. ``"Highmark Plan"`` does not contain-match ``"Highmark BCBS of
    PA"``, and the query returns no rows at all. Asking the model to use the exact
    string helps but does not guarantee it, so this closes the gap in code.

    Resolution order, first hit wins:

    1. exact match, ignoring case — just take the file's spelling;
    2. the term is a substring of exactly one entry, or one entry is a substring
       of the term (``"highmark"`` -> ``"Highmark BCBS of PA"``,
       ``"UPMC Presbyterian"`` -> ``"Upmc Presbyterian Shadyside"``);
    3. one entry is a clear best fuzzy match (``"higmark bcbs plan"``).

    Anything ambiguous is left untouched: guessing between two payers is worse
    than leaving the model's value, which the caller can report. An empty
    vocabulary passes everything through, so a pipeline with no MRF behaves
    exactly as before.
    """
    values = [value for value in (values or []) if value]

    if not vocabulary:
        return list(values)

    by_key = {entry.casefold(): entry for entry in vocabulary}
    folded = [entry.casefold() for entry in vocabulary]

    resolved: List[str] = []

    for value in values:
        term = str(value).strip()
        key = term.casefold()

        if key in by_key:
            resolved.append(by_key[key])
            continue

        # by_key maps casefolded -> original spelling, so iterate its keys.
        contained = [
            entry_key
            for entry_key in by_key
            if key in entry_key or entry_key in key
        ]

        if len(contained) == 1:
            resolved.append(by_key[contained[0]])
            continue

        scored = sorted(
            (
                (difflib.SequenceMatcher(None, key, entry_key).ratio(), entry_key)
                for entry_key in folded
            ),
            reverse=True,
        )
        best_score, best_key = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score >= cutoff and (best_score - runner_up) >= margin:
            resolved.append(by_key[best_key])
        else:
            resolved.append(term)

    return list(dict.fromkeys(resolved))


def vocabulary_block(
    hospitals: Optional[Sequence[str]] = None,
    insurers: Optional[Sequence[str]] = None,
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
