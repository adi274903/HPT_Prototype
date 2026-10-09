"""Intent analysis: one query in, one tool name out.

What the patient asks decides *which* capability runs; the capabilities
themselves are tools. There are two:

* ``price``   — what something will cost. The body of this tool is
  ``HealthcarePricingPipeline``, the existing 7-step orchestration, unchanged.
* ``lookup``  — what a *specific* billing code means. The query named a code, so
  the descriptor is fetched exactly from the index and the answer is grounded in
  it. Never from memory: CPT is reissued every January and a remembered
  definition may be wrong.
* ``explain`` — a general question about a procedure, test or condition, with no
  code named. Answered straight from MedGemma, which is a clinical model and knows
  this material. No retrieval happens, so there is no query vector to get wrong.

Selection is deterministic first. Choosing the wrong tool produces a
confidently wrong-shaped answer, so this must not depend on sampling from a 4B
clinical fine-tune. The model contributes a hint only where the lexical cues run
out, which is the one place rules genuinely cannot decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

PRICE = "price"
LOOKUP = "lookup"
EXPLAIN = "explain"

#: Every tool name a classifier may return.
TOOL_NAMES = (PRICE, LOOKUP, EXPLAIN)

#: A request about money, in the words patients use.
PRICE_CUES = (
    "cost",
    "costs",
    "price",
    "prices",
    "pricing",
    "how much",
    "charge",
    "charges",
    "charged",
    "billed",
    "billing",
    "bill",
    "pay",
    "pays",
    "paying",
    "payment",
    "out of pocket",
    "out-of-pocket",
    "deductible",
    "coinsurance",
    "copay",
    "co-pay",
    "coverage",
    "covered",
    "reimburse",
    "reimbursed",
    "reimbursement",
    "negotiated",
    "discount",
    "cash price",
    "afford",
    "affordable",
    "cheaper",
    "cheapest",
    "expense",
    "expenses",
    "expensive",
    "estimate",
    "quote",
    "fee",
    "fees",
    "money",
    "dollar",
    "dollars",
    "spend",
    "$",
)

# Word boundaries, because substring matching would fire on "rate" inside
# "accurate" and route a definition question down the price path.
_PRICE_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(cue) for cue in PRICE_CUES),
    re.IGNORECASE,
)

# CPT is 5 digits; Category II/III are 4 digits plus F or T; HCPCS Level II is a
# letter plus 4 digits. The bare-code case is the one the model must never decide.
_CODE_PATTERNS = (
    re.compile(r"\b\d{5}\b"),
    re.compile(r"\b\d{4}[FTft]\b"),
    re.compile(r"\b[A-Za-z]\d{4}\b"),
)


@dataclass
class Intent:
    """Which tool should run, and why."""

    tool: str
    codes: List[str] = field(default_factory=list)
    reason: str = ""
    #: "rules" when the lexical pass decided, "model" when the hint did,
    #: "default" when nothing matched and we fell back.
    source: str = "rules"


def extract_codes(query: str) -> List[str]:
    """Bare CPT/HCPCS codes in the query, upper-cased, first-seen order."""
    found: List[str] = []

    for pattern in _CODE_PATTERNS:
        for match in pattern.findall(query or ""):
            code = match.upper()
            if code not in found:
                found.append(code)

    return found


def has_price_cue(query: str) -> bool:
    """True when the wording can only be a question about money."""
    return bool(_PRICE_RE.search(query or ""))


def lexical_intent(
    query: str,
    entities: Optional[Mapping[str, Any]] = None,
) -> Optional[Intent]:
    """Rule-based selection, or ``None`` when no cue matched at all.

    Precedence is deliberate:

    1. price wording — explicit, and it wins even when a code is present, so
       "how much does 45378 cost at UPMC" prices rather than defines;
    2. a bare code — "what is CPT 45378" is a lookup;
    3. a named facility or payer — naming your hospital or plan is a price
       signal even without the word "cost";
    4. nothing — the caller decides.
    """
    codes = extract_codes(query)

    if has_price_cue(query):
        return Intent(PRICE, codes, "price wording in the query")

    if codes:
        return Intent(LOOKUP, codes, "bare code in the query")

    entities = entities or {}

    if any(entities.get(key) for key in ("hospital", "insurer")):
        return Intent(PRICE, codes, "names a facility or payer")

    return None


def classify(
    query: str,
    engine: Optional[Any] = None,
    entities: Optional[Mapping[str, Any]] = None,
) -> Intent:
    """Pick the tool for ``query``.

    ``engine`` is optional and only consulted when the lexical pass finds no cue
    at all — the resident case, where rules have nothing to go on. A model that
    is missing, broken or nonsensical degrades to the ``explain`` default rather
    than raising: a routing failure must not take down the request.
    """
    lexical = lexical_intent(query, entities)

    if lexical is not None:
        return lexical

    hinted = model_intent(engine, query)

    if hinted is not None:
        return hinted

    return Intent(
        EXPLAIN,
        [],
        "no price, code or facility cue",
        source="default",
    )


def model_intent(engine: Optional[Any], query: str) -> Optional[Intent]:
    """Ask the model to break the tie, or ``None`` if it cannot.

    Returns ``None`` for a missing engine, an exception, or output naming no
    known tool. Separators are normalised first, so "code lookup" and
    "code_lookup" both read as the tool name.
    """
    if engine is None:
        return None

    try:
        raw = engine.classify_intent(query)
    except Exception:
        return None

    text = str(raw or "").lower().replace("-", "_").replace(" ", "_")

    for name in TOOL_NAMES:
        if name in text:
            return Intent(
                name,
                extract_codes(query),
                "model hint",
                source="model",
            )

    return None
