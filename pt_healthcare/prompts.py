"""Prompt builders.

The prompt text is preserved verbatim from the prototype notebook so the
models see exactly the same instructions. Each builder is a pure function of
its inputs (returns a string) which makes it trivially unit-testable.

Cells: 15 (categorizer), 17 (decision), 18 (answer).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import config
from .parsing import compact_candidates


def build_categorizer_prompt(user_query: str) -> str:
    """Entity-extraction prompt: query -> JSON with medical/hospital/insurer/medication.

    Kept deliberately short. An earlier version carried the whole hospital and payer
    vocabulary inline — about 280 tokens of names — so that extraction became
    slot-filling rather than open-ended. That list was both redundant and harmful:
    redundant because ``vocabulary.resolve_to_vocabulary()`` already snaps a near
    miss onto the real entry ("UPMC Presby" -> "Upmc Presbyterian Shadyside",
    "Higmark BCBS plan" -> "Highmark BCBS of PA"), and harmful because with two
    categories holding an enumerated list and two not, the model filled the listed
    ones and returned ``medical: []`` for a query that plainly named a procedure.
    """
    return f"""You are a healthcare-query entity extractor.

    Extract entities from the user's query into four categories:

    - medical: the procedure, test, condition, symptom, treatment or service named
      in the query. This is the one that matters — if the query names a service, it
      belongs here.
    - hospital: a named facility, clinic, health system or provider.
    - insurer: a named health plan, payer or insurance company.
    - medication: a specific drug named in the query.

    Return ONLY valid JSON. Do not explain. Do not add Markdown fences.

    Use exactly this schema:
    {{
      "medical": [],
      "hospital": [],
      "insurer": [],
      "medication" : [],
    }}

    Rules:
    - Copy the query's own wording for each entity.
    - Use an empty list when a category has nothing in it.
    - Never emit a bare category word: "hospital", "clinic", "doctor", "provider",
      "insurance" or "my plan" are not entities. A query asking which hospital to
      use names no facility.
    - A service named by a short or clipped phrase still counts. "Cost of
      colonoscopy at UPMC Presby?" names a colonoscopy; "How much would an X-ray
      cost?" names an X-ray.

    Examples:

    Query: Does Blue Cross cover an MRI at Houston Methodist?
    Output: {{"medical": ["MRI"], "hospital": ["Houston Methodist"], "insurer": ["Blue Cross"], "medication": []}}

    Query: Cost of colonoscopy at UPMC Presby?
    Output: {{"medical": ["colonoscopy"], "hospital": ["UPMC Presby"], "insurer": [], "medication": []}}

    Query: How much would an X-ray cost?
    Output: {{"medical": ["X-ray"], "hospital": [], "insurer": [], "medication": []}}

    Query: Which hospital should I go to for a colonoscopy?
    Output: {{"medical": ["colonoscopy"], "hospital": [], "insurer": [], "medication": []}}

    Now classify this query:

    Query: {user_query}

    JSON:
    """


def build_intent_prompt(user_query: str) -> str:
    """Route a patient question to one of the tools.

    Only used when the lexical cues find nothing — this is the tie-breaker, not
    the router.
    """
    return f"""Route a patient's question to one of three capabilities.

    price   - what something will cost, what a bill means, what an amount covers,
              for a named hospital, plan or procedure.
    lookup  - the meaning of a specific billing code (CPT or HCPCS) that the
              question itself names, such as "what is 45378".
    explain - what a procedure, test or condition is, in general terms, when no
              code is named and no cost is asked about.

    Answer with exactly one word: price, lookup or explain. Do not explain your
    choice.

    Question: {user_query}

    Answer:"""
def build_decision_prompt(categorization: Mapping[str, Any], user_query: str) -> str:
    """Ask MedGemma to choose a small set of CPT/HCPCS code candidates.

    ``categorization`` is expected to carry ``medical``, ``cpt_candidates`` and
    ``hcpcs_candidates`` (the latter two as records).
    """
    cpt_candidates = compact_candidates(
        categorization.get("cpt_candidates", []),
        max_candidates=config.max_candidates_per_family(),
    )

    hcpcs_candidates = compact_candidates(
        categorization.get("hcpcs_candidates", []),
        max_candidates=config.max_candidates_per_family(),
    )

    return f"""Select billing-code candidates.

Procedure: {json.dumps(categorization.get("medical", []))}

CPT candidates:
{json.dumps(cpt_candidates, ensure_ascii=False)}

HCPCS candidates:
{json.dumps(hcpcs_candidates, ensure_ascii=False)}

Query: {user_query}

Return ONLY one complete JSON object. No thought, explanation, Markdown, or text outside JSON.

Rules:
- Choose codes only from the supplied candidates.
- Select at most 5 unique CPT codes.
- Select at most 5 unique HCPCS codes.
- Do not repeat codes.
- "use_codes" must be exactly one of: "cpt", "hcpcs", "both", "none".
- If no selected CPT codes, use an empty CPT list.
- If no selected HCPCS codes, use an empty HCPCS list.
- Stop immediately after the final }}.

Required JSON:
{{
  "use_codes": "cpt",
  "cpt_list": [],
  "hcpcs_list": []
}}"""


_ANSWER_ROW_FIELDS = (
    ("row_id", "row"),
    ("CPT", "CPT"),
    ("HCPCS", "HCPCS"),
    ("hospital_name", "hospital"),
    ("payer_name", "payer"),
    ("plan_name", "plan"),
    ("standard_charge|negotiated_dollar", "negotiated"),
    ("standard_charge|discounted_cash", "cash"),
    ("standard_charge|gross", "gross"),
)


def _records(value: Any) -> List[Dict[str, Any]]:
    """Rows from a DataFrame or a list of dicts, without importing pandas here."""
    if value is None:
        return []
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if hasattr(value, "to_dict"):
        try:
            return list(value.to_dict(orient="records"))
        except TypeError:
            return []
    return []


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${number:,.2f}"


def build_answer_context(
    output: Any,
    code_plausible: Any,
    *,
    max_rows: int = 12,
    max_candidates: int = 8,
) -> str:
    """A compact, grounded context for the answer step.

    The step used to receive the entire pipeline dump — every candidate and every
    matched MRF row with all nineteen columns — which cost about 8,000 prompt tokens
    before the model generated anything. The answer only needs a handful of rows and
    the codes that were chosen, one line each.
    """
    lines: List[str] = []

    chosen: List[str] = []
    plausible = code_plausible if isinstance(code_plausible, Mapping) else {}
    for family in ("cpt_list", "hcpcs_list", "codes"):
        for code in plausible.get(family) or []:
            text = str(code).strip()
            if text and text not in chosen:
                chosen.append(text)
    if not chosen and isinstance(output, Mapping):
        for family in ("cpt_list", "hcpcs_list"):
            for code in output.get(family) or []:
                text = str(code).strip()
                if text and text not in chosen:
                    chosen.append(text)

    if chosen:
        lines.append("Codes under consideration: " + ", ".join(chosen[:max_candidates]))

    if plausible and plausible.get("match_count") is not None:
        lines.append(f"Rows matched before the display cap: {plausible['match_count']}")

    rows = _records(plausible.get("price_summary") if plausible else None)
    if not rows:
        rows = _records(output.get("matches") if isinstance(output, Mapping) else None)

    priced = [row for row in rows if row.get("standard_charge|negotiated_dollar")]
    unpriced = [row for row in rows if not row.get("standard_charge|negotiated_dollar")]
    selected = (priced + unpriced)[:max_rows]

    if not selected:
        lines.append("No priced rows matched for this query.")
    else:
        lines.append(f"Published price rows ({len(selected)} of {len(rows)}):")
        for row in selected:
            parts = []
            for key, label in _ANSWER_ROW_FIELDS:
                value = row.get(key)
                if value is None or (isinstance(value, float) and value != value):
                    continue
                text = _money(value) if "|" in key or key == "median_amount" else str(value).strip()
                if text:
                    parts.append(f"{label} {text}")
            if parts:
                lines.append(" | ".join(parts))

    return "\n".join(lines)


def build_answer_prompt(user_query: str, output: Any, code_plausible: Any) -> str:
    """Patient-facing answer prompt grounded in the matched MRF rows.

    The context is compacted here rather than in the caller so every path — UI,
    CLI, eval — pays the same small prompt.
    """
    context = build_answer_context(output, code_plausible)
    return f"""You are a helpful healthcare navigation assistant.

    Answer the user's question using the original question, extracted entities, and
    entity-linked plausible code candidates below.

    The codes are candidates only. They are not confirmed billing codes, coverage
    guarantees, price quotes, or medical advice.

    Rules:
    - Mention each code only with its linked medical entity.
    - Never imply an unconfirmed code is final.
    - Do not estimate a specific cost unless one is explicitly provided.
    - Explain that final billing depends on documentation and services actually performed.
    - For price or coverage questions, recommend confirming with the provider's billing
      office and the user's insurer.
    - Try to use all costs mentioned i nthe data, negotiated dollar, cash discount,etc. Mostly a patient would like to know out of pocket charge, if asked.
    - Explain to the patient also what each costm ight mean, if asked.
    - Keep the answer clear and patient-friendly.

    User query:
    {user_query}

    {context}

    Answer:
    """
