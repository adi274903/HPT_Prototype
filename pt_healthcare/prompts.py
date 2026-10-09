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
from .vocabulary import vocabulary_block


def build_categorizer_prompt(
    user_query: str,
    hospitals: Optional[Sequence[str]] = None,
    insurers: Optional[Sequence[str]] = None,
) -> str:
    """Entity-extraction prompt: query -> JSON with medical/hospital/insurer/medication.

    ``hospitals`` / ``insurers`` are the known values from the published price
    file. Supplying them replaces open-ended NER with slot-filling, which is what
    the downstream substring filters need — and it stops the model substituting a
    placeholder like "hospital" for a facility it cannot name. Omit them and the
    prompt is exactly the prototype's.
    """
    known = vocabulary_block(list(hospitals or []), list(insurers or []))

    return f"""You are a healthcare-query entity classifier and NER extractor.

    Analyze the user's query and extract entities into these categories:

    - medical: Medical conditions, symptoms, procedures, medications, specialties,
      tests, treatments, anatomy, diagnoses, CPT/HCPCS/ICD-related concepts,
      or healthcare services.
    - hospital: Hospitals, clinics, health systems, urgent-care centers, physician
      groups, laboratories, imaging centers, pharmacies, or other providers/facilities.
    - insurer: Health insurance companies, payers, Medicare, Medicaid, insurance
      plans, PBMs, prior-authorization organizations, or claims administrators.
{known}
    Return ONLY valid JSON. Do not explain. Do not add Markdown fences.

    Use exactly this schema:
    {{
      "medical": [],
      "hospital": [],
      "insurer": [],
      "medication" : [],
    }}

    Rules:
    - Extract exact relevant text spans from the query when possible.
    - An entity may belong to multiple categories only if it truly fits each one.
    - Use an empty list when no entity exists in a category.
    - Do not infer entities that were not stated.
    - Normalize obvious capitalization while preserving proper names.
    - Include generic references such as "my hospital" and "my insurance" if present.

    Examples:

    Query: Does Blue Cross cover an MRI at Houston Methodist?
    Output:
    {{
      "medical": ["MRI"],
      "hospital": ["Houston Methodist"],
      "insurer": ["Blue Cross"],
      "medication" : []
    }}

    Query: What is the recovery time after knee replacement surgery?
    Output:
    {{
      "medical": ["knee replacement surgery"],
      "hospital": [],
      "insurer": [],
      "medication" : []
    }}

    Query: Does Cigna cover a CT Scan? And would I need Tylenol?
    Output:
    {{
      "medical": ["CT Scan"],
      "hospital": [],
      "insurer": ["Cigna"],
      "medication" : ["Tylenol"]
    }}


    Query: How much would Houston Hospital cost me for colonsocopy?
    Output:
    {{
      "medical": ["colonoscopy"],
      "hospital": ["Houston Hospital"],
      "insurer": [],
      "medication" : []
    }}

    Now classify this query:

    Query: {user_query}

    JSON:
    """


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


def build_answer_prompt(user_query: str, output: Any, code_plausible: Any) -> str:
    """Patient-facing answer prompt grounded in the matched MRF rows."""
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

    Extracted entities:
    {output}

    Entity-linked code candidates:
    {code_plausible}

    Answer:
    """
