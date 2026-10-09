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
    - Only extract a hospital or an insurer that the query actually names or
      otherwise identifies. A bare category word is not an entity: "what hospital
      should I go to" and "does my insurance cover this" name neither, so those
      lists stay empty.
    - Never emit a generic word such as "hospital", "clinic", "doctor",
      "provider", "insurance", "my hospital", "my insurance" or "my plan" as an
      entity. Those never match a real entry in the price file.

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


    Query: Which hospital should I go to for a colonoscopy?
    Output:
    {{
      "medical": ["colonoscopy"],
      "hospital": [],
      "insurer": [],
      "medication" : []
    }}

    "hospital" is empty because the query names no facility. Asking *which*
    hospital you should use is not naming one.

    Query: What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?
    Output:
    {{
      "medical": ["diagnostic mammogram"],
      "hospital": ["UPMC Presbyterian"],
      "insurer": ["UPMC Health Plan"],
      "medication" : []
    }}

    Every category is filled from the query text. Only the hospital and insurer
    values additionally get matched against the reference lists.

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


def build_general_prompt(user_query: str) -> str:
    """A general question about a procedure, test or condition.

    No index lookup happens on this path, so the prompt says so explicitly: the
    model must not claim to have looked anything up, must not guess a code, and
    must not quote a price it has no data for.
    """
    return f"""You are a healthcare navigation assistant answering a patient's
    general question about a procedure, test, condition or billing term.

    Rules:
    - Answer from your own clinical knowledge. No index lookup was performed for
      this question, so do not claim to have looked anything up.
    - Say that this is general information, not advice about their situation.
    - Never guess or assert a specific billing code. If the patient wants a code,
      tell them this service can look up a code they name; if they want a price,
      it can price a procedure at a named hospital and plan.
    - Never quote or estimate a price; you have no price data here.
    - Never advise the patient on their own care or say whether a procedure suits
      them. Point them to their clinician for anything about their own situation.
    - Keep it clear and patient-friendly, and short.

    Patient question:
    {user_query}

    Answer:
    """


def build_explain_prompt(user_query: str, context: str) -> str:
    """Explain a code or a procedure, grounded in whatever the index returned.

    ``context`` carries the retrieved descriptors, or an explicit note that the
    code is absent. The strictness is asymmetric on purpose: a billing code must
    be grounded, because codes are reissued every year and a remembered
    definition may be wrong; general clinical education may come from the model's
    own knowledge, because that is what it is for.
    """
    return f"""You are a healthcare navigation assistant helping a patient
    understand billing codes and medical procedures.

    Rules:
    - If the index entries below cover what was asked, explain from them and name
      the codes plainly.
    - If a specific billing code is marked NOT in the index, say that this
      service does not have a definition for it. Never describe a specific code
      from memory: billing codes are reissued every year and a remembered
      definition may be out of date.
    - For a general question about a procedure, test or condition, you may answer
      from your own clinical knowledge, but say that it is general information.
    - Never quote or estimate a price; you have no price data here.
    - Never advise the patient on their own care or say whether a procedure suits
      them. Point them to their clinician for anything about their own situation.
    - Keep it clear and patient-friendly.

    Patient question:
    {user_query}

    Index entries retrieved for this question:
    {context}

    Answer:
    """
