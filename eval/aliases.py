"""Fixed literal lists for the prompt sets: payer aliases, personas, profiles, invented services.

Nothing here is generated at runtime: B2, A2, A3 and D2 need a closed set so the eval
is reproducible and a reviewer can audit every string. Every PAYER_ALIASES key is an
exact MRF ``payer_name``; ``python3 eval/aliases.py --verify`` checks that against the
combos table.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List


# ── B2: canonical payer -> patient phrasings ────────────────────────────────
PAYER_ALIASES: Dict[str, List[str]] = {
    "United Healthcare": ["UHC", "United Health Care", "United"],
    "UPMC Health Plan": [
        "the hospital's affiliated insurance plan",
        "UPMC's own plan",
        "the UPMC plan",
    ],
    "Highmark BCBS of PA": [
        "Blue Cross Blue Shield of Pennsylvania", "Highmark Blues", "BCBS PA",
    ],
    "Coventry/First Health": ["First Health", "Coventry"],
    "Health Partners Plans (Jefferson Health Plan)": [
        "Jefferson Health Plan", "Health Partners",
    ],
    "Aetna of PA": ["Aetna Pennsylvania", "the Aetna PA plan"],
    "United Healthcare Community Plan for Families": [
        "UHC Community Plan", "United community plan",
    ],
    "Geisinger": ["Geisinger Health Plan", "the Geisinger plan"],
    "Humana": ["Humana Gold", "the Humana plan"],
    "Cigna": ["Cigna HealthCare", "my Cigna plan"],
}

# ── A2: ten personas (personality, language level, recall, confusion) ───────
PERSONAS: List[Dict[str, str]] = [
    {"id": "P01", "label": "anxious first-timer", "language": "plain",
     "recall": "remembers the plan, forgets the hospital", "confusion": "none"},
    {"id": "P02", "label": "terse clinician-adjacent", "language": "clinical",
     "recall": "knows the CPT code by heart", "confusion": "none"},
    {"id": "P03", "label": "elderly low-literacy", "language": "very plain, short sentences",
     "recall": "forgets the plan, remembers the hospital", "confusion": "colonoscopy vs endoscopy"},
    {"id": "P04", "label": "bilingual parent", "language": "ESL phrasing",
     "recall": "remembers the plan", "confusion": "deductible vs negotiated rate"},
    {"id": "P05", "label": "impatient mobile user", "language": "fragments, no punctuation",
     "recall": "remembers nothing exactly", "confusion": "plan tiers"},
    {"id": "P06", "label": "over-explainer", "language": "rambling, three sentences of context",
     "recall": "accurate", "confusion": "names two services at once"},
    {"id": "P07", "label": "self-pay cash patient", "language": "plain",
     "recall": "accurate", "confusion": "asks for cash price, not insurance"},
    {"id": "P08", "label": "caregiver asking for a relative", "language": "plain",
     "recall": "remembers the plan, not the hospital", "confusion": "whose plan it is"},
    {"id": "P09", "label": "skeptical price shopper", "language": "precise, comparative",
     "recall": "accurate", "confusion": "asks which number is real"},
    {"id": "P10", "label": "confused by abbreviations", "language": "plain",
     "recall": "mangles the payer name", "confusion": "HMO/PPO/ASO"},
]

# ── A3: ten demographic / communication profiles ────────────────────────────
PROFILES: List[Dict[str, str]] = [
    {"id": "D01", "age": "72", "context": "retired schoolteacher in Altoona, PA",
     "framing": "wants plain language"},
    {"id": "D02", "age": "34", "context": "warehouse supervisor in Pittsburgh",
     "framing": "asks how the deductible applies"},
    {"id": "D03", "age": "45", "context": "small-business owner in Erie, PA",
     "framing": "compares hospitals before committing"},
    {"id": "D04", "age": "28", "context": "graduate student in State College",
     "framing": "low income, asks about the cash price too"},
    {"id": "D05", "age": "58", "context": "steelworker in Johnstown",
     "framing": "hard of hearing, prefers written detail"},
    {"id": "D06", "age": "39", "context": "home health aide in Harrisburg",
     "framing": "tentative about paperwork"},
    {"id": "D07", "age": "51", "context": "truck driver in Carlisle, rarely home",
     "framing": "blunt, wants one number"},
    {"id": "D08", "age": "66", "context": "farmer in Wellsboro",
     "framing": "drives an hour to whichever hospital is cheaper"},
    {"id": "D09", "age": "23", "context": "dental hygienist in Lititz",
     "framing": "asks a relative to translate"},
    {"id": "D10", "age": "47", "context": "municipal clerk in Altoona",
     "framing": "anxious about surprise bills"},
]

# ── D2: invented services (must have NO accepted CPT/HCPCS code) ────────────
INVENTED_SERVICES: List[str] = [
    "bariatric duodenal sleeve plication with fluoroscopic mapping",
    "transdermal vagal neuromodulation titration visit",
    "whole-body cryogenic recovery session",
    "intraoral myofascial release with photobiomodulation",
    "pediatric sensory integration immersion hour",
    "prophylactic cerebrospinal fluid washout",
    "photonic collagen remodeling of the plantar fascia",
    "ambulatory gut microbiome transplantation capsule",
    "low-dose ketamine resilience infusion",
    "electroacupuncture lymphatic drainage protocol",
    "subdermal ozone oxygen therapy panel",
    "quantum bioresonance allergy desensitization",
]


def verify_payers(db_path: str) -> int:
    """Assert every alias key is a real payer_name in the prepared combos table."""
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    known = {r[0] for r in con.execute("SELECT DISTINCT payer FROM combos").fetchall()}
    missing = [p for p in PAYER_ALIASES if p not in known]

    for payer in PAYER_ALIASES:
        print(f"{'ok  ' if payer in known else 'FAIL'} {payer}")
    if missing:
        print(f"\n{len(missing)} alias key(s) not in the MRF: {missing}")
        near = sorted(known)
        for bad in missing:
            close = [k for k in near if bad.split()[0].lower() in k.lower()]
            print(f"  {bad!r} -> candidates: {close[:4]}")
    return len(missing)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--db", default=".cache/mrf_eval.duckdb")
    a = ap.parse_args()

    if a.verify:
        return 1 if verify_payers(a.db) else 0
    print(f"{len(PAYER_ALIASES)} payers, {len(PERSONAS)} personas, "
          f"{len(PROFILES)} profiles, {len(INVENTED_SERVICES)} invented services")
    return 0


if __name__ == "__main__":
    sys.exit(main())
