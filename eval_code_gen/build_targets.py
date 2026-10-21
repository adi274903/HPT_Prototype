"""Derive the intent and entity targets for ``eval_dataset`` — gazetteer lookups only.

The dataset's seven keys describe the *answer*: which codes, which MRF rows, which
hospital and payer. They say nothing about what the extraction step should return,
so nothing could score it. This script adds that, without a model and without
annotation: the hospitals, payers and service phrases are all known strings, so a
substring scan over the rendered question recovers the entities and their spans.

Anything it cannot place is reported, not guessed. The audit block at the end
cross-checks the recovered entities against the dataset's own ground truth, which
makes this a second, independent derivation of the same facts.

    python eval_code_gen/build_targets.py --dataset eval_dataset --db .cache/mrf_eval.duckdb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb

# The generators live next to the dataset in the published tree (eval_code_gen/)
# but in eval/ in the development checkout; resolve whichever this is.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent / "eval"):
    if (_candidate / "categories.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

from aliases import INVENTED_SERVICES, PAYER_ALIASES  # noqa: E402
from categories import CATEGORIES, load_catalog  # noqa: E402

# The published router (pt_healthcare/router.py) labels three capabilities. Every
# case in the 600 asks what something costs, so the expected label is "price"
# throughout; the map exists so a routing set (E1 code lookup, E2 general
# explanation) scores without touching this file.
INTENT_BY_SUBTYPE: Dict[str, str] = {"E1": "lookup", "E2": "explain"}
INTENT_DEFAULT = "price"
FIELD_ORDER = ("medical", "hospital", "insurer", "medication")
ARTICLE_RX = re.compile(r"^(?:a|an|the|my)\s+", re.IGNORECASE)


def normalize(value: str) -> str:
    """Comparison form: case, articles and punctuation removed."""
    text = ARTICLE_RX.sub("", str(value).strip().strip(".,;:!?\"'"))
    return re.sub(r"\s+", " ", text).casefold().strip()


def scan(query: str, terms: Iterable[Tuple[str, str]]) -> List[Dict[str, object]]:
    """Find ``(surface, canonical)`` terms in ``query``, longest surface first.

    Longest-first with span masking is what stops ``Aetna`` from matching inside
    ``Aetna of PA`` and ``UPMC`` from hijacking ``UPMC Health Plan``.
    """
    haystack = query.casefold()
    found: List[Dict[str, object]] = []
    taken: List[Tuple[int, int]] = []

    for surface, canonical in sorted(terms, key=lambda pair: -len(pair[0])):
        needle = surface.casefold()
        if not needle:
            continue
        start = 0
        while True:
            index = haystack.find(needle, start)
            if index < 0:
                break
            end = index + len(needle)
            overlaps = any(index < stop and start_ < end for start_, stop in taken)
            if not overlaps and query[index:end]:
                taken.append((index, end))
                found.append({"text": query[index:end], "span": [index, end],
                              "canonical": canonical})
            start = index + 1

    return sorted(found, key=lambda item: item["span"][0])


def vocabularies(db: Path) -> Tuple[List[str], List[str]]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        hospitals = [r[0] for r in con.execute(
            "SELECT DISTINCT hospital FROM code_hospital WHERE hospital IS NOT NULL"
        ).fetchall()]
        payers = [r[0] for r in con.execute(
            "SELECT DISTINCT payer FROM combos WHERE payer IS NOT NULL"
        ).fetchall()]
    finally:
        con.close()
    return sorted(hospitals), sorted(payers)


def medical_terms(category: str) -> List[Tuple[str, str]]:
    cat = CATEGORIES.get(category)
    if cat is None:
        return []
    surfaces = [cat.lay, cat.clinical, *cat.aliases]
    return [(surface, surface) for surface in surfaces if surface]


def descriptor_match(query: str, codes: Sequence[str], family: str,
                     catalog: Dict[str, Dict[str, str]]) -> Optional[Dict[str, object]]:
    """A B3 clinical question quotes the catalog descriptor verbatim — find it."""
    haystack = query.casefold()
    texts = catalog.get(family, {})
    best: Optional[Dict[str, object]] = None
    for code in codes:
        text = texts.get(str(code))
        if not text or len(text) < 12:
            continue
        index = haystack.find(text.casefold())
        if index < 0:
            continue
        if best is None or len(text) > len(str(best["text"])):
            best = {"text": query[index:index + len(text)], "span": [index, index + len(text)],
                    "canonical": text}
    return best


def build(dataset_dir: Path, db: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    cases = [json.loads(line) for line in
             (dataset_dir / "pt_eval_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = {record["eval_id"]: record for record in
            (json.loads(line) for line in
             (dataset_dir / "pt_eval_v2_meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}

    hospitals, payers = vocabularies(db)
    catalog = load_catalog()

    payer_terms: List[Tuple[str, str]] = [(alias, canonical)
                                         for canonical, aliases in PAYER_ALIASES.items()
                                         for alias in aliases]
    payer_terms += [(payer, payer) for payer in payers]
    hospital_terms = [(hospital, hospital) for hospital in hospitals]

    targets: List[Dict[str, object]] = []
    audit = {"medical_withheld": 0, "no_hospital_but_named": 0, "hospital_not_in_gt": 0,
             "payer_not_in_gt": 0, "gt_hospital_unnamed": 0, "gt_payer_unnamed": 0,
             "multi_hospital": 0}

    for case in cases:
        eval_id = str(case["eval_id"])
        subtype = eval_id.rsplit("_", 1)[-1]
        record = meta.get(eval_id, {})
        category = str(record.get("category", ""))
        query = str(case["eval_query"])

        entity_map = {
            "medical": scan(query, medical_terms(category)),
            "hospital": scan(query, hospital_terms),
            "insurer": scan(query, payer_terms),
            "medication": scan(query, [("insulin", "insulin")]),
        }

        if not entity_map["medical"]:
            family = CATEGORIES[category].family if category in CATEGORIES else "CPT"
            fallback = descriptor_match(query, case.get("ground_truth_code_MRF", []), family, catalog)
            invented = record.get("invented_service")
            if fallback is not None:
                entity_map["medical"] = [fallback]
            elif invented:
                index = query.casefold().find(str(invented).casefold())
                text = query[index:index + len(str(invented))] if index >= 0 else invented
                entity_map["medical"] = [{"text": text,
                                          "span": [index, index + len(str(invented))] if index >= 0 else [-1, -1],
                                          "canonical": str(invented)}]
            else:
                audit["medical_withheld"] += 1

        gt_hospital = case.get("ground_truth_hospital_name")
        gt_payer = case.get("ground_truth_insurance_provider")
        named_hospitals = [item["canonical"] for item in entity_map["hospital"]]
        named_payers = [item["canonical"] for item in entity_map["insurer"]]

        if len(named_hospitals) > 1:
            audit["multi_hospital"] += 1
        if gt_hospital and gt_hospital not in named_hospitals:
            audit["hospital_not_in_gt"] += 1
        if gt_hospital is None and named_hospitals:
            audit["gt_hospital_unnamed"] += 1
        if not gt_hospital and named_hospitals:
            audit["no_hospital_but_named"] += 1
        if gt_payer and gt_payer not in named_payers:
            audit["payer_not_in_gt"] += 1
        if gt_payer is None and named_payers:
            audit["gt_payer_unnamed"] += 1

        targets.append({
            "eval_id": eval_id,
            "subtype": subtype,
            "intent": INTENT_BY_SUBTYPE.get(subtype[:2], INTENT_DEFAULT),
            "use_codes": "none" if subtype == "D2" else (
                CATEGORIES[category].family.lower() if category in CATEGORIES else "cpt"),
            "n_rows_gt": len(case.get("ground_truth_row_ids") or []),
            "entities": {field: entity_map[field] for field in FIELD_ORDER},
            "canonical": {field: [normalize(str(item["canonical"])) for item in entity_map[field]]
                          for field in FIELD_ORDER},
        })

    return targets, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval_dataset")
    parser.add_argument("--db", default=".cache/mrf_eval.duckdb")
    parser.add_argument("--out", default="eval_dataset/pt_eval_v2_targets.jsonl")
    args = parser.parse_args()

    targets, audit = build(Path(args.dataset), Path(args.db))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in targets:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_intent: Dict[str, int] = {}
    by_medication = 0
    for record in targets:
        by_intent[str(record["intent"])] = by_intent.get(str(record["intent"]), 0) + 1
        by_medication += 1 if record["canonical"]["medication"] else 0

    print(f"wrote {out} — {len(targets)} targets")
    print(f"intent: {by_intent}")
    print(f"with a medication entity: {by_medication}")
    print(f"audit: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
