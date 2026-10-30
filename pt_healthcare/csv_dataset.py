#!/usr/bin/env python3
"""Turn a flat eval CSV into the dataset directory ``eval_runner.evaluate`` reads.

Input columns (case-insensitive): Query, CPT, Hospital, Payer, Plan, Negotiated_rate.

    python -m pt_healthcare.csv_dataset --csv A1_dataset.csv --db .cache/mrf_eval.duckdb --out /content/a1_eval

Writes ``pt_eval_v2.jsonl`` and ``pt_eval_v2_targets.jsonl`` in ``--out``. The CSV has no
row ids, so each one is looked up in the ``combos`` table that ``prepare_eval_mrf.py``
builds: the rows for that code, hospital, payer and plan whose rate equals
Negotiated_rate. A line that matches no row is skipped and listed, never guessed.

The CSV carries no span annotations, so ``canonical.medical`` is empty and the medical
NER score for this set is not meaningful; hospital and insurer scores are.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb

REQUIRED = ("query", "cpt", "hospital", "payer", "plan", "negotiated_rate")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
        missing = [name for name in REQUIRED if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} is missing columns {missing}; has {reader.fieldnames}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader if (row.get("query") or "").strip()]


def convert(csv_path: Any, db_path: Any, out_dir: Any, subtype: str = "A1") -> Dict[str, Any]:
    rows = read_rows(Path(csv_path))
    con = duckdb.connect(str(db_path), read_only=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for index, row in enumerate(rows, start=1):
        code, hospital, payer, plan = row["cpt"].upper(), row["hospital"], row["payer"], row["plan"]
        try:
            rate = float(row["negotiated_rate"].replace(",", "").replace("$", ""))
        except ValueError:
            skipped.append(f"line {index + 1}: bad Negotiated_rate {row['negotiated_rate']!r}")
            continue
        found = con.execute(
            """SELECT row_id, family FROM combos
               WHERE code = ? AND hospital = ? AND payer = ? AND plan = ?
                 AND abs(rate - ?) < 0.005
               ORDER BY try_cast(row_id AS BIGINT)""",
            [code, hospital, payer, plan, rate]).fetchall()
        if not found:
            skipped.append(f"line {index + 1}: no MRF row for {code} / {hospital} / {payer} / {plan} @ {rate}")
            continue
        family = found[0][1]
        eval_id = f"{code}_{index:06d}_{subtype}"
        cases.append({
            "eval_id": eval_id, "eval_query": row["query"],
            "ground_truth_codes": [code], "ground_truth_code_MRF": [code],
            "ground_truth_row_ids": [str(r[0]) for r in found],
            "ground_truth_hospital_name": hospital,
            "ground_truth_insurance_provider": payer,
        })
        targets.append({
            "eval_id": eval_id, "subtype": subtype, "intent": "price",
            "use_codes": family.lower(), "n_rows_gt": len(found),
            "entities": {"medical": [], "hospital": [{"text": hospital, "canonical": hospital}],
                         "insurer": [{"text": payer, "canonical": payer}], "medication": []},
            "canonical": {"medical": [], "hospital": [hospital.lower()],
                          "insurer": [payer.lower()], "medication": []},
        })
    con.close()

    for name, records in (("pt_eval_v2.jsonl", cases), ("pt_eval_v2_targets.jsonl", targets)):
        with (out / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"cases": len(cases), "skipped": skipped, "out": str(out)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--db", default=".cache/mrf_eval.duckdb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--subtype", default="A1")
    args = parser.parse_args(argv)
    result = convert(args.csv, args.db, args.out, args.subtype)
    print(f"{result['cases']} cases -> {result['out']}; skipped {len(result['skipped'])}")
    for line in result["skipped"]:
        print("  skipped", line)
    return 0 if result["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
