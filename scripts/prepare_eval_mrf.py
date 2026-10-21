#!/usr/bin/env python3
"""Materialise the eval sampling universe from everyUPMCmrf_clean.csv into duckdb.

    ~/.hermes/bin/uv run --with duckdb python3 scripts/prepare_eval_mrf.py \
        --mrf ~/Downloads/everyUPMCmrf_clean.csv --out .cache/mrf_eval.duckdb

Creates two tables in --out (rebuilt from scratch each run):

  rows_ext  one row per MRF row, only the columns the eval needs, prices cast to DOUBLE
  combos    one row per (code, family, hospital, payer, plan, setting, rate). The MRF
            contains rows that describe the same price twice — e.g. rows 4822417 and
            4833810 for 45378 / Upmc Magee / UPMC Health Plan / Commercial are both
            1433.69, but only 4822417 carries gross and cash. Combos keeps the more
            complete row. `setting` is part of the key because inpatient/outpatient
            twins (71046 rows 1081022/1081023) are legitimately distinct rows.

`family` labels which MRF column the code came from; a row with both columns set
appears twice, once per family, because a CPT query and a HCPCS query are different
lookups even on the same row.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

COLUMNS = """
    row_id,
    hospital_name                                                        AS hospital,
    payer_name                                                           AS payer,
    plan_name                                                            AS plan,
    setting,
    description,
    nullif(trim("CPT"), '')                                              AS cpt,
    nullif(trim("HCPCS"), '')                                            AS hcpcs,
    TRY_CAST("standard_charge|gross" AS DOUBLE)                          AS gross,
    TRY_CAST("standard_charge|discounted_cash" AS DOUBLE)                AS cash,
    TRY_CAST("standard_charge|negotiated_dollar" AS DOUBLE)              AS rate,
    nullif(trim("standard_charge|methodology"), '')                      AS methodology
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrf", default=str(Path.home() / "Downloads" / "everyUPMCmrf_clean.csv"))
    ap.add_argument("--out", default=".cache/mrf_eval.duckdb")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if not Path(args.mrf).is_file():
        return print(f"missing MRF: {args.mrf}") or 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = duckdb.connect(str(out))
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute("PRAGMA memory_limit='6GB'")

    src = (
        f"read_csv_auto('{args.mrf}', header=true, all_varchar=true, sample_size=200000)"
    )

    t0 = time.perf_counter()
    con.execute(
        f"""CREATE TABLE rows_ext AS
            SELECT {COLUMNS}
            FROM {src}
            WHERE billing_class = 'facility'"""  # professional: 1,868 of 10.3M rows
    )
    n_rows = con.execute("SELECT count(*) FROM rows_ext").fetchone()[0]
    print(f"rows_ext  {n_rows:,} rows  {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    con.execute(
        """
        CREATE TABLE combos AS
        WITH scored AS (
            SELECT *,
                   (CASE WHEN gross       IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN cash        IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN methodology IS NOT NULL THEN 1 ELSE 0 END) AS completeness,
                   row_number() OVER (
                       PARTITION BY cpt, hcpcs, hospital, payer, plan, setting, rate
                       ORDER BY (CASE WHEN gross       IS NOT NULL THEN 1 ELSE 0 END
                               + CASE WHEN cash        IS NOT NULL THEN 1 ELSE 0 END
                               + CASE WHEN methodology IS NOT NULL THEN 1 ELSE 0 END) DESC,
                                TRY_CAST(row_id AS BIGINT)
                   ) AS rn
            FROM rows_ext
            WHERE rate IS NOT NULL AND rate > 0
        ), kept AS (SELECT * FROM scored WHERE rn = 1)
        SELECT row_id, hospital, payer, plan, setting, rate, gross, cash, methodology,
               cpt AS code, 'CPT' AS family
        FROM kept WHERE cpt IS NOT NULL
        UNION ALL
        SELECT row_id, hospital, payer, plan, setting, rate, gross, cash, methodology,
               hcpcs AS code, 'HCPCS' AS family
        FROM kept WHERE hcpcs IS NOT NULL
        """
    )
    n_combos = con.execute("SELECT count(*) FROM combos").fetchone()[0]
    print(f"combos    {n_combos:,} rows  {time.perf_counter() - t0:.0f}s")

    for family in ("CPT", "HCPCS"):
        n_codes = con.execute(
            "SELECT count(DISTINCT code) FROM combos WHERE family = ?", [family]
        ).fetchone()[0]
        print(f"  {family}: {n_codes:,} distinct codes")

    # Every (code, hospital) pair that exists anywhere in the file, facility or
    # professional, rate or no rate. Abstention cases are checked against this: a
    # hospital that has even one professional-billing row for the code is not absent.
    t0 = time.perf_counter()
    con.execute(
        f"""
        CREATE TABLE code_hospital AS
        SELECT DISTINCT nullif(trim("CPT"), '') AS code, 'CPT' AS family,
                        hospital_name AS hospital
        FROM {src}
        WHERE nullif(trim("CPT"), '') IS NOT NULL AND hospital_name IS NOT NULL
        UNION
        SELECT DISTINCT nullif(trim("HCPCS"), '') AS code, 'HCPCS' AS family,
                        hospital_name AS hospital
        FROM {src}
        WHERE nullif(trim("HCPCS"), '') IS NOT NULL AND hospital_name IS NOT NULL
        """
    )
    n_cov = con.execute("SELECT count(*) FROM code_hospital").fetchone()[0]
    print(f"coverage  {n_cov:,} (code, hospital) pairs  {time.perf_counter() - t0:.0f}s")

    con.execute("CREATE INDEX combos_code ON combos (code, family)")
    con.execute("CREATE INDEX coverage_code ON code_hospital (code, family)")
    con.close()

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
