"""Spot-check the generated eval against the raw MRF, independently of the derived tables."""

import json
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent / "eval_dataset"
cases = [json.loads(l) for l in (BASE / "pt_eval_v2.jsonl").read_text().splitlines()]
meta = {m["eval_id"]: m for m in
        (json.loads(l) for l in (BASE / "pt_eval_v2_meta.jsonl").read_text().splitlines())}

con = duckdb.connect()
con.execute("PRAGMA threads=4")
src = ("read_csv_auto('/home/adi274903/Downloads/everyUPMCmrf_clean.csv', header=true,"
       " all_varchar=true, sample_size=200000)")
con.execute(f"""CREATE VIEW raw AS
    SELECT row_id, nullif(trim("CPT"),'') cpt, nullif(trim("HCPCS"),'') hcpcs,
           hospital_name, payer_name, plan_name, setting,
           TRY_CAST("standard_charge|negotiated_dollar" AS DOUBLE) rate,
           TRY_CAST("standard_charge|discounted_cash" AS DOUBLE) cash
    FROM {src}""")

a1 = next(c for c in cases if c["eval_id"].endswith("_A1") and c["ground_truth_row_ids"])
rid = a1["ground_truth_row_ids"][0]
print("A1", a1["eval_id"], "|", a1["eval_query"][:130])
print("   raw :", con.execute(
    "SELECT row_id,cpt,hcpcs,hospital_name,payer_name,plan_name,setting,rate FROM raw WHERE row_id=?",
    [rid]).fetchall())
print("   meta:", {k: meta[a1["eval_id"]][k] for k in ("rate", "gross", "cash", "setting", "n_rows_total")})

c1 = next(c for c in cases if c["eval_id"].endswith("_C1"))
m = meta[c1["eval_id"]]
code = c1["eval_id"].split("_")[0]
print("\nC1", c1["eval_id"], "|", c1["eval_query"])
print("   meta:", m["cheapest"], m["cheapest_rate"], "->", m["priciest"], m["priciest_rate"],
      "delta", m["difference"], "n=", m["n_hospitals"])
rows = con.execute("""SELECT hospital_name, min(rate) FROM raw
                      WHERE (cpt=? OR hcpcs=?) AND payer_name=? AND rate>0
                      GROUP BY 1 ORDER BY 2""",
                   [code, code, c1["ground_truth_insurance_provider"]]).fetchall()
print("   raw :", rows[:2], "...", rows[-2:], f"({len(rows)} hospitals)")
print("   keyed rows:", c1["ground_truth_row_ids"])

c2 = next(c for c in cases if c["eval_id"].endswith("_C2"))
m2 = meta[c2["eval_id"]]
print("\nC2", c2["eval_id"], "|", c2["eval_query"])
print("   raw :", con.execute(
    "SELECT row_id,hospital_name,payer_name,plan_name,rate,cash FROM raw WHERE row_id=?",
    [c2["ground_truth_row_ids"][0]]).fetchall())
print("   meta cash/neg:", m2["cash"], "/", m2["negotiated"], "| rows:", c2["ground_truth_row_ids"])
