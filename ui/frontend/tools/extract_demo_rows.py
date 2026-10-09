import csv, json, sys
csv.field_size_limit(10**9)
F="/home/adi274903/.hermes/attachments/everyUPMCmrf_clean.csv"

WANT = {
  "mammo": {"cpt":"77065", "hospital":"presbyterian"},
  "colon": {"cpt":"45378", "payer":"highmark"},
}
COLS = ["row_id","hospital_name","payer_name","plan_name","description","billing_class","setting",
        "CPT","HCPCS","modifiers","standard_charge|gross","standard_charge|discounted_cash",
        "standard_charge|negotiated_dollar","standard_charge|negotiated_percentage",
        "standard_charge|negotiated_algorithm","median_amount","10th_percentile","90th_percentile",
        "count","standard_charge|min","standard_charge|max","standard_charge|methodology",
        "additional_generic_notes","charge_type","data_quality"]

hits={"mammo":[], "colon":[]}
n=0
with open(F, newline="", encoding="utf-8", errors="replace") as f:
    r=csv.DictReader(f)
    idx={c:i for i,c in enumerate(r.fieldnames)}
    avail=[c for c in COLS if c in idx]
    for row in r:
        n+=1
        if row["CPT"].strip()=="77065" and "presbyterian" in (row["hospital_name"] or "").lower():
            hits["mammo"].append({c:row[c] for c in avail})
        elif row["CPT"].strip()=="45378" and "highmark" in (row["payer_name"] or "").lower():
            hits["colon"].append({c:row[c] for c in avail})
        if n % 2000000 == 0: print(f"  {n:,}", file=sys.stderr, flush=True)

print("rows scanned:", n, file=sys.stderr)
for k,v in hits.items():
    print(f"{k}: {len(v)} rows", file=sys.stderr)
out="/home/adi274903/.hermes/cache/scratch/demo_rows.json"
with open(out,"w") as o: json.dump({"columns":avail,"mammo":hits["mammo"],"colon":hits["colon"]}, o, indent=1)
print("wrote", out, file=sys.stderr)
