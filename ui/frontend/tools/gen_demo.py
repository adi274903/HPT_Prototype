#!/usr/bin/env python3
"""
Generate src/demo.ts (the offline sample payload) from REAL data.

- Price rows come from the local UPMC MRF snapshot
  (/home/adi274903/.hermes/attachments/everyUPMCmrf_clean.csv, extracted by
  tools/extract_demo_rows.py into demo_rows.json).
- Candidates, decisions, timings and answers are transcribed verbatim from the
  notebook run (Copy_of_Try_PT-3.ipynb) that produced them.

Run from frontend/:  python3 tools/gen_demo.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ROWS_JSON = pathlib.Path("/home/adi274903/.hermes/cache/scratch/demo_rows.json")

SUMMARY_COLUMNS = [
    "row_id", "hospital_name", "payer_name", "plan_name", "description",
    "billing_class", "setting", "CPT", "HCPCS", "modifiers",
    "standard_charge|gross", "standard_charge|discounted_cash",
    "standard_charge|negotiated_dollar", "median_amount", "10th_percentile",
    "90th_percentile", "standard_charge|min", "standard_charge|max",
    "standard_charge|methodology",
]

NUMERIC = {
    "standard_charge|gross", "standard_charge|discounted_cash",
    "standard_charge|negotiated_dollar", "standard_charge|negotiated_percentage",
    "median_amount", "10th_percentile", "90th_percentile", "count",
    "standard_charge|min", "standard_charge|max",
}

STAGE_TITLES = [
    "Entity extraction",
    "CPT retrieval",
    "HCPCS retrieval",
    "Code decision",
    "Code validation",
    "MRF price filter",
    "Final answer",
]
STAGE_KEYS = ["entities", "cpt", "hcpcs", "decision", "validation", "mrf", "answer"]


def rows_from(rows) -> list[dict]:
    """Trim to the summary columns and coerce numerics/empties the way pandas would."""
    out = []
    for r in rows:
        rec = {}
        for c in SUMMARY_COLUMNS:
            v = r.get(c, "")
            if c in NUMERIC:
                try:
                    rec[c] = float(v) if str(v).strip() != "" else None
                except ValueError:
                    rec[c] = None
            else:
                s = (v or "").strip()
                rec[c] = s if s and s.lower() not in {"nan", "none", "null"} else None
        out.append(rec)
    # match sql_answer(): sort by hospital, payer, plan, CPT, HCPCS, negotiated
    out.sort(key=lambda r: tuple(str(r.get(k) or "") for k in
                                 ("hospital_name", "payer_name", "plan_name", "CPT", "HCPCS"))
             + ((r.get("standard_charge|negotiated_dollar") if
                 r.get("standard_charge|negotiated_dollar") is not None else float("inf")),))
    return out


def stages(times: list[float], details: list[str]) -> list[dict]:
    return [
        {"n": i + 1, "key": STAGE_KEYS[i], "title": STAGE_TITLES[i],
         "state": "done", "seconds": times[i], "detail": details[i]}
        for i in range(7)
    ]


def scenario(*, query, entities, cpt, hcpcs, decision, rejected,
             price_rows, match_count, answer, raw_cat, raw_dec,
             times, details, created):
    return {
        "query": query,
        "mode": "demo",
        "categorized": entities,
        "cpt_candidates": cpt,
        "hcpcs_candidates": hcpcs,
        "decision": decision,
        "rejected": rejected,
        "code_plausible": {
            "match_count": match_count,
            "filters_used": {
                "cpt_codes": decision["cpt_list"],
                "hcpcs_codes": decision["hcpcs_list"],
                "insurers": [s.lower() for s in entities["insurer"]],
                "hospitals": [s.lower() for s in entities["hospital"]],
            },
            "price_summary": rows_from(price_rows),
            "matched_rows_total": match_count,
        },
        "answer": answer,
        "raw": {"categorization": raw_cat, "decision": raw_dec},
        "stages": stages(times, details),
        "total_seconds": round(sum(times), 2),
        "created_at": created,
    }


def cand(rows):
    return [{"rank": i + 1, "score": s, "code": c, "text": t}
            for i, (s, c, t) in enumerate(rows)]


# ── Retrieval results (identical in every mammogram run: same medical entity) ──
MAMMO_CPT = cand([
    (0.896429, "76090", "Mammography; unilateral"),
    (0.866981, "76091", "Mammography; bilateral"),
    (0.860501, "77065", "Diagnostic mammography, unilateral, including CAD"),
    (0.848398, "77066", "Diagnostic mammography, bilateral, including CAD"),
    (0.824852, "99459", "Pelvic examination (clinical breast examination)"),
    (0.786874, "77055", "Mammography, unilateral (including CAD), with or without 3D imaging"),
    (0.781509, "77061", "Mammography, digital breast tomosynthesis, unilateral (including CAD)"),
    (0.778584, "77056", "Mammography, bilateral (including CAD), with or without 3D imaging"),
    (0.778563, "77067", "Screening mammography, bilateral, including CAD"),
    (0.762974, "77063", "Screening mammography, digital breast tomosynthesis, bilateral"),
])
MAMMO_HCPCS = cand([
    (0.799653, "G0204", "Diagnostic mammography, including computer-aided detection (cad) when performed; bilateral"),
    (0.796630, "G0206", "Diagnostic mammography, including computer-aided detection (cad) when performed; unilateral"),
    (0.751306, "GH", "Diagnostic mammogram converted from screening mammogram on same day"),
    (0.725492, "G0202", "Screening mammography, bilateral (2-view study of each breast), including computer-aided detection"),
    (0.700840, "S8080", "Scintimammography (radioimmunoscintigraphy of the breast), unilateral, including supply of radiopharmaceutical"),
    (0.698585, "C8903", "Magnetic resonance imaging with contrast, breast; unilateral"),
    (0.687986, "C8905", "Magnetic resonance imaging without contrast followed by with contrast, breast; unilateral"),
    (0.668098, "C8906", "Magnetic resonance imaging with contrast, breast; bilateral"),
    (0.663040, "C8904", "Magnetic resonance imaging without contrast, breast; unilateral"),
    (0.648542, "G9899", "Screening, diagnostic, film, digital or digital breast tomosynthesis mammography reported without a final assessment"),
])

COLON_CPT = cand([
    (0.895359, "45378", "Colonoscopy, flexible; diagnostic"),
    (0.880258, "45325", "Colonoscopy, rigid or flexible; diagnostic"),
    (0.872600, "45365", "Colonoscopy, fiberoptic; with biopsy, single or multiple"),
    (0.864252, "44407", "Colonoscopy with needle aspiration/biopsy"),
    (0.845305, "44406", "Colonoscopy with endoscopic ultrasound"),
    (0.844383, "44405", "Colonoscopy with dilation"),
    (0.835768, "45360", "Colonoscopy, fiberoptic; diagnostic, including collection of specimen(s) by brushing or washing"),
    (0.835572, "74263", "Computed tomography, colonography (virtual colonoscopy), screening, without contrast material"),
    (0.834518, "44404", "Colonoscopy with injection"),
    (0.834234, "74261", "Computed tomography, colonography (virtual colonoscopy), diagnostic, without contrast material"),
])
COLON_HCPCS = cand([
    (0.974941, "G9937", "Diagnostic colonoscopy"),
    (0.807058, "G0122", "Colorectal cancer screening; barium enema"),
    (0.782134, "S0601", "Screening proctoscopy"),
    (0.738899, "G0104", "Colorectal cancer screening; flexible sigmoidoscopy"),
    (0.728116, "PT", "Colorectal cancer screening test; converted to diagnostic test or other procedure"),
    (0.723015, "G9936", "Surveillance colonoscopy - personal history of colonic polyps"),
    (0.690434, "S0285", "Colonoscopy consultation performed prior to a screening or diagnostic colonoscopy"),
    (0.670335, "G9933", "Adenoma(s) or colorectal cancer detected during screening colonoscopy"),
    (0.669762, "G9252", "Adenoma(s) or other neoplasm detected during screening colonoscopy"),
    (0.662690, "G9253", "Adenoma(s) or other neoplasm not detected during screening colonoscopy"),
])

RAW_CAT_MAMMO = (
    "<unused94>thought\n"
    "The user wants me to extract medical, hospital, and insurer entities from the query "
    "\"What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?\".\n\n"
    "1.  **Identify medical entities:**\n    *   \"diagnostic mammogram\" is a medical procedure/test.\n"
    "2.  **Identify hospital entities:**\n    *   \"UPMC Presbyterian\" is a hospital name.\n"
    "3.  **Identify insurer entities:**\n    *   \"UPMC Health Plan\" is an insurance plan/payer.\n"
    "4.  **Identify medication entities:**\n    *   There are no medications mentioned.\n"
    "<unused95>```json\n{\n  \"medical\": [\"diagnostic mammogram\"],\n"
    "  \"hospital\": [\"UPMC Presbyterian\"],\n  \"insurer\": [\"UPMC Health Plan\"],\n"
    "  \"medication\" : []\n}\n```"
)

RAW_DEC_MAMMO = (
    "<unused94>thought\nThe user wants me to select billing codes for a \"diagnostic mammogram\" procedure.\n"
    "I need to choose codes from the provided lists (CPT and HCPCS) and return them in a specific JSON format.\n\n"
    "1.  **Analyze the procedure:** \"diagnostic mammogram\". This is a specific type of mammogram performed "
    "for diagnostic purposes, not screening.\n"
    "2.  **Review CPT candidates:**\n"
    "    *   `76090`: Mammography; unilateral - This is a good candidate for a diagnostic mammogram.\n"
    "    *   `76091`: Mammography; bilateral - This is also a good candidate for a diagnostic mammogram.\n"
    "    *   `77065`: Diagnostic mammography, unilateral, including CAD - This is a very specific and likely candidate.\n"
    "    *   `77066`: Diagnostic mammography, bilateral, including CAD - This is also a very specific and likely candidate.\n"
    "    *   `77061`: Mammography, digital breast tomosynthesis, unilateral (including CAD) - This is a specific "
    "type of diagnostic mammography.\n"
    "    *   `99459`: Pelvic examination (clinical breast examination) - This is not a mammogram.\n"
    "    *   `G0204` / `G0206` / `GH`: These are HCPCS codes, not CPT.\n"
    "... [reasoning truncated — full trace in the notebook]\n"
    "<unused95>```json\n{\n  \"use_codes\": \"cpt\",\n  \"cpt_list\": [\"77065\", \"77066\", \"76090\", \"76091\", \"77061\"],\n"
    "  \"hcpcs_list\": []\n}\n```"
)

RAW_CAT_COLON = (
    "```json\n{\n  \"medical\": [\"colonoscopy\"],\n  \"hospital\": [],\n"
    "  \"insurer\": [\"Highmark BCBS\"],\n  \"medication\" : []\n}\n```"
)

RAW_DEC_COLON = (
    "<unused94>thought\nThe user wants me to select billing codes based on the provided procedure \"colonoscopy\".\n"
    "I need to choose CPT and HCPCS codes from the given lists.\n"
    "1.  `45378`: Colonoscopy, flexible; diagnostic (0.8954) - This is a primary diagnostic colonoscopy.\n"
    "2.  `45325`: Colonoscopy, rigid or flexible; diagnostic (0.8803) - Similar to 45378.\n"
    "3.  `45365`: Colonoscopy, fiberoptic; with biopsy (0.8726) - Diagnostic colonoscopy *with* biopsy.\n"
    "4.  `74263`: Computed tomography, colonography (virtual colonoscopy) - a different procedure.\n"
    "5.  `G9937`: Diagnostic colonoscopy (0.9749) - specific HCPCS code for diagnostic colonoscopy.\n"
    "    *   `G0122`, `S0601`: different screening methods.\n"
    "... [reasoning truncated — full trace in the notebook]\n"
    "<unused95>```json\n{\n  \"use_codes\": \"cpt\",\n  \"cpt_list\": [\"45378\"],\n"
    "  \"hcpcs_list\": [\"G9937\"]\n}\n```"
)

ANSWER_MAMMO_INSURED = (
    "Based on the information provided, a diagnostic mammogram at UPMC Presbyterian with UPMC Health Plan "
    "might cost approximately $543.00 out of pocket.\n\n"
    "This cost is based on the standard charge for the procedure, which is $905.00. "
    "The discounted cash price is $543.00.\n\n"
    "It's important to note that the final cost can vary depending on your specific situation and the services "
    "actually performed. For the most accurate information, please contact UPMC Presbyterian's billing office "
    "or your UPMC Health Plan representative. They can provide details about your specific coverage and any "
    "potential additional charges."
)

ANSWER_MAMMO_OPEN = (
    "Based on the provided data, a diagnostic mammogram (CPT code 77065) at UPMC Presbyterian Shadyside "
    "might cost:\n\n"
    "* **Gross Charge:** $905.00\n"
    "* **Discounted Cash Price:** $543.00\n"
    "* **Negotiated Dollar Price:** $364.84\n"
    "* **Minimum Charge:** $58.27\n"
    "* **Maximum Charge:** $769.25\n\n"
    "The final cost can vary depending on your specific insurance plan, your demographics, and the exact "
    "services performed. For the most accurate out-of-pocket cost, please contact UPMC Presbyterian "
    "Shadyside's billing office or your insurance company."
)

ANSWER_COLON = (
    "Okay, I can help you look at the information available regarding colonoscopy procedures with "
    "Highmark BCBS.\n\n"
    "The CPT code for colonoscopy is **45378** and the HCPCS code is **G9937**.\n\n"
    "The data shows that the procedure is typically covered by Highmark BCBS of PA for many hospitals listed.\n\n"
    "To find the \"best\" and \"cheapest\" hospital, we need to look at the cost information provided. "
    "\"Best\" often depends on factors like location, quality of care, and patient reviews, which aren't "
    "included here. \"Cheapest\" can mean different things, so let's look at a few cost metrics:\n\n"
    "* **Standard Charge (Gross):** This is the base price before any discounts. The lowest recorded "
    "standard charge is around $180.50 and the highest is around $3124.67.\n"
    "* **Discounted Cash:** This is the price after a cash discount is applied. The lowest discounted "
    "cash is around $1500.00 and the highest is around $3124.67.\n"
    "* **Negotiated Dollar:** This is the price after a negotiated discount with the insurance company. "
    "The lowest negotiated dollar is around $729.80 and the highest is around $3124.67.\n"
    "* **Median Amount:** This represents the typical cost for a procedure at that hospital. The median "
    "amount is around $207.57.\n"
    "* **Percentiles:** The 10th percentile cost is around $180.50, and the 90th percentile cost is "
    "around $207.57.\n\n"
    "Based on these metrics, the hospital with the lowest standard charge, discounted cash, and "
    "negotiated dollar is generally considered the cheapest. The hospital with the lowest median "
    "amount is often considered the typical cost.\n\n"
    "**Recommendation:**\n\n"
    "To find the best and cheapest hospital *for you*, you should:\n\n"
    "1.  **Check the specific hospital's billing office:** They can provide the most current pricing "
    "and confirm coverage with Highmark BCBS for your specific plan.\n"
    "2.  **Consider location and quality:** Think about which hospital is most convenient and meets "
    "your quality needs.\n"
    "3.  **Compare costs:** Look at the different cost metrics (standard charge, discounted cash, "
    "negotiated dollar, median amount) to see which one is most important to you.\n\n"
    "Remember, the final cost depends on the specific services performed and the documentation "
    "provided by the doctor. Please confirm the exact cost and coverage details with the hospital's "
    "billing department and your Highmark BCBS insurance company before scheduling the procedure."
)


def main() -> int:
    data = json.loads(ROWS_JSON.read_text())
    mammo_ids = {"6297848", "6317794", "6357686", "6377632", "6397674",
                 "6554625", "6564506", "6670378", "6772230", "6823708"}
    mammo_rows = [r for r in data["mammo"] if r["row_id"] in mammo_ids]
    if len(mammo_rows) != 10:
        print(f"FATAL: expected 10 mammogram rows, found {len(mammo_rows)}", file=sys.stderr)
        return 1

    upmc_hp = [r for r in mammo_rows if r["payer_name"] == "UPMC Health Plan"]
    colon_rows = data["colon"]
    if not colon_rows:
        print("FATAL: no colonoscopy rows", file=sys.stderr)
        return 1

    scenarios = [
        scenario(
            query="What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?",
            entities={"medical": ["diagnostic mammogram"], "hospital": ["UPMC Presbyterian"],
                      "insurer": ["UPMC Health Plan"], "medication": []},
            cpt=MAMMO_CPT, hcpcs=MAMMO_HCPCS,
            decision={"use_codes": "cpt",
                      "cpt_list": ["77065", "77066", "76090", "76091", "77061"], "hcpcs_list": []},
            rejected={"cpt": [], "hcpcs": []},
            price_rows=upmc_hp, match_count=len(upmc_hp),
            answer=ANSWER_MAMMO_INSURED, raw_cat=RAW_CAT_MAMMO, raw_dec=RAW_DEC_MAMMO,
            times=[12.41, 0.06, 0.05, 86.56, 0.00, 18.21, 8.68],
            details=["4 entities", "10 candidates", "10 candidates", "5 CPT / 0 HCPCS selected",
                     "5 kept, 0 rejected", "1 row matched", "answer generated"],
            created="2026-10-08T22:10:00-04:00",
        ),
        scenario(
            query="What might a diagnostic mammogram cost at UPMC Presbyterian",
            entities={"medical": ["diagnostic mammogram"], "hospital": ["UPMC Presbyterian"],
                      "insurer": [], "medication": []},
            cpt=MAMMO_CPT, hcpcs=MAMMO_HCPCS,
            decision={"use_codes": "cpt",
                      "cpt_list": ["77065", "77066", "76090", "76091", "77061"], "hcpcs_list": []},
            rejected={"cpt": [], "hcpcs": []},
            price_rows=mammo_rows, match_count=len(mammo_rows),
            answer=ANSWER_MAMMO_OPEN, raw_cat=RAW_CAT_MAMMO, raw_dec=RAW_DEC_MAMMO,
            times=[14.06, 0.06, 0.05, 86.84, 0.00, 10.47, 76.14],
            details=["4 entities", "10 candidates", "10 candidates", "5 CPT / 0 HCPCS selected",
                     "5 kept, 0 rejected", "10 rows matched", "answer generated"],
            created="2026-10-08T22:14:00-04:00",
        ),
        scenario(
            query="Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?",
            entities={"medical": ["colonoscopy"], "hospital": [], "insurer": ["Highmark BCBS"],
                      "medication": []},
            cpt=COLON_CPT, hcpcs=COLON_HCPCS,
            decision={"use_codes": "cpt", "cpt_list": ["45378"], "hcpcs_list": ["G9937"]},
            rejected={"cpt": ["45325", "45365", "74263"], "hcpcs": ["G0122", "S0601"]},
            price_rows=colon_rows, match_count=len(colon_rows),
            answer=ANSWER_COLON, raw_cat=RAW_CAT_COLON, raw_dec=RAW_DEC_COLON,
            times=[2.67, 0.06, 0.05, 77.97, 0.00, 15.50, 113.91],
            details=["4 entities", "10 candidates", "10 candidates", "1 CPT / 1 HCPCS selected",
                     "2 kept, 5 rejected", f"{len(colon_rows)} rows matched", "answer generated"],
            created="2026-10-08T22:20:00-04:00",
        ),
    ]

    body = json.dumps(scenarios, ensure_ascii=False, indent=1)
    json_dest = ROOT / "src" / "demo.json"
    json_dest.write_text(body, encoding="utf-8")
    print(f"wrote {json_dest} ({json_dest.stat().st_size:,} bytes)")

    header = (
        "/* AUTO-GENERATED by tools/gen_demo.py — do not edit by hand.\n"
        " * Offline sample payload: price rows are real rows from the local UPMC MRF\n"
        " * snapshot; candidates, decisions, timings and answers are transcribed from\n"
        " * the notebook run of Copy_of_Try_PT-3.ipynb. */\n\n"
        "import data from \"./demo.json\";\n"
        "import type { PipelineResult } from \"./types\";\n\n"
        "export const DEMO_SCENARIOS = data as unknown as PipelineResult[];\n"
    )
    dest = ROOT / "src" / "demo.ts"
    dest.write_text(header, encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")
    for s in scenarios:
        print(f"  {s['code_plausible']['match_count']:>4} rows | "
              f"{s['total_seconds']:>6.2f}s | {s['query'][:58]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
