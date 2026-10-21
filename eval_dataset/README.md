# `eval_dataset` — price-transparency evaluation set (v2)

600 question/answer pairs asked against the published UPMC machine-readable file (MRF).
Built so that every value in an answer key is a **lookup**, never a model's guess: the
codes come from the two catalog CSVs the MedTE embedder was trained on, and the row ids
are re-derived from the exact hospital / payer / plan that the sentence names.

- `pt_eval_v2.jsonl` — 600 cases, one JSON object per line
- `pt_eval_v2_meta.jsonl` — the same 600 ids, with the provenance behind each key
  (rate, cash price, C1 ranking, persona, which field was withheld, …)

Generator: [`../eval_code_gen`](../eval_code_gen). Rebuilding from seed 13 reproduces both
files byte-for-byte.

## Schema

Keys are emitted in this order, always:

| key | type | meaning |
|---|---|---|
| `eval_id` | string | `[CPT/HCPCS code]_[6-digit sequence]_[subtype]`, e.g. `45378_000013_A1` |
| `eval_query` | string | the patient's question, in their own words |
| `ground_truth_codes` | list[str] | every code in the CPT/HCPCS category the question is about |
| `ground_truth_code_MRF` | list[str] | the subset of those codes that appears in this MRF |
| `ground_truth_row_ids` | list[str] | MRF `row_id`s matching exactly the entities the question names |
| `ground_truth_hospital_name` | string or null | null when the question does not name a hospital |
| `ground_truth_insurance_provider` | string or null | null when the question does not name a payer |

`ground_truth_row_ids` is a list because one question legitimately matches several rows:
the file prices inpatient and outpatient separately, carries several plan names under one
payer, and lists several negotiated rates for the same plan. The key is the **complete**
match set for the stated entities — nothing truncated, nothing invented. `n_rows_total` in
the sidecar records the full count, so a truncated key would be detectable.

`ground_truth_codes` vs `ground_truth_code_MRF` is a real distinction, not a filter for
tidiness: the colonoscopy category has 31 curated codes and 28 of them are in this file
(`44397`, `44406`, `44407` are not). Two categories are HCPCS (`J1815` insulin); the rest
are CPT.

## The ten prompt subtypes — 60 each

| set | subtype | what it asks | what makes it hard |
|---|---|---|---|
| 1 | `A1` | precise service, hospital, payer, plan, and either the exact CPT or the service name | both renderings of the same question must land on the same rows |
| 1 | `B1` | one of hospital / payer / plan / service / care setting is withheld (12 each) | the answer key must drop what the sentence drops |
| 1 | `B2` | payer named by an alias (`UHC`, `Highmark`, `UPMC`) | alias must resolve to the MRF's `payer_name` |
| 1 | `B3` | lay vs clinical terminology for the same category (30 each) | vocabulary, not the code, carries the question |
| 2 | `A2` | ten personas (self-insured 20-something, 62-year-old on Medicare Advantage, caregiver calling for a parent, …) | persona changes register, not the ground truth |
| 2 | `A3` | ten demographic profiles (age, county, plan type) plus a named hospital | the profile must not leak a payer the sentence never states |
| 3 | `C1` | which of two hospitals is cheaper for one payer, and by how much | ranking across hospitals; key spans both hospitals |
| 3 | `C2` | cash price vs negotiated rate at one hospital | the two price columns are different facts |
| D | `D1` | a real code at a real hospital that has **no row** | must abstain; absence verified against the whole file |
| D | `D2` | an invented service that has no accepted code | must abstain and not invent a code |

Coverage: 12 categories — colonoscopy 71, insulin 86, hip replacement 54, chest x-ray 50,
CT abdomen 50, EGD 47, knee arthroscopy 46, mammogram 45, CMP 44, MRI brain 41, knee
replacement 30, physical therapy 36. Codes per case run 0 (the two abstention sets) to 31.
1,700 row ids in total: 192 cases key exactly one row, 288 key more than one (up to 48),
120 key none.

## Sidecar fields

`subtype`, `set_id`, `category`, `pivot_code`, `pivot_code_source`, `plan`, `setting`,
`n_rows_total`, `rate`, `cash`, `negotiated`, `gross`, `methodology`, `descriptor`,
`withheld_field`, `missing_field`, `variant`, `alias_used`, `persona`, `persona_label`,
`profile`, `profile_context`, `profile_framing`, `invented_service`, `abstention`,
`code_hospitals_with_rows`, `cheapest`, `cheapest_rate`, `priciest`, `priciest_rate`,
`difference`, `ranking`, `generator_version`.

C1's `ranking` holds every hospital with a priced row for that payer and code, cheapest
first, which is more than the question asks for but makes the comparison auditable.

## Using it for scoring

- **Retrieval**: score `ground_truth_code_MRF` as the relevant set; `ground_truth_codes`
  is the full universe and includes codes this file does not price.
- **Row-level grounding**: score `ground_truth_row_ids` as a set (precision/recall). Do
  not expect a single id — see above.
- **Abstention**: `D1` and `D2` are the only cases where an empty row list is correct;
  they are the only place a "no price found" answer should be credited.
- **Numeric faithfulness**: `rate`, `cash` and C1's `cheapest_rate` / `priciest_rate` are
  read straight out of the CSV, so any `$` figure in an answer must match one of them.

## What is not in this repo

The MRF itself (~2.4 GB) and the catalog CSVs. Both are inputs to
[`../eval_code_gen`](../eval_code_gen/README.md), which documents where they go.

Prices are the hospitals' own published figures, extracted verbatim. They are not quotes,
not coverage guarantees, and will drift as hospitals republish.
