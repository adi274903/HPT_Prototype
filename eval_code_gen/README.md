# `eval_code_gen` — how `eval_dataset` was built

Generators and validators for [`../eval_dataset`](../eval_dataset/README.md). Nothing here
ships price data or the code catalogs; both are inputs you supply.

```
eval_code_gen/
├── prepare_eval_mrf.py   # MRF CSV  ->  .cache/mrf_eval.duckdb  (rows, combos, coverage)
├── categories.py         # curated CPT/HCPCS categories + catalog loader + --suggest
├── aliases.py            # payer aliases, personas, demographic profiles, invented services
├── build_eval_v2.py      # the ten-subtype generator (deterministic)
├── spotcheck_v2.py       # re-reads the raw CSV to confirm one rate and one ranking
└── tests/test_eval_v2.py # schema + provenance invariants (16 tests)
```

## Inputs you must supply

| input | where it goes | why |
|---|---|---|
| the cleaned UPMC MRF CSV (~2.4 GB) | anywhere; `--mrf` (default `~/Downloads/everyUPMCmrf_clean.csv`) | every code, row id and rate comes from here |
| CPT catalog, `code,text` | `data/codes/cpt.csv`, else `~/Downloads/cpt_embed_ready.csv` | the CPT vocabulary the embedder was trained on |
| HCPCS catalog, `code,text` | `data/codes/hcpcs.csv`, else `~/Downloads/hcpcs_embed_ready.csv` | same for HCPCS |

Both catalogs are read for **text only** (to word a question the way a patient or a
clinician would); they are never written into the output. Neither catalog, and no slice of
either, is committed to this repo — `*.csv` is git-ignored. Point `CATALOG_DIR` in
`categories.py` at your copies if they live elsewhere.

Python 3.10+, `duckdb`, `pandas`. The MRF is read with `read_csv_auto(..., all_varchar=true)`
so a stray type in one column cannot reshape the file.

## Run it

```bash
# 1. build the sampling universe (~30 s for 2.4 GB, 509 MB output)
python eval_code_gen/prepare_eval_mrf.py --mrf /path/everyUPMCmrf_clean.csv --out .cache/mrf_eval.duckdb

# 2. check the curated vocabularies still match the catalogs and the MRF
python eval_code_gen/categories.py --verify
python eval_code_gen/aliases.py --verify --db .cache/mrf_eval.duckdb

# 3. generate (from the repo root: defaults write into eval_dataset/)
python eval_code_gen/build_eval_v2.py --per-subtype 60 --seed 13

# 4. prove a sample against the raw CSV, then run the invariants
python eval_code_gen/spotcheck_v2.py
python -m pytest eval_code_gen/tests -q
```

`--per-subtype N` is the only knob that changes the size: 60 → 600 cases. Seed 13 is the
published one; any seed reproduces byte-for-byte on a rebuild.

## What the generator guarantees

- **Every answer key is a lookup.** Codes come from the catalogs, row ids and rates from
  the MRF. No question is written by a model, and no missing value is ever back-filled.
- **The key is derived from the rendered sentence**, not from generator intent
  (`gt_from_query`): if the wording dropped the payer, the key has no payer. A test asserts
  every hospital/payer in a key appears in its question.
- **Unanswerable beats wrong.** A unit that cannot be grounded is skipped and counted in
  the `skips` line rather than patched: no row for the pivot code, a duplicate question,
  a key over 50 rows, a C1 pair with no real spread.
- **Determinism.** Fixed seed, sorted row ordering in every query, ids assigned by a
  six-digit counter — two runs produce identical bytes.

## Invariants the tests enforce

Schema and key order, id format, 60 per subtype, unique queries, `ground_truth_code_MRF ⊆
ground_truth_codes`, every row id re-queried from the **raw CSV** and confirmed to carry the
pivot code and the named hospital/payer, D1 pairs absent from the *whole* file, D2 phrases
absent from every MRF description, the colonoscopy 31/28 split, C1 arithmetic, and a
deterministic rebuild.

## Pitfalls that shaped the code

- **Absence must be checked against the whole file, not the priced subset.** `combos` holds
  only rows with a positive negotiated rate; a hospital with a professional-billing or
  unpriced row would look absent. `code_hospital` exists for exactly this and is what D1
  consults.
- **Look codes up across both columns.** 29% of rows in the `HCPCS` column carry Level I CPT
  codes, so `74182` at one hospital exists only under `HCPCS`. Scoping lookup by "family"
  produced false abstention cases.
- **`setting` belongs in the dedupe key.** Collapsing duplicates without it merges the
  inpatient and outpatient twins into one row (e.g. `71046` rows `1081022`/`1081023`).
- **C1 needs distinct minima, not just two hospitals.** A payer with the same rate at eight
  hospitals is a vacuous ranking; a spread under 5% or under $5 is not a reasoning question.
- **Prose templates drop plan names containing `/`** — `EPO Partners/EPO Select Plan/HMO`
  reads as three separate plans inside a sentence.
- **Category discovery is hand-curated.** `COLONOSCOP` over the master CPT list returns 51
  hits including anesthesia and AI codes; the category is 31 curated codes, and the three
  that this file does not price are the reason `ground_truth_codes` and
  `ground_truth_code_MRF` differ.

## Adding a category

Add a `Category(...)` entry to `CATEGORIES` in `categories.py` with the lay phrase, the
clinical phrase and the curated code list, then `categories.py --verify` (every code must
exist in the catalog), and regenerate. Subtype balance is automatic: each generator draws
from every category.
