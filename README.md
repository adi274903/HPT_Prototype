# Healthcare Price Transparency (HPT) — Prototype

Production-shaped Python refactor of the Colab prototype notebook
(`notebooks/HPT_prototype_original.ipynb`).

The prototype answers patient questions about **what a medical service might
cost**, by combining:

1. **Entity extraction** — MedGemma classifies a free-text query into
   `medical` / `hospital` / `insurer` / `medication` entities.
2. **Semantic code retrieval** — a MedTE sentence embedder searches two Qdrant
   collections (`cpt_medte`, `hcpcs_medte`) for candidate **CPT** / **HCPCS** codes.
3. **Code decision** — MedGemma picks a small, plausible subset of the retrieved codes.
4. **Validation** — selected codes are hard-checked against the retrieved
   candidate set (hallucinated codes are rejected).
5. **Price lookup** — the validated codes, payer and hospital filter a cleaned
   hospital **MRF** (machine-readable file) DataFrame.
6. **Answer generation** — MedGemma writes a patient-friendly, non-committal answer
   grounded in the matched price rows.

## Repository layout

```
PT_Healthcare/
├── pt_healthcare/            # the package (importable, no notebook state)
│   ├── config.py             # model ids, collection names, paths, TOP_K
│   ├── models.py             # load_models() -> (MedGemma pipe, MedTE embedder)
│   ├── data.py               # HF login, Qdrant client, MRF CSV, archive extract
│   ├── utils.py              # list/string normalization helpers
│   ├── parsing.py            # robust JSON extraction from LLM output
│   ├── prompts.py            # prompt builders (pure, testable)
│   ├── llm.py                # MedGemmaEngine: pipe -> categorize/decide/answer
│   ├── retrieval.py          # CodeRetriever: embed + Qdrant search
│   ├── pricing.py            # sql_answer(): filter the MRF DataFrame
│   ├── pipeline.py           # HealthcarePricingPipeline: the 7-step orchestration
│   └── cli.py                # command-line entry point
├── colab_run.py              # Google Colab runner: pip/drive/tar glue + CLI
├── scripts/download_db.sh    # fetch + extract the Qdrant snapshot
├── examples/run_pipeline.py  # minimal end-to-end example
├── tests/                    # pytest suite for every pure component
└── notebooks/                # original prototype + Colab driver notebook
```

### How the notebook maps to the package

| Notebook cell(s)                 | Package module                |
| -------------------------------- | ----------------------------- |
| `pip install ...`                | `requirements.txt` / `colab_run.install_dependencies()` |
| imports / `drive.mount` / HF login | `data.py` + `colab_run.py` |
| `load_models()`                  | `models.py`                   |
| DB download + `!cp` + `!tar -xzf` | `colab_run.stage_db()` + `scripts/download_db.sh` |
| `QdrantClient`, collection stats | `data.py`                     |
| `pd.read_csv(mrf...)`            | `data.py` (`load_mrf_data`)   |
| `TOP_K = 10`                     | `config.py`                   |
| `retrieve_codes`                 | `retrieval.py` (`CodeRetriever.retrieve_codes`) |
| `_clean_list`, `build_medical_query` | `utils.py` / `retrieval.py` |
| `retrieve_cpt`, `retrieve_hcpcs` | `retrieval.py`                |
| `categorizer_prompt`             | `prompts.py` + `llm.py`       |
| `compact_candidates`             | `parsing.py`                  |
| `decision_prompt`                | `prompts.py` + `llm.py`       |
| `answer_user`                    | `prompts.py` + `llm.py`       |
| `parse_json_output`              | `parsing.py`                  |
| `_as_clean_list`, `_normalize_series`, `_contains_any`, `sql_answer` | `utils.py` / `pricing.py` |
| `orchestration`                  | `pipeline.py`                 |
| `orchestration(user_query)` calls | `cli.py` / `examples/run_pipeline.py` |

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended (MedGemma 1.5-4B, bfloat16).
- Data you must supply yourself (not in this repo):
  - **MedGemma weights** (`google/medgemma-1.5-4b-it`) and the MedTE embedder
    (`MohammadKhodadad/MedTE-cl15-step-8000`) — both gated on Hugging Face;
    set `HUGGINGFACE_API_KEY` in `.env`.
  - A **Qdrant snapshot** with `cpt_medte` / `hcpcs_medte` collections
    (see `scripts/download_db.sh`).
  - A cleaned **hospital MRF CSV** (the prototype used
    `everyUPMCmrf_clean (1).csv`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # optional: installs the `pt-healthcare` command
```

## Configure

```bash
cp .env.example .env        # then edit
```

| Variable              | Default                                | Meaning                          |
| --------------------- | -------------------------------------- | -------------------------------- |
| `HUGGINGFACE_API_KEY` | —                                      | HF token for gated model access  |
| `PT_MRF_CSV`          | `everyUPMCmrf_clean.csv`               | cleaned MRF CSV path             |
| `PT_QDRANT_PATH`      | `New_PT_DB`                            | local Qdrant store directory     |
| `PT_TOP_K`            | `10`                                   | candidates retrieved per code set |

## Run in Google Colab

All the Colab-specific glue (the `!pip install` cell, `drive.mount`, and the
`!cp` + `!tar -xzf` snapshot staging) lives in **`colab_run.py`** — one
self-contained entry point you can call from a Colab cell.

Open `notebooks/HPT_colab.ipynb` (Runtime → GPU), or do it manually:

```python
# 1. get the code (equivalent to the notebook's !pip install cell)
!git clone https://github.com/adi274903/HPT_Prototype /content/HPT_Prototype
%cd /content/HPT_Prototype
!pip install -q -r requirements.txt

# 2. mount Drive + stage the Qdrant snapshot (drive.mount + cp + tar -xzf)
import colab_run
colab_run.mount_drive()
colab_run.stage_db()               # -> /content/New_PT_DB

# 3. load models + data
pipeline = colab_run.setup(mount=False, stage=False)

# 4. ask
result = pipeline.run(
    "What might a diagnostic mammogram cost at UPMC Presbyterian "
    "with UPMC Health Plan?"
)
print(result["answer"])
```

One-liner instead of steps 2–3:

```python
import colab_run
result = colab_run.run("What might a diagnostic mammogram cost at UPMC Presbyterian?")
```

From a shell cell, this runs the notebook's three validation queries end-to-end:

```bash
!python colab_run.py
```

`colab_run.py` uses the same Drive paths the prototype did
(`MyDrive/New_PT_DB_Backups/New_PT_DB.tar.gz` and
`MyDrive/Price_Transparency/datasets/everyUPMCmrf_clean (1).csv`) and resolves
the HF token from the environment, a `.env`, or Colab secrets
(`HUGGINGFACE_API_KEY`). Every path can be overridden — see
`python colab_run.py --help`.

## Run

```bash
# after `pip install -e .`
pt-healthcare --query "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?"

# or without installing
python -m pt_healthcare --query "..." --mrf-csv /path/to/mrf.csv --qdrant-path /path/to/New_PT_DB

# interactive REPL
python -m pt_healthcare --interactive
```

`--json` emits the full result dict as JSON; `--quiet` suppresses step logging.

## Tests

The pure components (parsing, utils, pricing, retrieval plumbing, prompt
building, pipeline validation) are covered without needing a GPU or the data:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes / caveats carried over from the prototype

- The retrieved codes are **candidates only** — never billing advice, coverage
  guarantees or price quotes.
- MedGemma emits a `<unused94>thought ... <unused95>answer` wrapper; the final
  user-facing text is the segment after `<unused95>`. `parse_json_output`
  strips `<unusedNN>` control tokens before parsing.
- `sql_answer` treats CPT and HCPCS as an **OR** within the code filter and
  **AND**s the payer / hospital substring filters on top.
- `TOP_K` candidates are retrieved, but the decision prompt sends at most 8 per
  code family (`compact_candidates`), and the model may select at most 5 each.
