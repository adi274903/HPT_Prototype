# Healthcare Price Transparency (HPT) — Prototype

Production-shaped Python refactor of the Colab prototype notebook
(`notebooks/HPT_prototype_original.ipynb`).

The prototype answers patient questions about **what a medical service might
cost**, by combining:

1. **Entity extraction** — MedGemma classifies a free-text query into
   `medical` / `hospital` / `insurer` / `medication` entities. The hospital and
   payer names are chosen from the values that actually exist in the MRF (see
   [Closed vocabulary](#closed-vocabulary-for-the-categorizer)), not invented.
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
├── colab_run.py              # Colab runner: pip/drive/tar glue + UI serving + tunnel
├── ui/                       # TypeScript front end (2 pages) + pt_serve.py server
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

Values are read **when the pipeline is built**, not at import, so a `.env`
loaded late — the normal Colab case, where the notebook runs somewhere other
than this repo — still takes effect. The file itself is searched for in the
working directory, Colab's `/content`, and this repo's root (override with
`PT_ENV_FILE`). A real environment variable or Colab secret always beats the
file.

| Variable              | Default                                | Meaning                          |
| --------------------- | -------------------------------------- | -------------------------------- |
| `HUGGINGFACE_API_KEY` | —                                      | HF token for gated model access (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` also accepted) |
| `PT_MRF_CSV`          | `everyUPMCmrf_clean.csv`               | cleaned MRF CSV path             |
| `PT_QDRANT_PATH`      | `New_PT_DB`                            | local Qdrant store directory     |
| `PT_TOP_K`            | `10`                                   | candidates retrieved per code set |
| `PT_ENV_FILE`         | —                                      | explicit `.env` path             |

`PT_MEDGEMMA_MODEL`, `PT_EMBED_MODEL`, `PT_CPT_COLLECTION`,
`PT_HCPCS_COLLECTION`, `PT_MAX_CANDIDATES`, `PT_CANDIDATE_TEXT_LIMIT` and
`PT_MAX_NEW_TOKENS` are also honoured; see `.env.example`.

If you only use the module constants (`config.TOP_K`, `config.MRF_CSV_PATH`, …)
rather than calling the pipeline, call `config.refresh()` after your own
`load_dotenv()`.

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

Forgot step 1? `setup()` checks for the runtime packages and installs any that
are missing inside Colab (mirroring the notebook's first cell). Pass
`install=False` to forbid that, or `install=True` to force it; either way a
missing package raises a `RuntimeError` naming the packages and the command,
rather than a `ModuleNotFoundError` from deep inside the stack.

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

## Front end (TypeScript UI)

`ui/` holds the two front-end pages and the stdlib-only server that hosts them:
`pt_patient.html` (patient chat, every `$` figure cited to the MRF row behind
it), `pt_frontend.html` (the seven pipeline steps, candidates, validation and
price table), and `pt_serve.py`.

`colab_run.serve_ui()` wires the backend into them and, by default, publishes a
public Cloudflare quick tunnel (no account required):

```python
import colab_run

ui = colab_run.serve_ui(pipeline, page="patient", tunnel="cloudflare")
print(ui.pages)
# {'patient': 'https://<random>.trycloudflare.com/',
#  'dashboard': 'https://<random>.trycloudflare.com/pipeline'}

ui.stop()          # end the server and the tunnel
```

Or build the pipeline and serve in one call — `serve_ui()` accepts the same
kwargs as `setup()`:

```python
ui = colab_run.serve_ui()          # models + data + UI + tunnel
```

| Argument | Default | Notes |
| --- | --- | --- |
| `tunnel` | `cloudflare` | `cloudflare` (cloudflared quick tunnel, auto-downloaded), `ngrok` (needs `pip install pyngrok` + `NGROK_AUTHTOKEN`), `none` (localhost / Colab kernel-port iframe only) |
| `page` | `patient` | Which page the in-notebook iframe opens; both are always served |
| `port` | `8000` | Falls back through 8001 / 8080 / 8888 / a free port |
| `host` | `127.0.0.1` | `0.0.0.0` to also expose on the LAN |
| `logs` | `True` | Adds a live backend log at `/logs` (JSON polling, not SSE) |
| `log_to_notebook` | `True` | Echoes every pipeline log line into the cell output |
| `polling` | `True` | Adds `/api/run` + `/api/poll`, the transport the built client prefers |

### Transports, and why the client polls

The server exposes three ways to run a query. The built page tries them in order:

| Transport | Endpoints | Why |
| --- | --- | --- |
| polling (default) | `POST /api/run` → `GET /api/poll?run=…&since=…` | Ordinary request/response pairs. Nothing to strand. |
| streaming | `GET /api/stream?q=…` | Original SSE path; kept for older servers. |
| blocking | `POST /api/query` | Last resort for a plain static host. |

The reason for the ordering: `pt_serve._sse_open` sends `connection: close` with
no `Transfer-Encoding: chunked`, so the SSE response's end is defined by
connection close — and a proxy is entitled to buffer the whole thing and release
it at the end. A Cloudflare quick tunnel does exactly that, which is why the page
used to sit on its static "Reading the published price file…" bubble forever
while the backend ran fine.

Polling is additive: `install_routes()` subclasses the vendored handler, so
`ui/pt_serve.py` is still untouched, and `/api/stream` and `/api/query` keep
working unchanged for anything already pointed at them. The polling endpoint
projects the run through the same `pt_serve.to_payload`, so the client consumes
one payload contract regardless of transport.

Verified against the front-end's own suite (259 checks) with both the polling
server and its mock backend.

### Closed vocabulary for the categorizer

Step 1 is open-ended NER, which is why it misbehaves on real queries: asked
"what hospital should I go to for my colonoscopy if I have my *Higmark BCBS*
plan?", it spends a paragraph deciding whether to emit the literal string
`"hospital"` as a placeholder, and it returns the user's typo (`"Higmark BCBS
plan"`) verbatim. That name then matches nothing downstream, because the price
filter is a case-insensitive *substring* match against `payer_name`.

Two bugs in the categorizer prompt caused most of this, and both were inherited
from the prototype:

- a rule that *asked* for it — `Include generic references such as "my hospital"
  and "my insurance" if present` — so `"hospital"` came back as an entity;
- an example, `Query: How much would Houston Hospital cost me for colonsocopy?`
  → `"hospital": ["Houston Hospital"]`, which teaches that a token ending in
  "Hospital" is a facility. "What hospital should I go to" pattern-matches
  straight into it.

Both are now replaced with the opposite guidance and a worked example showing
`"hospital": []` for a query that asks *which* hospital but names none.

Because a 4B model will still normalise loosely, there are three guards:

1. **The vocabulary goes into the prompt** — the distinct `hospital_name` /
   `payer_name` values from the MRF (`pipeline.hospital_names()` /
   `.payer_names()`, computed once and cached), so extraction is slot-filling
   instead of free invention.
2. **`drop_placeholders`** removes bare category words outright. `"hospital"` is
   not a name, and as a substring match it silently selects an arbitrary slice of
   the file or nothing at all.
3. **`resolve_to_vocabulary`** snaps near misses onto real entries — exact match,
   then unique substring either way (`"highmark"` → `"Highmark BCBS of PA"`,
   `"UPMC Presbyterian"` → `"Upmc Presbyterian Shadyside"`), then one clear fuzzy
   winner. Genuinely ambiguous terms are left alone: guessing between two UPMC
   payers is worse than reporting what the model produced.

Every correction is logged, so you can see it happen:

```
Insurers resolved to price-file entries: ['highmark plan'] -> ['Highmark BCBS of PA']
Hospitals resolved to price-file entries: ['hospital'] -> []
```

The lists are bounded (`vocabulary.DEFAULT_LIMIT`, 300) and cost little prompt
space; with no MRF loaded the block is omitted and the prompt is byte-identical
to the prototype's. Same shape as the code side: the decision step has always
chosen CPT/HCPCS codes from the retrieved candidates rather than inventing them.

```
Known entries in the published price file. When the query refers to one of
these, return the exact string from the list below — matching loosely, since a
misspelling, an abbreviation or a partial name still refers to an entry. Do not
invent a name, and never return the list itself: only what the query actually
refers to.

hospitals (12):
- Upmc Altoona
- Upmc Bedford
...
insurers (9):
- Aetna
- Highmark BCBS of PA
...
```

### Watch what the backend is doing

`/logs` on the same URL is a live backend log — the STEP lines, timings, matched
row counts and rejections, straight from the pipeline.

```python
ui = colab_run.serve_ui()
print(ui.pages)                     # ...plus: Logs: <url>/logs
```

It matters because the patient page's progress is **cosmetic until real frames
arrive**: the bubble always says "Reading the published price file…", and the
seven step circles only tick when a `stage` frame lands. A proxy that buffers SSE
(Cloudflare quick tunnels do) leaves that bubble static forever while the backend
runs fine. `/logs` polls over ordinary HTTP, so it keeps working — and
`log_to_notebook=True` echoes the same lines into the notebook cell so you can
watch a run without opening a browser at all.

Both are added by subclassing the vendored server's handler — `ui/pt_serve.py`
itself is untouched.

From the CLI:

```bash
python colab_run.py --serve-ui                      # tunnel, blocks (Ctrl-C stops)
python colab_run.py --serve-ui --tunnel none        # local only
python colab_run.py --serve-ui --ui-page dashboard  # open the pipeline view
```

**Security.** A tunnel URL is public and *unauthenticated*: anyone holding it
can run queries against the pipeline and consume the GPU. Prefer `tunnel="none"`
unless you actually need remote access, and call `ui.stop()` when you are done.
There is no built-in auth — adding one means patching `ui/pt_serve.py`, which is
deliberately byte-identical to its source of truth (see `ui/README.md`).

`pt_serve.py` calls `pipeline.run` directly and parses the pipeline's own log
lines, so the UI reports the real backend's behaviour rather than a
reimplementation of it. The front-end sources and its own test suite live in
`~/projects/reseearch_work/Price_Transparency/frontend`.

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
