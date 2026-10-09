# Price Transparency — front end

Two TypeScript pages for the CMU–NIST AIMSEC hospital price-transparency agent
(`Copy_of_Try_PT-3.ipynb`), fed by the same payload, written for two different
readers.

| Page | Who it's for | What it is |
| --- | --- | --- |
| **`dist/pt_patient.html`** | A patient | A chat. Ask what a procedure costs; every dollar figure is linked to the published record behind it. |
| **`dist/pt_frontend.html`** | A reviewer / engineer | The seven pipeline steps — retrieval, code validation, the full MRF price table, timings, raw model output. |

Each is one self-contained HTML file with a light/dark theme and the CMU + NIST
marks in the header. Both run live against the notebook kernel, and both fall
back to a bundled sample with no backend at all.

```bash
npm install
npm run build      # -> dist/pt_patient.html, dist/pt_frontend.html
npm run verify     # 212 checks: typecheck + 3 DOM suites + API + browser e2e
npm run serve      # local preview on :8788 with a mock backend
```

---

## The patient page

The design brief was the opposite of a logging dashboard. A patient does not want
candidate rankings, cosine scores, `use_codes`, or a log stream. They want one
number and a reason to believe it — so the internals are hidden and the
*evidence* is surfaced.

**Figures carry citations.** Every `$` figure in the model's answer is matched
against the values actually present in the matched MRF rows. A figure that
matches gets a superscript marker pointing at the record; clicking it opens the
source list and highlights that record. Matching is exact to the cent — no fuzzy
comparison, because a near-miss on a price is worse than no citation at all.

**Figures that match nothing are flagged, not hidden.** If the model produces a
number no published row corroborates, it renders with a dotted underline, an
explicit "not in the published file" label, and a warning above the sources
naming the untraced values. That is the most useful thing the page does: it turns
a trust problem into a visible signal.

**Every source is a real record.** Hospital, payer, plan, code, descriptor, the
price and *which kind* of price it is (negotiated rate / cash price / list price),
plus how the hospital says it charges. A disclosure exposes the underlying row —
record id, methodology, billing class, setting — so anyone can check it.

**Prices are never shown without their methodology.** Per-visit, per-unit and
case rates for the same code differ by roughly 500% on real MRFs. When the
hospitals in a comparison charge differently, the page says so rather than
letting a lower number look like cheaper care.

**Progress is described in human terms.** "Reading the hospital's published price
file", not `STEP 6/7 — Filtering UPMC MRF data`.

**There is an About panel, not another page.** It sits beside "How it works" in
the top bar and opens in place, so a patient who opens it mid-question keeps
their conversation (the tests assert the thread and composer nodes are never
re-created). It covers who built the tool, where the prices come from — the
federal machine-readable-file rule, UPMC's file — the five steps an answer takes,
and, at the same length as everything else, what it cannot do: not a bill or
coverage advice, one health system, does not know your plan, rates only
comparable when the method matches, not medical advice. It deliberately contains
no `.amount` elements: it describes the untraced-figure treatment without
demonstrating it, so nothing scanning the DOM can mistake a sample for a real
figure.

---

## Running it in Colab

Nothing to install — the server is Python standard library only.

1. Upload **`dist/pt_patient.html`**, **`dist/pt_frontend.html`** and
   **`colab/pt_serve.py`** into `/content`.

2. Add a cell **after** the pipeline is defined. Cells 6, 9, 11 and 21 of the
   notebook define `mrf_data`, `TOP_K`, `client` and `orchestration`:

```python
import sys
sys.path.insert(0, "/content")
import pt_serve

handle = pt_serve.launch(
    orchestration=orchestration,   # used exactly as the notebook defines it
    mrf_data=mrf_data,             # -> the "MRF rows" figure in the status strip
    client=client,                 # -> the CPT/HCPCS index counts
    top_k=TOP_K,
    page="patient",                # or "dashboard"
)
```

`/` serves the patient chat, `/pipeline` the dashboard, and each links to the
other. `page=` only chooses which one the iframe opens. Re-running the cell is
safe — the previous server is shut down first. `handle.stop()` ends it.

Other entry points:

```python
# Blocking call, no UI — returns the same payload the pages consume
payload = pt_serve.to_payload(query=q, result=orchestration(q))

# Static snapshot of a finished run, embedded in the notebook output
pt_serve.render(payload)                     # patient chat
pt_serve.render(payload, page="dashboard")   # pipeline view
```

If the iframe does not appear, `launch()` falls back to a plain iframe and then
to a printed URL. The client resolves Colab's `/proxy/<port>/` prefix on its own,
so no extra configuration is needed.

---

## How it talks to the notebook

**Transport.** The client tries three in order: `POST /api/run` +
`GET /api/poll?run=…&since=…` (polling, the default), then `GET
/api/stream?q=…` (SSE), then `POST /api/query` (blocking). Polling leads because
SSE here is a single long-lived response whose end is defined by connection close
— `_sse_open` sends no `Transfer-Encoding: chunked` — and a buffering proxy
(Cloudflare quick tunnels) will hold every frame until the run ends, leaving the
page blank while the backend works. A 404/405/501 from `/api/run` means the
server predates it, and the client falls back to the stream; the mock backend
used by this repo's own suite exercises exactly that path.

`orchestration()` is called **unmodified**. There is no fork of the pipeline logic
to drift out of sync. Instead the server tee's the run's `stdout` and reads the
pipeline's own log:

| Notebook output | Becomes |
| --- | --- |
| `STEP 4/7 — Asking decision model…` | step 4 marked *running* |
| `Decision completed in 86.56s` | step 4 *done*, with its timing |
| `MRF rows matched: 157` | row-count detail on step 6 |
| `Rejected CPT codes not found…: [...]` | the struck-through rows in step 5 |
| `Raw categorizer output:` + `repr(...)` | the raw-output disclosures |
| anything else | streamed into the "Kernel output" panel |

The same parsing powers two transports feeding one JSON envelope:

- **`GET /api/stream?q=…`** — SSE, what the UI uses. Step events arrive while the
  model is still thinking.
- **`POST /api/query`** — one blocking JSON response.

`GET /api/health` reports MRF row count, Qdrant point counts, `top_k` and the
available pages. The UI probes it on load: reachable → **Live kernel**;
unreachable → **Offline sample**.

---

## Layout

```
src/
  app.ts               dashboard: state, render strategy, event wiring
  api.ts               transports (SSE + POST), Colab /proxy/ base resolution
  theme.ts             light/dark, persisted, follows the OS until you choose
  types.ts             the wire contract with the Python side
  format.ts            currency/number/markdown helpers, thought-trace splitting
  state.ts             dashboard state + step metadata
  ui/shell.ts          header (logos, theme toggle), hero, status strip, footer
  ui/steps.ts          dashboard steps 1–5
  ui/price.ts          dashboard step 6: sortable MRF table, CSV export
  ui/answer.ts         dashboard step 7: answer, grounding table, disclosures
  styles.css           dashboard design

  patient/
    main.ts            chat state, send/answer flow, citation navigation
    cite.ts            THE GROUNDING LOGIC: figure -> published row matching
    render.ts          bubbles, source records, comparison, progress, notices
    styles.css         patient design (standalone; shares nothing but the idea)

  demo.json            offline sample payload — generated, see below
assets/                CMU + NIST logo files, inlined as data URIs at build time
colab/pt_serve.py      the server (stdlib only); serves both pages
build.mjs              esbuild -> two self-contained HTML pages
devserver.mjs          local preview (--mock serves the API from the sample data)
tools/                 demo generator + four verification suites
```

### Build output

| File | What it is |
| --- | --- |
| `dist/pt_patient.html` | The patient chat. Zero external references. |
| `dist/pt_frontend.html` | The pipeline dashboard. Zero external references. |

Zero external references is what makes them work over the Colab proxy, from a
static server, and straight off the filesystem.

### Build flags

```bash
node build.mjs --page patient        # one page only
node build.mjs --embed-demo 2        # bake a demo scenario in as the payload
node build.mjs --out preview.html    # one-off artifact under another name
node build.mjs --no-minify           # readable bundle, for debugging
```

---

## Design decisions worth knowing

**Vanilla TypeScript, no framework.** Each page has to be one self-contained file
that boots in a Colab iframe with no network. A framework would add a runtime and
a second bundle for no benefit — each page is a handful of pure functions from
state to HTML strings.

**Log-parsing over monkey-patching.** Wrapping or editing the notebook's
functions would fork the pipeline and rot. Reading its stdout keeps the UI honest
about what the pipeline actually did, including failures.

**The two pages share no CSS.** They are written for different readers, and a
shared stylesheet would couple a consumer surface to an engineering one. They
duplicate about forty lines of colour tokens and nothing else.

**The dashboard never claims a run it did not make.** In live mode it starts
empty rather than showing the bundled sample, so a sample can never be mistaken
for this kernel's output.

**Sorting never puts "unknown" first.** Rows with no negotiated rate sink to the
bottom in both directions, so an ascending sort never implies a missing rate is
the cheapest.

**`<unused94>` control tokens are stripped** from everything either page displays,
including the raw-output disclosures.

---

## The offline sample

`src/demo.json` is generated, not hand-written:

```bash
python3 tools/gen_demo.py
```

Price rows are real rows from the local UPMC MRF snapshot — the mammogram rows are
matched by the exact `row_id`s the notebook printed. Candidates, decisions, timings
and answers are transcribed verbatim from the notebook run. Three scenarios: a
single-payer price, one hospital across ten payers, and a 157-row cross-hospital
comparison.

Regenerating needs `demo_rows.json`, produced by `tools/extract_demo_rows.py`
(a single ~10M-row streaming pass over the 2.4 GB CSV).

---

## Verification

```bash
npm run verify
```

Four suites, 257 assertions, no network or GPU required:

| Suite | Covers |
| --- | --- |
| `tools/verify_dom.mjs` | Dashboard in jsdom: branding, all 7 cards, both themes, sorting, the sample scenarios, live-kernel detection. |
| `tools/verify_patient.mjs` | Patient page in jsdom: grounded answers, citation↔source consistency, the unsourced-figure path, methodology warnings, the evidence pane and its tabs, the About panel, the live SSE run, and the no-backend path. |
| `tools/verify_colab.py` | The real server against a notebook-shaped stub: every endpoint and alias, the SSE frame sequence, payload projection, rejection/raw-trace recovery, the error path, both pages standalone. |
| `tools/verify_e2e.mjs` | The built dashboard in a browser against the live server over real HTTP + SSE: clicks Run, watches the rail advance, checks the rendered table, CSV export, theme toggle. |

Bugs these caught that would otherwise have shipped:

- **Validation timings silently dropped.** The pipeline logs `Validated in 0.00s`,
  not `… completed in …`, so a regex requiring the word `completed` lost step 5.
- **Every successful run marked "failed".** `EventSource` fires `error` when the
  server closes the stream; a trailing error after the result flipped the status.
  That would have hit every run in Colab.
- **All four logos stacked in light mode.** `.logos img` out-specified
  `.logo-on-dark`, so the hidden variant still rendered.
- **The theme toggle reverted** wherever `localStorage` is unavailable, because
  the render path re-derived the theme instead of using the chosen one.
- **Citations did not highlight their record** — `scrollIntoView` is missing in
  some environments and threw inside the click handler.
- **The dashboard header stayed white in dark mode.** `[data-theme="dark"] header.top`
  and `:root header.top` have identical specificity — (0,2,1) vs (0,2,1) — so source
  order decided the winner and the light value always won. It is now one rule
  deriving from `--surface`. Note that jsdom *cannot* catch this class of bug: it
  does not evaluate `color-mix()`, so `getComputedStyle` reports `rgba(0,0,0,0)` for
  the header in both themes. The guard is therefore a check on the stylesheet source
  — no `:root <descendant>` selectors anywhere, and the header must take its
  background from a theme token.
- **The About panel shipped a decoy figure.** It illustrated the untraced-figure
  treatment using a real `<span class="amount unsourced">`, which corrupted the
  document-wide unsourced-figure counts on every page load — the page would report
  a phantom untraced amount that no answer produced. It now uses its own
  `.about-flag` class and the suite asserts the panel contains no `.amount` at all.
- **"N published records" did not take you to the records.** The jump button set
  the focused record but left the pane on whichever tab was active, so it could
  highlight a record nobody could see. It now switches to the Sources tab.

---

## Limits

- One query runs at a time — `PipelineRunner` serialises them, since the pipeline
  owns the GPU.
- The dashboard caps the price table at 1,000 rows and the log at 500 lines; CSV
  export contains the full sorted table.
- Citations match on exact value. If two procedures happen to publish the same
  price, a figure can cite both — the source list shows which record is which.
- Heavy loads (a first MedGemma download, a long colonoscopy run) still take
  minutes; the UI streams progress, but it cannot make the model faster.
