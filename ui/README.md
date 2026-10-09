# Front end (vendored)

Two TypeScript pages for the price-transparency agent, plus the stdlib-only
server that hosts them, served against this repo's backend by
`colab_run.serve_ui()`.

| File | What it is |
| --- | --- |
| `pt_patient.html` | The patient-facing page — a chat where every `$` figure is cited back to the published MRF row behind it, and figures nothing corroborates are flagged rather than hidden. |
| `pt_frontend.html` | The reviewer/engineer page — the seven pipeline steps, retrieval candidates, code validation, the full price table, timings and raw model output. |
| `pt_serve.py` | Python standard library only. A threaded HTTP + SSE server: it calls the backend's orchestration callable, tee's its stdout, and parses the pipeline's own log lines into progress events. |

Both pages are self-contained single files (CSS, JS and the CMU/NIST marks are
inlined); there are no external requests, so a tunnel serves everything.

## Why `pt_serve.py` is unmodified

The server deliberately never reimplements the pipeline. It calls
`orchestration(query)` exactly as the notebook did and reads the pipeline's own
stdout (`STEP 4/7 — …`, `Decision completed in 86.56s`, `MRF rows matched: N`,
`Rejected CPT codes not found in retrieval results: […]`, `Raw categorizer
output:`). `pt_healthcare.pipeline` preserves those strings verbatim, so the UI
cannot drift from the backend's behaviour.

This copy is **byte-identical** to the source of truth

```
~/projects/reseearch_work/Price_Transparency/frontend/colab/pt_serve.py
```

(verify: `sha256sum ui/pt_serve.py`). Two consequences:

- The front-end repo's own test suite still covers exactly this file.
- If you change `pt_serve.py` here, you have forked it — port the change back.

## Rebuilding the pages

The pages are build artifacts, not sources. Sources live in the front-end repo:

```bash
cd ~/projects/reseearch_work/Price_Transparency/frontend
npm install
npm run build      # -> dist/pt_patient.html, dist/pt_frontend.html
npm run verify     # typecheck + DOM suites + API + e2e
```

Then copy back:

```bash
cp dist/pt_patient.html dist/pt_frontend.html \
   ~/projects/workspace2/project/PT_Healthcare/ui/
```

`pt_serve.py` resolves pages relative to itself, but `colab_run.serve_ui()`
passes both paths explicitly, so the flat layout here is what gets served.

## Serving

```python
import colab_run
ui = colab_run.serve_ui(pipeline, page="patient", tunnel="cloudflare")
print(ui.pages)   # {'patient': 'https://<id>.trycloudflare.com/', 'dashboard': '.../pipeline'}
ui.stop()
```

`/` is the patient chat, `/pipeline` the dashboard; each links to the other.

### Backend log pane (`/logs`)

`serve_ui(logs=True)` — the default — adds a third route: a live backend log at
`/logs`, backed by `/api/log?since=N`, a plain JSON polling endpoint.

It exists because the patient page's progress display is *cosmetic* until real
frames arrive: `botBody()` always renders the literal string "Reading the
published price file…", and the seven step circles only tick when a `stage`
frame lands. Behind a proxy that buffers SSE (Cloudflare's quick tunnels do,
since `_sse_open` sends `connection: close` with no `Transfer-Encoding:
chunked`), the page sits on that static bubble indefinitely while the backend is
working fine. `/logs` polls over ordinary HTTP, so it shows what is actually
running regardless.

The same lines are echoed into the notebook cell output by default
(`log_to_notebook=True`), which is the quickest way to watch a run.

This is added **without modifying `pt_serve.py`**: `pt_serve.serve()` resolves its
module-global `_Handler` at call time, so `colab_run.install_log_routes()`
subclasses it and swaps it in. The vendored file stays byte-identical (see
`sha256sum ui/pt_serve.py`). The log lines come from the pipeline's own
`printer` hook, which sees every line before pt_serve's stdout tee does.

```python
ui = colab_run.serve_ui(logs=True, log_to_notebook=True)   # defaults
ui = colab_run.serve_ui(logs=False)                        # no /logs, no notebook echo
```

## Not present here

- `dist/pt_colab.py`, the single-file bundle that embedded both pages, was
  deliberately dropped from the front-end build — the server is what remains.
- No authentication. A tunnel URL is public: anyone holding it can run queries
  on the pipeline. `tunnel="none"` keeps it on the kernel/localhost instead.
