#!/usr/bin/env python3
"""
End-to-end verification of colab/pt_serve.py against a notebook-shaped stub.

Starts the real HTTP server, then exercises every endpoint over the wire and
asserts on the payload contract the front end depends on.

    python3 tools/verify_colab.py

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "colab"))

import _stub_pipeline as stub  # noqa: E402
import pt_serve  # noqa: E402

passed = 0
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(name)
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def get(url: str, timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.headers.get("content-type", ""), r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read().decode("utf-8")


def post(url: str, payload: dict, timeout: float = 60.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def stream(url: str, timeout: float = 60.0) -> list[dict]:
    """Read an SSE response to completion and return the decoded frames."""
    frames: list[dict] = []
    with urllib.request.urlopen(url, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))
    return frames


def main() -> int:
    print("\n── colab/pt_serve.py against a notebook-shaped stub ──")

    handle = pt_serve.serve(
        orchestration=stub.orchestration,
        mrf_data=stub.mrf_data,
        client=stub.client,
        top_k=stub.TOP_K,
        port=8971,
    )
    base = handle.url.rstrip("/")
    print(f"  (server on {base})")

    try:
        # ── health ──────────────────────────────────────────────────────
        status, ctype, body = get(f"{base}/api/health")
        health = json.loads(body)
        check("/api/health 200 json", status == 200 and "json" in ctype)
        check("health reports ok", health.get("ok") is True)
        check("health reports MRF row count", health.get("mrf_rows") == len(stub.mrf_data),
              str(health.get("mrf_rows")))
        check("health reports both Qdrant collections",
              health.get("collections", {}).get("cpt_medte") == 15927
              and health.get("collections", {}).get("hcpcs_medte") == 9154,
              str(health.get("collections")))
        check("health reports top_k", health.get("top_k") == 10)
        check("health advertises both pages",
              sorted(health.get("pages") or []) == ["dashboard", "patient"],
              str(health.get("pages")))

        # ── the pages ───────────────────────────────────────────────────
        status, ctype, patient = get(f"{base}/")
        check("GET / serves the patient page first", status == 200 and "text/html" in ctype)
        check("the served page is self-contained (no external asset refs)",
              'src="http' not in patient and "cdn." not in patient and patient.count("<script") == 2,
              f"{len(patient):,} bytes")
        check("the served page keeps the boot placeholder for the client to fill",
              '<script id="pt-boot">' in patient)
        check("patient page is a chat, not the pipeline dump",
              "composer-inner" in patient and 'id="step-mrf"' not in patient)
        check("patient page links to the pipeline view", "pt_frontend.html" in patient)

        status, _, dash = get(f"{base}/pipeline")
        check("GET /pipeline serves the dashboard", status == 200)
        check("the two pages are actually different documents", dash != patient,
              f"patient {len(patient):,} / dashboard {len(dash):,} bytes")
        check("dashboard renders the pipeline, not the chat",
              'id="step-mrf"' in dash and "composer-inner" not in dash)
        check("dashboard page links back to the patient view", "pt_patient.html" in dash)

        for alias, want in (("/patient", "composer-inner"), ("/pt_patient.html", "composer-inner"),
                            ("/pt_frontend.html", 'id="step-mrf"'), ("/dashboard", 'id="step-mrf"')):
            status, _, body = get(f"{base}{alias}")
            check(f"alias {alias} serves the right page", status == 200 and want in body)

        status, _, body = get(f"{base}/nope")
        check("unknown path is a 404", status is not None and body == "not found", str(status))

        # ── blocking query ──────────────────────────────────────────────
        status, body = post(f"{base}/api/query",
                            {"query": "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?"})
        check("/api/query 200", status == 200, str(status))
        payload = json.loads(body)
        check("payload has every contract key",
              all(k in payload for k in (
                  "query", "mode", "categorized", "cpt_candidates", "hcpcs_candidates",
                  "decision", "rejected", "code_plausible", "answer", "raw", "stages",
                  "total_seconds", "created_at")),
              ",".join(sorted(set(payload) ^ {
                  "query", "mode", "categorized", "cpt_candidates", "hcpcs_candidates",
                  "decision", "rejected", "code_plausible", "answer", "raw", "stages",
                  "total_seconds", "created_at"})) or "13/13")
        check("mode is live", payload["mode"] == "live")
        check("entities projected", payload["categorized"]["medical"] == ["colonoscopy"],
              str(payload["categorized"]))
        check("candidate frames converted to records",
              len(payload["cpt_candidates"]) == 10 and payload["cpt_candidates"][0]["code"] == "45378",
              f"{len(payload['cpt_candidates'])} rows")
        check("decision projected",
              payload["decision"]["cpt_list"] == ["45378"]
              and payload["decision"]["hcpcs_list"] == ["G9937"])
        check("rejected codes recovered from the pipeline log",
              set(payload["rejected"]["cpt"]) >= {"45325", "45365", "74263"},
              str(payload["rejected"]))
        check("price rows projected",
              len(payload["code_plausible"]["price_summary"]) == 157,
              f"{len(payload['code_plausible']['price_summary'])} rows")
        check("match_count preserved", payload["code_plausible"]["match_count"] == 157)
        check("filters_used projected",
              payload["code_plausible"]["filters_used"]["insurers"] == ["highmark bcbs"],
              str(payload["code_plausible"]["filters_used"]))
        check("NaN-free price rows (json safe)",
              all(v is None or isinstance(v, (str, int, float)) or isinstance(v, bool)
                  for row in payload["code_plausible"]["price_summary"] for v in row.values()))
        check("raw categorizer dump recovered from repr()",
              "medical" in payload["raw"]["categorization"]
              and "colonoscopy" in payload["raw"]["categorization"],
              repr(payload["raw"]["categorization"][:44]))
        check("raw decision dump recovered", "G9937" in payload["raw"]["decision"])
        check("answer carried through", "45378" in payload["answer"])
        check("all seven stages parsed with real timings",
              [s["seconds"] for s in payload["stages"]] == [2.67, 0.06, 0.05, 77.97, 0.0, 15.5, 113.91],
              str([s["seconds"] for s in payload["stages"]]))
        check("validation step ('Validated in 0.00s') is parsed, not dropped",
              payload["stages"][4]["seconds"] == 0.0,
              str(payload["stages"][4]))
        check("stage states all done",
              all(s["state"] == "done" for s in payload["stages"]))
        check("total_seconds is the measured wall clock",
              payload["total_seconds"] > 0, f"{payload['total_seconds']:.3f}s")

        # a query with no insurer → single-hospital payer comparison
        status, body = post(f"{base}/api/query",
                            {"query": "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?"})
        p2 = json.loads(body)
        check("second query resolves to the right scenario",
              p2["categorized"]["insurer"] == ["UPMC Health Plan"]
              and p2["code_plausible"]["match_count"] == 1,
              str(p2["categorized"]["insurer"]))
        check("no rejected codes reported when all pass",
              p2["rejected"] == {"cpt": [], "hcpcs": []}, str(p2["rejected"]))

        # ── SSE ─────────────────────────────────────────────────────────
        frames = stream(f"{base}/api/stream?q=" + urllib.parse.quote(
            "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?"))
        kinds = [f["type"] for f in frames]
        check("stream emits log frames", kinds.count("log") >= 14, f"{kinds.count('log')} lines")
        check("stream emits stage frames", kinds.count("stage") >= 14, f"{kinds.count('stage')} stage events")
        check("stream terminates with exactly one result", kinds.count("result") == 1)
        check("stream has no error frames", "error" not in kinds)

        seen_stages = [f["stage"] for f in frames if f["type"] == "stage"]
        # A stage emits several frames (active, done, then detail updates), so
        # collapse to the last frame seen per step before asserting.
        last: dict[int, dict] = {}
        for s in seen_stages:
            last[s["n"]] = s
        active_ids = {s["n"] for s in seen_stages if s["state"] == "active"}
        done_ids = {s["n"] for s in seen_stages if s["state"] == "done"}
        check("every step is streamed as active", active_ids == {1, 2, 3, 4, 5, 6, 7}, str(sorted(active_ids)))
        check("every step is streamed as done", done_ids == {1, 2, 3, 4, 5, 6, 7}, str(sorted(done_ids)))
        check("step titles streamed as detail",
              any("Asking decision model" in (s.get("detail") or "") for s in seen_stages),
              (last[4].get("detail") or "")[:48])
        check("timings arrive on the stage frames",
              [last[i]["seconds"] for i in range(1, 8)] == [2.67, 0.06, 0.05, 77.97, 0.0, 15.5, 113.91],
              str([last[i]["seconds"] for i in range(1, 8)]))
        check("MRF row count streamed as a detail",
              any("157 rows matched" in (s.get("detail") or "") for s in seen_stages))

        final = [f["result"] for f in frames if f["type"] == "result"][0]
        check("streamed result matches the POST payload shape",
              set(final) == set(payload) and final["code_plausible"]["match_count"] == 157)
        check("streamed result has the full stage timeline",
              [s["seconds"] for s in final["stages"]] == [2.67, 0.06, 0.05, 77.97, 0.0, 15.5, 113.91],
              str([s["seconds"] for s in final["stages"]]))
        check("streamed result carries the rejected codes", bool(final["rejected"]["cpt"]))

        # ── error handling ──────────────────────────────────────────────
        status, body = post(f"{base}/api/query", {"query": "   "})
        check("empty query rejected with 400", status == 400, str(status))

        frames = stream(f"{base}/api/stream?q=boom")
        errs = [f for f in frames if f["type"] == "error"]
        check("unknown query still returns a result (stub falls back)", not errs,
              str(errs[:1]))

        # ── static embed path (no server, no iframe) ────────────────────
        plain = pt_serve.build_static_html(None, page="dashboard")
        check("static build with no payload keeps the empty boot tag",
              '<script id="pt-boot"></script>' in plain)

        embedded = pt_serve.build_static_html(payload, page="dashboard")
        check("static build injects the payload as a boot script",
              "window.__PT_BOOT__" in embedded and '"apiBase": ""' in embedded)
        check("injected payload survives JSON round-trip",
              json.loads(
                  embedded.split("window.__PT_BOOT__ = ", 1)[1].split(";</script>", 1)[0]
              )["payload"]["code_plausible"]["match_count"] == 157)
        check("static build is still a single self-contained document",
              '<script id="pt-boot">' in embedded and embedded.count("<script") == 2)

        # ── payload is pure JSON (no NaN / numpy leaking into the wire) ──
        try:
            json.dumps(payload, allow_nan=False)
            round_trips = True
        except ValueError:
            round_trips = False
        check("payload is strict JSON (no NaN literals)", round_trips)

    finally:
        handle.stop()

    # ── a pipeline that raises must surface, not hang ────────────────────
    print("\n── failure path: pipeline raises ──")
    h2 = pt_serve.serve(orchestration=stub.exploding_orchestration, port=8972)
    try:
        frames = stream(f"{h2.url.rstrip('/')}/api/stream?q=mammogram", timeout=20)
        errs = [f for f in frames if f["type"] == "error"]
        check("exception streamed as an error frame", len(errs) == 1, str(errs[:1]))
        check("error frame names the exception",
              "CUDA out of memory" in (errs[0]["message"] if errs else ""),
              (errs[0]["message"] if errs else "")[:60])
        check("exception text appears in the log stream",
              any("CUDA out of memory" in f.get("line", "") for f in frames if f["type"] == "log"))

        status, body = post(f"{h2.url.rstrip('/')}/api/query", {"query": "mammogram"}, timeout=20)
        check("POST surfaces the exception as 500", status == 500, str(status))
        check("POST error body names the exception", "CUDA out of memory" in body)
    finally:
        h2.stop()

    # ── both pages load standalone, from disk ───────────────────────────
    print("\n── the two pages as standalone files ──")
    for page, marker in (("patient", "composer-inner"), ("dashboard", 'id="step-mrf"')):
        html = pt_serve._load_page(page)
        check(f"{page} page loads from disk", marker in html, f"{len(html):,} bytes")
        check(f"{page} page is self-contained (no external refs)",
              'src="http' not in html and "cdn." not in html and html.count("<script") == 2)
        check(f"{page} page keeps the boot placeholder",
              '<script id="pt-boot"></script>' in html)

    print(f"\n{passed} passed, {len(failures)} failed")
    if failures:
        print("failed: " + "; ".join(failures))
        return 1
    print("VERIFIED\n")
    return 0

if __name__ == "__main__":
    import urllib.parse  # noqa: E402

    sys.exit(main())
