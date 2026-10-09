"""
Colab integration for the CMU-NIST AIMSEC price-transparency front end.

Standard library only — nothing to pip install.

Typical use, in the notebook that already defines `orchestration`, `mrf_data`
and `client` (Copy_of_Try_PT-3.ipynb), add one cell:

    import sys; sys.path.insert(0, "/content/pt_frontend/colab")
    import pt_serve
    pt_serve.launch(
        orchestration=orchestration,
        mrf_data=mrf_data,
        client=client,
        top_k=TOP_K,
    )

That starts a threaded HTTP server on the kernel and renders the UI in an
iframe, wired to a live SSE stream of the real pipeline.

Why it is built this way
------------------------
`orchestration()` is used *unmodified*. Instead of re-implementing or wrapping
its logic, the run's stdout is tee'd: the server recognises the pipeline's own
log lines ("STEP 4/7 — ...", "Decision completed in 86.56s") and turns them into
progress events, and reads the raw categorizer/decision dumps and the
"Rejected ... codes" lines straight out of the log. Editing the notebook is
therefore unnecessary, and the UI can never drift from the pipeline's behaviour.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
import socket
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = ["launch", "serve", "render", "build_static_html", "to_payload", "ServerHandle"]

# Optional: build.mjs can inline a base64 copy of each page here to produce a
# single-file drop-in for notebooks (dist/pt_colab.py). Left empty by default,
# in which case the pages are read from disk next to this module.
_EMBEDDED_HTML_B64 = ""
_EMBEDDED_PATIENT_B64 = ""

# Which page each URL serves. The patient chat is the product surface, so it
# owns "/"; the pipeline dashboard is one link away.
_ROUTES = {
    "/": "patient",
    "/index.html": "patient",
    "/patient": "patient",
    "/pt_patient.html": "patient",
    "/pipeline": "dashboard",
    "/dashboard": "dashboard",
    "/pt_frontend.html": "dashboard",
}

_PAGE_FILES = {
    "patient": "pt_patient.html",
    "dashboard": "pt_frontend.html",
}


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(value: Any) -> Any:
    """Convert pandas/numpy scalars, NaN and frames into plain JSON values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else value  # NaN -> None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    # numpy scalars expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass

    # pandas NA / NaT
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None

    try:
        if value != value:  # noqa: PLR0124 - NaN check for exotic types
            return None
    except Exception:
        pass

    return str(value)


def _records(frame: Any, limit: int | None = None) -> list[dict]:
    """DataFrame -> list[dict], NaN normalised to null. Empty for anything else."""
    if frame is None or not hasattr(frame, "to_dict"):
        return []
    try:
        view = frame.head(limit) if limit else frame
        return [_jsonable(rec) for rec in view.to_dict(orient="records")]
    except Exception:
        return []


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out = []
    for v in value:
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            out.append(s)
    return out


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

# "Categorization completed in 12.41s" -> which stage that timing belongs to.
_TIMING_PHRASES = {
    "categorization": "entities",
    "cpt retrieval": "cpt",
    "hcpcs retrieval": "hcpcs",
    "decision": "decision",
    "validated": "validation",
    "mrf filtering": "mrf",
    "answer generation": "answer",
}

_STEP_RE = re.compile(r"^STEP\s+(\d)/7\s+[—-]\s+(.*)$")
# The pipeline logs most steps as "<Phase> completed in 3.21s" but validation as
# "Validated in 0.00s" — accept both forms or step 5 silently loses its timing.
_TIME_RE = re.compile(r"^(.*?)\s+(?:completed\s+)?in\s+([\d.]+)s\s*$")
_MATCH_RE = re.compile(r"^MRF rows matched:\s*(\S+)")
_REJECT_RE = re.compile(r"^Rejected (CPT|HCPCS) codes not found in retrieval results:\s*(\[.*\])\s*$")
_RAW_MARKER = re.compile(r"^Raw (categorizer|decision-model) output:\s*$")

PRICE_ROW_LIMIT = 1000


def to_payload(
    *,
    query: str,
    result: dict,
    run_log: str = "",
    stages: list[dict] | None = None,
    total_seconds: float | None = None,
    mode: str = "live",
) -> dict:
    """Project `orchestration()`'s return value onto the front-end contract."""
    result = result or {}
    categorized = {
        "medical": _list((result.get("categorized") or {}).get("medical")),
        "hospital": _list((result.get("categorized") or {}).get("hospital")),
        "insurer": _list((result.get("categorized") or {}).get("insurer")),
        "medication": _list((result.get("categorized") or {}).get("medication")),
    }

    decision = result.get("output") or {}
    cpt_list = [c.upper() for c in _list(decision.get("cpt_list"))]
    hcpcs_list = [c.upper() for c in _list(decision.get("hcpcs_list"))]

    plausible = result.get("code_plausible") or {}
    if not isinstance(plausible, dict):
        plausible = {}

    price_summary = _records(plausible.get("price_summary"), limit=PRICE_ROW_LIMIT)
    try:
        match_count = int(plausible.get("match_count"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        match_count = len(price_summary)

    filters = plausible.get("filters_used") or {}
    if not isinstance(filters, dict):
        filters = {}

    rejected = _parse_rejections(run_log)
    raw_cat, raw_decision = _parse_raw(run_log)

    return {
        "query": query,
        "mode": mode,
        "categorized": categorized,
        "cpt_candidates": _records(result.get("cpt_candidates")),
        "hcpcs_candidates": _records(result.get("hcpcs_candidates")),
        "decision": {
            "use_codes": str(decision.get("use_codes") or "none"),
            "cpt_list": cpt_list,
            "hcpcs_list": hcpcs_list,
        },
        "rejected": rejected,
        "code_plausible": {
            "match_count": match_count,
            "matched_rows_total": match_count,
            "filters_used": {
                "cpt_codes": _list(filters.get("cpt_codes")),
                "hcpcs_codes": _list(filters.get("hcpcs_codes")),
                "insurers": _list(filters.get("insurers")),
                "hospitals": _list(filters.get("hospitals")),
            },
            "price_summary": price_summary,
        },
        "answer": str(result.get("answer") or ""),
        "raw": {"categorization": raw_cat, "decision": raw_decision},
        "stages": stages or _stages_from_log(run_log) or _blank_stages(done=True),
        "total_seconds": float(
            total_seconds if total_seconds is not None else (result.get("total_seconds") or 0.0)
        ),
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def _blank_stages(*, done: bool = False) -> list[dict]:
    return [
        {
            "n": i + 1,
            "key": STAGE_KEYS[i],
            "title": STAGE_TITLES[i],
            "state": "done" if done else "pending",
            "seconds": None,
            "detail": None,
        }
        for i in range(7)
    ]


def _stages_from_log(log: str) -> list[dict]:
    """Rebuild the step timeline from the pipeline's own stdout."""
    stages = _blank_stages()
    seen_any = False

    for line in (log or "").splitlines():
        line = line.strip()

        m = _STEP_RE.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < 7:
                stages[idx]["state"] = "active"
                stages[idx]["detail"] = m.group(2).strip()
                seen_any = True
            continue

        m = _TIME_RE.match(line)
        if m:
            phrase = m.group(1).strip().lower()
            for needle, key in _TIMING_PHRASES.items():
                if needle in phrase:
                    idx = STAGE_KEYS.index(key)
                    stages[idx]["seconds"] = float(m.group(2))
                    stages[idx]["state"] = "done"
                    seen_any = True
                    break
            continue

        m = _MATCH_RE.match(line)
        if m:
            idx = STAGE_KEYS.index("mrf")
            stages[idx]["detail"] = f"{m.group(1)} rows matched"
            seen_any = True

    if seen_any:
        for s in stages:
            if s["state"] == "active":
                s["state"] = "done"
    return stages if seen_any else []


def _parse_rejections(log: str) -> dict:
    """Read the pipeline's own 'Rejected ... codes' lines back out of the log."""
    out = {"cpt": [], "hcpcs": []}
    for line in (log or "").splitlines():
        m = _REJECT_RE.match(line.strip())
        if not m:
            continue
        try:
            codes = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            codes = []
        key = "cpt" if m.group(1) == "CPT" else "hcpcs"
        for c in codes if isinstance(codes, (list, tuple)) else []:
            s = str(c).strip()
            if s:
                out[key].append(s)
    return out


def _parse_raw(log: str) -> tuple[str, str]:
    """
    Recover the raw model dumps.

    The pipeline logs them with `repr(...)`, which is always a single line, so
    the payload is the line immediately after the marker.
    """
    lines = (log or "").splitlines()
    cat = ""
    dec = ""
    for i, line in enumerate(lines):
        m = _RAW_MARKER.match(line.strip())
        if not m or i + 1 >= len(lines):
            continue
        raw = lines[i + 1].strip()
        try:
            text = ast.literal_eval(raw)
            if not isinstance(text, str):
                text = raw
        except (ValueError, SyntaxError):
            text = raw
        if m.group(1) == "categorizer":
            cat = text
        else:
            dec = text
    return cat, dec


# ─────────────────────────────────────────────────────────────────────────────
# Running a query while streaming its log
# ─────────────────────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """stdout that forwards each complete line to a callback."""

    def __init__(self, sink: Callable[[str], None], mirror: io.TextIOBase | None = None):
        self._sink = sink
        self._mirror = mirror
        self._buf = ""
        self.lines: list[str] = []

    def write(self, s: str) -> int:  # noqa: D102
        if self._mirror is not None:
            try:
                self._mirror.write(s)
                self._mirror.flush()
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.lines.append(line)
            try:
                self._sink(line)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:  # noqa: D102
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass

    def close(self) -> None:  # noqa: D102
        if self._buf:
            self.lines.append(self._buf)
            try:
                self._sink(self._buf)
            except Exception:
                pass
            self._buf = ""


class PipelineRunner:
    """Serialises runs and gives each one a log stream."""

    def __init__(self, orchestration: Callable[..., dict]):
        self._orch = orchestration
        self._lock = threading.Lock()

    def run(self, query: str, on_line: Callable[[str], None], mirror: bool = False):
        """
        Execute one query. Returns (result, log_text, elapsed_seconds).

        `orchestration` is called exactly as the notebook calls it; the only
        difference is that stdout is tee'd while it runs.
        """
        import sys

        with self._lock:
            tee = _Tee(on_line, mirror=sys.__stdout__ if mirror else None)
            real_out, real_err = sys.stdout, sys.stderr
            sys.stdout = tee
            t0 = time.perf_counter()
            try:
                result = self._orch(query)
                elapsed = time.perf_counter() - t0
                error = None
            except Exception as exc:  # surface the traceback in the log
                elapsed = time.perf_counter() - t0
                result = {}
                error = exc
                tee.write(traceback.format_exc())
            finally:
                tee.close()
                sys.stdout = real_out
                sys.stderr = real_err
            return result, "\n".join(tee.lines), elapsed, error


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

class ServerHandle:
    def __init__(self, server: ThreadingHTTPServer, port: int, thread: threading.Thread, url: str):
        self.server = server
        self.port = port
        self.thread = thread
        self.url = url

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def __repr__(self) -> str:
        return f"<ServerHandle port={self.port} url={self.url}>"


def _free_port(preferred: int = 8000) -> int:
    for candidate in [preferred, 8001, 8080, 8888, 0]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    return 8000


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PTFrontend/1.0"

    # populated by `serve()`
    pages: dict = {}
    runner: PipelineRunner | None = None
    meta: dict = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # keep the notebook output clean

    # ── helpers ──────────────────────────────────────────────────────────

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _sse_open(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

    def _sse(self, obj: Any) -> None:
        self.wfile.write(f"data: {json.dumps(obj, default=str)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # ── routes ───────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if len(path) > 1:
            path = path.rstrip("/") or "/"

        page = _ROUTES.get(path)
        if page:
            html = self.pages.get(page)
            if html is None:
                self._send(404, f"page '{page}' not loaded".encode("utf-8"), "text/plain")
                return
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/health":
            self._json({"ok": True, "mode": "live", **self.meta})
            return

        if path == "/api/stream":
            query = ""
            if "?" in self.path:
                from urllib.parse import parse_qs

                query = (parse_qs(self.path.split("?", 1)[1]).get("q") or [""])[0]
            self._stream(query)
            return

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/api/query":
            self._send(404, b"not found", "text/plain")
            return

        try:
            length = int(self.headers.get("content-length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            payload = {}
        query = str(payload.get("query") or "")

        if not query.strip():
            self._json({"error": "empty query"}, 400)
            return

        if self.runner is None:
            self._json({"error": "server not initialised"}, 500)
            return

        result, log, elapsed, error = self.runner.run(query, lambda _l: None)
        if error is not None:
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)
            return
        self._json(to_payload(query=query, result=result, run_log=log, total_seconds=elapsed))

    # ── streaming ────────────────────────────────────────────────────────

    def _stream(self, query: str) -> None:
        self._sse_open()
        if not query.strip():
            self._sse({"type": "error", "message": "empty query"})
            return

        if self.runner is None:
            self._sse({"type": "error", "message": "server not initialised"})
            return

        stages = _blank_stages()

        def emit_stage(idx: int, state: str | None = None, seconds: float | None = None, detail=None):
            if state:
                stages[idx]["state"] = state
            if seconds is not None:
                stages[idx]["seconds"] = seconds
            if detail is not None:
                stages[idx]["detail"] = detail
            self._sse({"type": "stage", "stage": dict(stages[idx])})

        def on_line(line: str) -> None:
            self._sse({"type": "log", "line": line})

            m = _STEP_RE.match(line.strip())
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < 7:
                    emit_stage(idx, state="active", detail=m.group(2).strip())
                return

            m = _TIME_RE.match(line.strip())
            if m:
                phrase = m.group(1).strip().lower()
                for needle, key in _TIMING_PHRASES.items():
                    if needle in phrase:
                        idx = STAGE_KEYS.index(key)
                        emit_stage(idx, state="done", seconds=float(m.group(2)))
                        break
                return

            m = _MATCH_RE.match(line.strip())
            if m:
                emit_stage(STAGE_KEYS.index("mrf"), detail=f"{m.group(1)} rows matched")

        try:
            result, log, elapsed, error = self.runner.run(query, on_line)

            for i, s in enumerate(stages):
                if s["state"] in ("active", "pending"):
                    emit_stage(i, state="error" if error is not None else "done")

            if error is not None:
                self._sse({"type": "error", "message": f"{type(error).__name__}: {error}"})
                return

            payload = to_payload(
                query=query, result=result, run_log=log, stages=stages, total_seconds=elapsed
            )
            self._sse({"type": "result", "result": payload})
        except (BrokenPipeError, ConnectionResetError):
            return  # the browser navigated away mid-run; the kernel keeps going
        except Exception as exc:
            try:
                self._sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass


def serve(
    *,
    orchestration: Callable[..., dict],
    html_path: str | Path | None = None,
    patient_html_path: str | Path | None = None,
    mrf_data: Any = None,
    client: Any = None,
    top_k: int = 10,
    models: Iterable[str] | None = None,
    port: int = 8000,
    host: str = "127.0.0.1",
) -> ServerHandle:
    """Start the HTTP server. Idempotent per call — keep the handle to stop it."""
    pages = {
        "patient": _load_page("patient", patient_html_path),
        "dashboard": _load_page("dashboard", html_path),
    }

    meta: dict[str, Any] = {
        "mrf_rows": None,
        "collections": {},
        "top_k": top_k,
        "models": list(models or _default_models()),
        "pages": sorted(pages),
    }
    try:
        if mrf_data is not None and hasattr(mrf_data, "__len__"):
            meta["mrf_rows"] = int(len(mrf_data))
    except Exception:
        pass

    if client is not None and hasattr(client, "get_collection"):
        for name in ("cpt_medte", "hcpcs_medte"):
            try:
                meta["collections"][name] = int(client.get_collection(name).points_count)
            except Exception:
                pass

    chosen = _free_port(port)
    _Handler.pages = pages
    _Handler.runner = PipelineRunner(orchestration)
    _Handler.meta = meta

    httpd = ThreadingHTTPServer((host, chosen), _Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="pt-frontend", daemon=True)
    thread.start()
    return ServerHandle(httpd, chosen, thread, f"http://{host}:{chosen}/")


def _default_models() -> list[str]:
    return [
        "google/medgemma-1.5-4b-it",
        "MohammadKhodadad/MedTE-cl15-step8000",
    ]


def _load_page(page: str = "patient", html_path: str | Path | None = None) -> str:
    """Return the HTML for a page: embedded copy, or a file next to this module."""
    key = "dashboard" if page == "dashboard" else "patient"

    if html_path:
        p = Path(html_path)
        if p.is_file():
            return p.read_text(encoding="utf-8")
        raise FileNotFoundError(f"html_path not found: {p}")

    embedded = _EMBEDDED_HTML_B64 if key == "dashboard" else _EMBEDDED_PATIENT_B64
    if embedded:
        return base64.b64decode(embedded).decode("utf-8")

    here = Path(__file__).resolve().parent
    name = _PAGE_FILES[key]
    for c in (
        here / name,
        here.parent / "dist" / name,
        here.parent / name,
        Path("/content") / name,
        Path("/content/dist") / name,
    ):
        try:
            if c.is_file():
                return c.read_text(encoding="utf-8")
        except OSError:
            continue

    raise FileNotFoundError(
        f"{name} not found. Build it with `node build.mjs` and upload it next to "
        "this file, use dist/pt_colab.py (which embeds both pages), or pass html_path=..."
    )


def _load_html(html_path: str | Path | None = None) -> str:
    """Back-compat: the pipeline dashboard."""
    return _load_page("dashboard", html_path)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook entry points
# ─────────────────────────────────────────────────────────────────────────────

_LAST_HANDLE: "ServerHandle | None" = None


def launch(
    *,
    orchestration: Callable[..., dict],
    mrf_data: Any = None,
    client: Any = None,
    top_k: int = 10,
    models: Iterable[str] | None = None,
    page: str = "patient",
    html_path: str | Path | None = None,
    patient_html_path: str | Path | None = None,
    port: int = 8000,
    height: int = 1180,
    width: str = "100%",
) -> ServerHandle:
    """
    Start the server and display a page in a Colab iframe. Returns the handle.

    `page` selects what the iframe opens: "patient" (the chat, default) or
    "dashboard" (the step-by-step pipeline view). Both are served either way.

    Safe to re-run: a server started by a previous `launch()` is shut down first,
    so re-executing the cell does not leave listeners stacked on the kernel.

    Falls back to a static (offline-sample) render if the Colab output proxy is
    unavailable, e.g. when running in a plain Jupyter or a local script.
    """
    global _LAST_HANDLE
    if _LAST_HANDLE is not None:
        try:
            _LAST_HANDLE.stop()
        except Exception:
            pass
        _LAST_HANDLE = None

    handle = serve(
        orchestration=orchestration,
        html_path=html_path,
        patient_html_path=patient_html_path,
        mrf_data=mrf_data,
        client=client,
        top_k=top_k,
        models=models,
        port=port,
    )
    _LAST_HANDLE = handle

    path = "/" if page != "dashboard" else "/pipeline"

    try:
        from google.colab import output  # type: ignore

        try:
            output.serve_kernel_port_as_iframe(handle.port, path=path, height=height, width=width)
        except TypeError:
            # Older Colab helpers do not take `path`; "/" is the patient page.
            output.serve_kernel_port_as_iframe(handle.port, height=height, width=width)
            if path != "/":
                print(f"Dashboard: {handle.url.rstrip('/')}/pipeline")
        return handle
    except Exception:
        pass

    try:
        from IPython.display import HTML, display  # type: ignore

        display(
            HTML(
                f'<iframe src="{handle.url.rstrip("/")}{path}" width="{width}" height="{height}" '
                'style="border:1px solid #e4e4e1;border-radius:10px"></iframe>'
            )
        )
    except Exception:
        print(f"Front end serving at {handle.url.rstrip('/')}{path}")

    return handle


def build_static_html(
    payload: dict | None,
    html_path: str | Path | None = None,
    page: str = "patient",
) -> str:
    """
    Return a built page with a payload (or a plain boot config) injected.

    Split out from `render()` so the transformation is testable without IPython.
    """
    html = _load_page(page, html_path)
    marker = '<script id="pt-boot"></script>'
    if marker not in html:
        raise ValueError("boot placeholder missing from the built HTML — rebuild with `node build.mjs`")

    if payload is None:
        return html

    boot = json.dumps({"apiBase": "", "payload": payload}, ensure_ascii=False)
    return html.replace(marker, f'<script id="pt-boot">window.__PT_BOOT__ = {boot};</script>')


def render(
    result: dict | None = None,
    *,
    query: str = "",
    html_path: str | Path | None = None,
    page: str = "patient",
    run_log: str = "",
    height: int = 1180,
    width: str = "100%",
) -> Any:
    """
    Display a page as a static snapshot of one payload — no server, no iframe.
    Use this to embed a finished run in the notebook output.

        payload = pt_serve.to_payload(query=q, result=result)
        pt_serve.render(payload)                       # patient chat
        pt_serve.render(payload, page="dashboard")     # pipeline view
    """
    from IPython.display import HTML, display  # type: ignore

    html = build_static_html(result, html_path, page)
    iframe = (
        f'<iframe srcdoc="{_attr_escape(html)}" width="{width}" height="{height}" '
        'style="border:1px solid #e4e4e1;border-radius:10px;background:#fff"></iframe>'
    )
    display(HTML(iframe))
    return None


def _attr_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
