"""Colab runner for the Healthcare Price Transparency (HPT) prototype.

Single entry point that keeps every Colab-specific bit (the ``!pip install``
cell, ``drive.mount``, the ``!cp`` + ``!tar -xzf`` snapshot staging, dotenv /
Colab-secret token lookup) next to the wiring for models, Qdrant, the MRF *and*
the TypeScript front end, which can be published on a public URL through a
Cloudflare quick tunnel.

Ways to use it in Colab
-----------------------

**Backend only** (answers in the notebook)::

    %cd /content/HPT_Prototype
    import colab_run

    colab_run.mount_drive()               # drive.mount("/content/drive")
    colab_run.stage_db()                  # !cp ... && !tar -xzf ...
    pipeline = colab_run.setup(mount=False, stage=False)

    result = colab_run.run("What might a diagnostic mammogram cost at "
                           "UPMC Presbyterian with UPMC Health Plan?", pipeline)

**Front end + backend on a public URL** (the chat page, reachable anywhere)::

    %cd /content/HPT_Prototype
    import colab_run

    ui = colab_run.serve_ui()             # models + data + UI + tunnel
    print(ui.page_url)                    # https://<random>.trycloudflare.com/

    ui.stop()                             # ends the server and the tunnel

``serve_ui()`` drives the vendored ``ui/pt_serve.py`` (stdlib HTTP + SSE server)
with ``pipeline.run`` as its orchestration callable, so the UI streams the real
backend: the server tee's stdout and parses the pipeline's own log lines, which
means the pages can never drift from the pipeline's behaviour.

**As a script**::

    !git clone https://github.com/adi274903/HPT_Prototype /content/HPT_Prototype
    %cd /content/HPT_Prototype
    !pip install -q -r requirements.txt
    !python colab_run.py                       # backend: the 3 sample queries
    !python colab_run.py --serve-ui --tunnel none

SECURITY: a tunnel URL is public and unauthenticated. Anyone holding it can run
queries against the pipeline (and consume the GPU). Use ``--tunnel none`` when
you do not need a public URL, and stop the server (``ui.stop()``) when finished.

Nothing here is imported by the ``pt_healthcare`` package, so the package still
imports cleanly on a machine without ``google.colab``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --- make the sibling package importable when running from the repo root ----
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pt_healthcare import config  # noqa: E402
from pt_healthcare.data import (  # noqa: E402
    extract_archive,
    get_qdrant_client,
    load_mrf_data,
    login_huggingface,
)
from pt_healthcare.llm import MedGemmaEngine  # noqa: E402
from pt_healthcare.models import load_models  # noqa: E402
from pt_healthcare.pipeline import HealthcarePricingPipeline  # noqa: E402
from pt_healthcare.retrieval import CodeRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# Colab defaults — exactly the paths the original notebook used
# ---------------------------------------------------------------------------
DRIVE_MOUNT_POINT = "/content/drive"
COLAB_WORKDIR = "/content"

#: Snapshot the notebook copied from Drive before extracting.
DRIVE_DB_ARCHIVE = "/content/drive/MyDrive/New_PT_DB_Backups/New_PT_DB.tar.gz"

#: Extracted local Qdrant store (what the notebook passed to QdrantClient).
COLAB_DB_DIR = "/content/New_PT_DB"

#: Cleaned MRF CSV the notebook read.
DRIVE_MRF_CSV = (
    "/content/drive/MyDrive/Price_Transparency/datasets/everyUPMCmrf_clean (1).csv"
)

#: The notebook's ``!pip install`` line, as a package list.
DEFAULT_PIP_PACKAGES: Sequence[str] = (
    "huggingface_hub",
    "transformers",
    "torchaudio",
    "torchvision",
    "sentence-transformers",
    "qdrant-client",
)

#: The notebook's validation queries.
DEFAULT_QUERIES: Sequence[str] = (
    "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?",
    "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?",
    "Best and cheapest hospital for me to get my mammogram done?",
)

# ---------------------------------------------------------------------------
# Front-end assets (vendored into ui/)
# ---------------------------------------------------------------------------
UI_DIR = os.path.join(_REPO_ROOT, "ui")
PATIENT_HTML = os.path.join(UI_DIR, "pt_patient.html")
DASHBOARD_HTML = os.path.join(UI_DIR, "pt_frontend.html")
PT_SERVE_MODULE = "pt_serve"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules or os.path.isdir("/content")


def pip_install_command(packages: Sequence[str] = DEFAULT_PIP_PACKAGES) -> str:
    """Return the ``!pip install ...`` line for a notebook cell."""
    return "pip install " + " ".join(packages)


def install_dependencies(
    packages: Sequence[str] = DEFAULT_PIP_PACKAGES,
    quiet: bool = True,
) -> int:
    """Install the runtime packages (the notebook's ``!pip install`` cell)."""
    command = [sys.executable, "-m", "pip", "install", *packages]

    if quiet:
        command.append("-q")

    return subprocess.call(command)


#: (import name, pip name) for everything the pipeline needs at runtime.
REQUIRED_PACKAGES: Sequence[Tuple[str, str]] = (
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("sentence_transformers", "sentence-transformers"),
    ("qdrant_client", "qdrant-client"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("dotenv", "python-dotenv"),
)

REQUIREMENTS_FILE = os.path.join(_REPO_ROOT, "requirements.txt")


def missing_dependencies() -> List[str]:
    """Pip names of the runtime packages that are not importable right now."""
    import importlib.util

    missing: List[str] = []

    for module_name, package_name in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None

        if spec is None:
            missing.append(package_name)

    return missing


def install_requirements(
    packages: Optional[Sequence[str]] = None,
    quiet: bool = True,
) -> int:
    """``pip install -r requirements.txt``, or just ``packages`` if the file is gone."""
    if packages is None and os.path.isfile(REQUIREMENTS_FILE):
        command = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
    else:
        command = [sys.executable, "-m", "pip", "install", *(packages or [])]

    if quiet:
        command.append("-q")

    return subprocess.call(command)


def ensure_dependencies(
    auto_install: Optional[bool] = None,
    quiet: bool = True,
) -> List[str]:
    """Make sure the runtime packages are importable.

    Returns the packages that were missing (and installed, when allowed).
    ``auto_install=None`` means "yes inside Colab, no elsewhere", which matches
    the notebook, whose first cell was the ``!pip install`` line.

    Raises a plain ``RuntimeError`` with the fix in it rather than letting a
    ``ModuleNotFoundError`` surface from three frames down.
    """
    if auto_install is None:
        auto_install = in_colab()

    missing = missing_dependencies()

    if not missing:
        return []

    if not auto_install:
        raise RuntimeError(
            "Missing runtime packages: " + ", ".join(missing) + "\n"
            "Install them first, from the repo root:\n"
            "    !pip install -q -r requirements.txt\n"
            "or call colab_run.setup(install=True)."
        )

    print("Installing missing packages: " + ", ".join(missing))
    install_requirements(quiet=quiet)

    still_missing = missing_dependencies()

    if still_missing:
        raise RuntimeError(
            "Still missing after installing: " + ", ".join(still_missing) + "\n"
            "In Colab this normally means the runtime has to be restarted "
            "(Runtime -> Restart session) once the install finishes."
        )

    return missing


def load_env(
    dotenv_path: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """Load ``.env`` files and refresh the config snapshot; return the paths used.

    Searches the working directory, Colab's ``/content`` and this repo's root
    (or ``dotenv_path`` alone if given), so a ``.env`` that lives outside this
    repo — the normal Colab case — is still honoured. Existing environment
    variables and Colab secrets win over file values.
    """
    paths = [dotenv_path] if dotenv_path else None

    loaded = config.load_dotenv_files(paths, verbose=verbose)
    config.refresh()

    if verbose and not loaded:
        print(
            "No .env found (looked in "
            + ", ".join(config.env_file_candidates())
            + "); using environment variables only."
        )

    return loaded


def hf_token() -> Optional[str]:
    """Resolve the Hugging Face token from env, ``.env``, or Colab secrets.

    Checks ``HUGGINGFACE_API_KEY``, then ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``
    (the names huggingface_hub reads on its own), then Colab secrets under any of
    those names. ``build_pipeline()`` calls ``load_env()`` first, so a ``.env``
    in the working directory is already loaded by the time this runs.
    """
    token = config.hf_token_from_env()

    if token:
        return token

    try:
        from google.colab import userdata
    except Exception:
        return None

    for name in config.HF_TOKEN_ENV_VARS:
        try:
            value = userdata.get(name)
        except Exception:
            continue
        if value:
            return value

    return None


# ---------------------------------------------------------------------------
# Drive + snapshot staging
# ---------------------------------------------------------------------------
def mount_drive(
    mount_point: str = DRIVE_MOUNT_POINT,
    force_remount: bool = False,
) -> Any:
    """Mount Google Drive (the notebook's ``drive.mount(...)`` cell)."""
    from google.colab import drive

    return drive.mount(mount_point, force_remount=force_remount)


def stage_db(
    archive: str = DRIVE_DB_ARCHIVE,
    workdir: str = COLAB_WORKDIR,
    dest: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Copy the Qdrant snapshot into the runtime and extract it.

    Mirrors the notebook::

        !cp "/content/drive/.../New_PT_DB.tar.gz" /content/
        !tar -xzf "/content/New_PT_DB.tar.gz" -C /content/

    Returns the extracted store directory to hand to ``QdrantClient``.
    """
    if not os.path.isfile(archive):
        raise FileNotFoundError(
            f"Qdrant snapshot not found: {archive}\n"
            "Is Drive mounted (mount_drive()) and is the path correct?"
        )

    os.makedirs(workdir, exist_ok=True)

    local_archive = os.path.join(workdir, os.path.basename(archive))

    if os.path.abspath(archive) != os.path.abspath(local_archive):
        shutil.copy(archive, local_archive)

    dest = dest or os.path.join(workdir, "New_PT_DB")

    if verbose:
        print(f"Extracting {local_archive} -> {workdir}")

    extract_archive(local_archive, workdir)

    if not os.path.isdir(dest):
        raise RuntimeError(
            f"Expected Qdrant store at {dest} after extracting {local_archive}. "
            f"Contents of {workdir}: {sorted(os.listdir(workdir))}"
        )

    if verbose:
        print(f"Qdrant store ready at {dest}")

    return dest


# ---------------------------------------------------------------------------
# Wiring (backend)
# ---------------------------------------------------------------------------
def default_paths() -> Dict[str, str]:
    """The Colab paths the notebook used, as a dict."""
    return {
        "drive_mount_point": DRIVE_MOUNT_POINT,
        "db_archive": DRIVE_DB_ARCHIVE,
        "qdrant_path": COLAB_DB_DIR,
        "mrf_csv": DRIVE_MRF_CSV,
    }


def resolve_mrf_csv(explicit: Optional[str] = None) -> str:
    """explicit argument -> ``PT_MRF_CSV`` -> the Drive path the notebook used."""
    return explicit or os.getenv("PT_MRF_CSV") or DRIVE_MRF_CSV


def resolve_qdrant_path(explicit: Optional[str] = None) -> str:
    """explicit argument -> ``PT_QDRANT_PATH`` -> the extracted ``/content/New_PT_DB``."""
    return explicit or os.getenv("PT_QDRANT_PATH") or COLAB_DB_DIR


def build_pipeline(
    mrf_csv: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    device: str = "cuda",
    top_k: Optional[int] = None,
    login: bool = True,
    token: Optional[str] = None,
    install: Optional[bool] = None,
) -> HealthcarePricingPipeline:
    """Load models + data and return a ready-to-run pipeline.

    The pipeline keeps its own handles, which is what the UI server needs:

    - ``pipeline.retriever.client``      -> the Qdrant store
    - ``pipeline.retriever.embed_model``
    - ``pipeline.mrf_data``              -> row count for the status strip
    - ``pipeline.top_k``

    Paths resolve as explicit argument -> env (``PT_MRF_CSV`` / ``PT_QDRANT_PATH``,
    including anything from a ``.env``) -> the Colab Drive default.
    """
    ensure_dependencies(install)
    load_env()

    if login:
        login_huggingface(token or hf_token())

    pipe, embed_model = load_models(device=device)

    client = get_qdrant_client(resolve_qdrant_path(qdrant_path))
    mrf_data = load_mrf_data(resolve_mrf_csv(mrf_csv))

    top_k = config.top_k() if top_k is None else top_k

    return HealthcarePricingPipeline(
        retriever=CodeRetriever(embed_model, client, top_k=top_k),
        engine=MedGemmaEngine(pipe),
        mrf_data=mrf_data,
        top_k=top_k,
    )


def setup(
    install: Optional[bool] = None,
    mount: bool = True,
    stage: bool = True,
    device: str = "cuda",
    top_k: Optional[int] = None,
    mrf_csv: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    token: Optional[str] = None,
    env_file: Optional[str] = None,
    verbose: bool = True,
) -> HealthcarePricingPipeline:
    """One-call Colab bootstrap: install deps, mount Drive, stage the DB, wire models.

    Returns the pipeline so you can call it repeatedly::

        pipeline = setup()
        pipeline.run("...")

    ``env_file`` points at a specific ``.env``; otherwise the working directory,
    ``/content`` and the repo root are searched, and any ``PT_*`` value found
    there wins over the built-in defaults.

    ``install`` controls what happens when runtime packages are missing:
    ``None`` (default) installs them automatically inside Colab and raises
    otherwise; ``True`` always installs; ``False`` never does, raising a
    ``RuntimeError`` that names the packages and the command instead.
    """
    ensure_dependencies(install, quiet=not verbose)

    # Honour a .env — wherever it lives — before anything reads a path.
    load_env(env_file, verbose=verbose)

    if mount:
        mount_drive()

    resolved_mrf = resolve_mrf_csv(mrf_csv)
    resolved_db = resolve_qdrant_path(qdrant_path)

    if stage:
        resolved_db = stage_db(verbose=verbose)

    if verbose:
        print(f"MRF CSV   : {resolved_mrf}")
        print(f"Qdrant DB : {resolved_db}")

    return build_pipeline(
        mrf_csv=resolved_mrf,
        qdrant_path=resolved_db,
        device=device,
        top_k=top_k,
        token=token,
        install=False,  # already ensured above
    )


def run(
    query: str,
    pipeline: Optional[HealthcarePricingPipeline] = None,
    verbose: bool = True,
    **setup_kwargs: Any,
) -> Dict[str, Any]:
    """Answer one query, building the pipeline first if one was not passed."""
    if pipeline is None:
        pipeline = setup(**setup_kwargs)

    return pipeline.run(query, verbose=verbose)


def run_batch(
    queries: Optional[Sequence[str]] = None,
    pipeline: Optional[HealthcarePricingPipeline] = None,
    verbose: bool = True,
    **setup_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run several queries against one warm pipeline."""
    if pipeline is None:
        pipeline = setup(**setup_kwargs)

    return [pipeline.run(query, verbose=verbose) for query in (queries or DEFAULT_QUERIES)]


# ---------------------------------------------------------------------------
# Front end: the vendored pt_serve server
# ---------------------------------------------------------------------------
def import_pt_serve() -> Any:
    """Import the vendored ``ui/pt_serve.py`` (stdlib-only HTTP + SSE server)."""
    if UI_DIR not in sys.path:
        sys.path.insert(0, UI_DIR)

    import importlib

    module = importlib.import_module(PT_SERVE_MODULE)

    if not hasattr(module, "serve"):
        raise RuntimeError(
            f"{PT_SERVE_MODULE} at {UI_DIR} does not look like the pt_serve server."
        )

    return module


def check_ui_assets() -> Dict[str, bool]:
    """Which front-end assets are present (both pages are needed for the UI)."""
    return {
        "pt_serve.py": os.path.isfile(os.path.join(UI_DIR, "pt_serve.py")),
        "pt_patient.html": os.path.isfile(PATIENT_HTML),
        "pt_frontend.html": os.path.isfile(DASHBOARD_HTML),
    }


# ---------------------------------------------------------------------------
# Front end: a backend log pane
# ---------------------------------------------------------------------------
#: Page served at /logs. Self-contained, polls /api/log, so it works even when
#: the SSE stream is being buffered by a proxy (Cloudflare's, typically).
LOGS_PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backend log</title><style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin:0; background:#0e1113; color:#dfe4e6;
       font:13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
header { position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:10px;
         padding:9px 14px; background:#15191c; border-bottom:1px solid #262c30; }
h1 { font-size:13px; font-weight:600; margin:0; letter-spacing:.02em; }
.meta { margin-left:auto; display:flex; gap:14px; color:#8a9499; }
.dot { width:8px; height:8px; border-radius:50%; background:#3fb950; flex:none; }
.dot.off { background:#d29922; }
#log { margin:0; padding:14px; white-space:pre-wrap; word-break:break-word; }
#log div { padding:1px 0; }
.err { color:#f85149; }
.step { color:#58a6ff; }
label { user-select:none; cursor:pointer; }
@media (prefers-color-scheme: light) {
  body { background:#fff; color:#1c2024; }
  header { background:#f6f8fa; border-color:#d8dee4; }
  .meta { color:#59636e; } .step { color:#0550ae; }
}
</style></head><body>
<header>
  <span class="dot" id="dot"></span><h1>Backend log</h1><span id="state">connecting…</span>
  <span class="meta"><label><input type="checkbox" id="pin" checked> follow</label>
  <span id="count">0 lines</span></span>
</header>
<pre id="log"></pre><script>
(function () {
  var since = 0, total = 0, log = document.getElementById('log'),
      dot = document.getElementById('dot'), state = document.getElementById('state'),
      count = document.getElementById('count'), pin = document.getElementById('pin');

  function render(lines) {
    var frag = document.createDocumentFragment();
    lines.forEach(function (line) {
      var d = document.createElement('div');
      d.textContent = line === '' ? ' ' : line;
      if (/traceback|error/i.test(line)) d.className = 'err';
      else if (/^STEP/.test(line)) d.className = 'step';
      frag.appendChild(d);
    });
    log.appendChild(frag);
    total += lines.length;
    count.textContent = total + ' lines';
    while (log.childNodes.length > 5000) log.removeChild(log.firstChild);
    if (pin.checked) window.scrollTo(0, document.body.scrollHeight);
  }

  function tick() {
    fetch('/api/log?since=' + since, { headers: { accept: 'application/json' } })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (d) {
        dot.className = 'dot'; state.textContent = 'live';
        since = d.next;
        if (d.lines && d.lines.length) render(d.lines);
      })
      .catch(function (e) {
        dot.className = 'dot off';
        state.textContent = 'reconnecting… ' + (e && e.message ? e.message : '');
      })
      .then(function () { setTimeout(tick, 1000); });
  }

  tick();
})();
</script></body></html>
"""


class LogBuffer:
    """Thread-safe ring buffer of backend log lines, with absolute indexing."""

    def __init__(self, maxlen: int = 5000) -> None:
        self._lines: List[str] = []
        self._start = 0          # absolute index of _lines[0]
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self._maxlen:
                drop = len(self._lines) - self._maxlen
                del self._lines[:drop]
                self._start += drop

    def snapshot(self, since: int = 0) -> Tuple[List[str], int]:
        """Lines after absolute index ``since``, plus the new absolute index."""
        with self._lock:
            start, end = self._start, self._start + len(self._lines)

            if since <= start:
                return list(self._lines), end

            return list(self._lines[since - start :]), end

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)


def log_printer(
    buffer: Optional[LogBuffer] = None,
    echo: Optional[Any] = None,
    extra_sink: Optional[Callable[[str], None]] = None,
    tee_type: Optional[type] = None,
) -> Callable[[str], None]:
    """Build the pipeline's ``printer``.

    Writes to ``sys.stdout`` (which pt_serve's tee replaces during a run, so the
    SSE stream still sees every line), records into ``buffer`` for the /logs
    pane, forwards to ``extra_sink`` (the run manager, for the polling
    transport), and echoes to ``echo`` — but only while ``sys.stdout`` really is
    pt_serve's tee.

    ``tee_type`` exists because the obvious test — ``sys.stdout is not
    sys.__stdout__`` — is wrong in a notebook: Jupyter and pytest both swap
    ``sys.stdout`` for their own object, so that check fires permanently and
    every line gets echoed twice.
    """

    def printer(message: str) -> None:
        text = str(message)
        under_tee = tee_type is not None and isinstance(sys.stdout, tee_type)

        try:
            print(text)
        except Exception:
            pass

        if buffer is not None:
            buffer.append(text)

        if extra_sink is not None:
            try:
                extra_sink(text)
            except Exception:
                pass

        if echo is not None and under_tee:
            try:
                echo.write(text + "\n")
                echo.flush()
            except Exception:
                pass

    return printer


def make_logging_handler(base: Any, buffer: LogBuffer) -> Any:
    """Subclass pt_serve's handler to add /logs and /api/log.

    ``pt_serve.serve()`` looks up its module-global ``_Handler`` when it builds
    the server, so swapping it here adds routes without editing the vendored file
    (which stays byte-identical to the front-end repo's copy).
    """
    from urllib.parse import parse_qs

    class _LoggingHandler(base):  # type: ignore[misc, valid-type]
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if len(path) > 1:
                path = path.rstrip("/") or "/"

            if path == "/logs":
                self._send(200, LOGS_PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/log":
                since = 0
                if "?" in self.path:
                    raw = (parse_qs(self.path.split("?", 1)[1]).get("since") or ["0"])[0]
                    try:
                        since = int(raw)
                    except ValueError:
                        since = 0

                lines, next_index = buffer.snapshot(since)
                self._json({"lines": lines, "next": next_index})
                return

            return super().do_GET()

    return _LoggingHandler


def install_log_routes(pt_serve: Any, buffer: LogBuffer) -> Any:
    """Point pt_serve at a handler that also serves /logs and /api/log."""
    handler = make_logging_handler(getattr(pt_serve, "_Handler"), buffer)
    pt_serve._Handler = handler
    return handler


# ---------------------------------------------------------------------------
# Front end: polling transport
# ---------------------------------------------------------------------------
# The streaming answer rides one long-lived SSE response. A proxy that buffers
# responses without Transfer-Encoding: chunked (Cloudflare's quick tunnels do)
# holds every frame until the run ends, so the page shows nothing while the
# backend is fine. These routes carry the same information over ordinary
# request/response pairs, which no proxy can strand.
STAGE_KEYS: Sequence[str] = (
    "entities",
    "cpt",
    "hcpcs",
    "decision",
    "validation",
    "mrf",
    "answer",
)

STAGE_TITLES: Sequence[str] = (
    "Entity extraction",
    "CPT retrieval",
    "HCPCS retrieval",
    "Code decision",
    "Code validation",
    "MRF price filter",
    "Final answer",
)

# Mirrors pt_serve's parsing so the polling and streaming paths agree.
# tests/test_colab_run.py asserts this against pt_serve._stages_from_log.
_STEP_RE = re.compile(r"^STEP\s+(\d)/7\s+[—-]\s+(.*)$")
_TIME_RE = re.compile(r"^(.*?)\s+(?:completed\s+)?in\s+([\d.]+)s\s*$")
_MATCH_RE = re.compile(r"^MRF rows matched:\s*(\S+)")

_TIMING_PHRASES = {
    "categorization": "entities",
    "cpt retrieval": "cpt",
    "hcpcs retrieval": "hcpcs",
    "decision": "decision",
    "validated": "validation",
    "mrf filtering": "mrf",
    "answer generation": "answer",
}


def blank_stages() -> List[Dict[str, Any]]:
    """The seven stage rows, all pending."""
    return [
        {
            "n": i + 1,
            "key": STAGE_KEYS[i],
            "title": STAGE_TITLES[i],
            "state": "pending",
            "seconds": None,
            "detail": None,
        }
        for i in range(len(STAGE_KEYS))
    ]


def stage_rows(lines: Sequence[str], done: bool = False) -> List[Dict[str, Any]]:
    """Derive the seven stage rows from a run's log lines.

    Unlike pt_serve's ``_stages_from_log`` (which finalises everything to "done"),
    a stage still being worked on stays ``active`` until ``done`` is True — so the
    UI can show a step in progress rather than ticking it early.
    """
    stages = blank_stages()
    started = False

    for raw in lines:
        line = str(raw).strip()

        match = _STEP_RE.match(line)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(stages):
                stages[index]["state"] = "active"
                stages[index]["detail"] = match.group(2).strip()
                started = True
            continue

        match = _TIME_RE.match(line)
        if match:
            phrase = match.group(1).strip().lower()
            for needle, key in _TIMING_PHRASES.items():
                if needle in phrase:
                    index = STAGE_KEYS.index(key)
                    stages[index]["seconds"] = float(match.group(2))
                    stages[index]["state"] = "done"
                    started = True
                    break
            continue

        match = _MATCH_RE.match(line)
        if match:
            stages[STAGE_KEYS.index("mrf")]["detail"] = f"{match.group(1)} rows matched"
            started = True

    if done and started:
        for stage in stages:
            if stage["state"] in ("active", "pending"):
                stage["state"] = "done"

    return stages


@dataclass
class RunRecord:
    """One orchestration run, pollable while it executes."""

    id: str
    query: str
    state: str = "running"           # running | done | error
    result: Any = None               # the front-end payload once finished
    error: Optional[str] = None
    seconds: Optional[float] = None
    lines: List[str] = field(default_factory=list)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def all_lines(self) -> List[str]:
        with self._lock:
            return list(self.lines)

    def as_dict(self, since: int = 0) -> Dict[str, Any]:
        lines = self.all_lines()
        since = max(0, min(since, len(lines)))

        return {
            "run": self.id,
            "query": self.query,
            "state": self.state,
            "lines": lines[since:],
            "next": len(lines),
            "stages": stage_rows(lines, done=self.state != "running"),
            "seconds": self.seconds,
            "error": self.error,
            "result": self.result if self.state == "done" else None,
        }


class RunManager:
    """Start runs in the background and let a client poll them.

    Replaces the SSE transport's job without touching it: ``/api/stream`` and
    ``/api/query`` keep working exactly as before. Runs are serialised, because
    the pipeline juggles ``sys.stdout`` and one GPU.
    """

    def __init__(
        self,
        orchestration: Callable[..., Any],
        payload_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        max_runs: int = 25,
    ) -> None:
        self._orchestration = orchestration
        self._payload_fn = payload_fn
        self._max_runs = max_runs
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._counter = 0
        self._active: Optional[RunRecord] = None

    # -- lifecycle ------------------------------------------------------
    def start(self, query: str) -> RunRecord:
        with self._lock:
            self._counter += 1
            record = RunRecord(id=f"r{self._counter}", query=query)
            self._runs[record.id] = record
            self._evict_locked(keep=record.id)

        record.thread = threading.Thread(
            target=self._execute,
            args=(record,),
            name=f"pt-run-{record.id}",
            daemon=True,
        )
        record.thread.start()

        return record

    def _evict_locked(self, keep: str) -> None:
        while len(self._runs) > self._max_runs:
            oldest = next(iter(self._runs))
            if oldest == keep or self._runs[oldest].state == "running":
                return
            del self._runs[oldest]

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def _execute(self, record: RunRecord) -> None:
        with self._lock:
            self._active = record

        start = time.perf_counter()

        try:
            with self._run_lock:
                raw = self._orchestration(record.query)

            record.seconds = time.perf_counter() - start

            if self._payload_fn is not None:
                try:
                    record.result = self._payload_fn(record, raw)
                except Exception as error:
                    # Don't lose a completed run to a projection bug: hand back the
                    # raw result and say so in the log.
                    record.append(
                        f"[payload projection failed: {type(error).__name__}: {error}]"
                    )
                    record.result = raw
            else:
                record.result = raw

            record.state = "done"
        except Exception as error:  # surface it to the poller, don't kill the thread
            record.error = f"{type(error).__name__}: {error}"
            record.state = "error"
            record.append(traceback.format_exc())
        finally:
            if record.seconds is None:
                record.seconds = time.perf_counter() - start
            with self._lock:
                if self._active is record:
                    self._active = None

    # -- log capture ----------------------------------------------------
    def record_line(self, line: str) -> None:
        """Feed one pipeline log line into the active run, if any."""
        with self._lock:
            record = self._active

        if record is not None:
            record.append(line)


def make_extended_handler(
    base: Any,
    buffer: Optional[LogBuffer] = None,
    run_manager: Optional[RunManager] = None,
) -> Any:
    """Subclass pt_serve's handler to add /logs, /api/log, /api/run, /api/poll.

    ``pt_serve.serve()`` resolves its module-global ``_Handler`` when it builds
    the server and dispatches every route through ``self._...``, so subclassing
    adds routes without editing the vendored file — which stays byte-identical to
    the front-end repo's copy.
    """
    from urllib.parse import parse_qs

    class _ExtendedHandler(base):  # type: ignore[misc, valid-type]
        def _query_params(self) -> Dict[str, List[str]]:
            if "?" not in self.path:
                return {}
            return parse_qs(self.path.split("?", 1)[1])

        def _read_json(self) -> Dict[str, Any]:
            # Deliberately narrow: a blanket except here once hid a NameError for
            # half an hour by turning every POST body into {}.
            try:
                length = int(self.headers.get("content-length") or 0)
            except (TypeError, ValueError):
                return {}

            raw = self.rfile.read(length) if length > 0 else b""

            if not raw:
                return {}

            try:
                parsed = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                return {}

            return parsed if isinstance(parsed, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if len(path) > 1:
                path = path.rstrip("/") or "/"

            if path == "/logs" and buffer is not None:
                self._send(200, LOGS_PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/log" and buffer is not None:
                raw = (self._query_params().get("since") or ["0"])[0]
                try:
                    since = int(raw)
                except ValueError:
                    since = 0

                lines, next_index = buffer.snapshot(since)
                self._json({"lines": lines, "next": next_index})
                return

            if path == "/api/run" and run_manager is not None:
                query = (self._query_params().get("q") or [""])[0]
                self._start_run(query)
                return

            if path == "/api/poll" and run_manager is not None:
                self._poll(run_manager)
                return

            return super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"

            if path == "/api/run" and run_manager is not None:
                self._start_run(str(self._read_json().get("query") or ""))
                return

            return super().do_POST()

        def _start_run(self, query: str) -> None:
            if not query.strip():
                self._json({"error": "empty query"}, 400)
                return

            record = run_manager.start(query)
            self._json({"run": record.id, "state": record.state, "query": record.query})

        def _poll(self, manager: RunManager) -> None:
            params = self._query_params()
            run_id = (params.get("run") or [""])[0]
            raw = (params.get("since") or ["0"])[0]

            try:
                since = int(raw)
            except ValueError:
                since = 0

            record = manager.get(run_id)
            if record is None:
                self._json({"error": f"unknown run {run_id!r}"}, 404)
                return

            self._json(record.as_dict(since))

    return _ExtendedHandler


def payload_projection(pt_serve: Any) -> Callable[[RunRecord, Any], Dict[str, Any]]:
    """Wrap ``pt_serve.to_payload`` so a polled run returns the same shape as a streamed one.

    The client consumes one payload contract regardless of transport, so this
    must produce exactly what ``/api/stream`` would have sent.
    """

    def project(record: RunRecord, raw: Any) -> Dict[str, Any]:
        lines = record.all_lines()

        return pt_serve.to_payload(
            query=record.query,
            result=raw,
            run_log="\n".join(lines),
            stages=stage_rows(lines, done=True),
            total_seconds=record.seconds,
        )

    return project


def install_routes(
    pt_serve: Any,
    buffer: Optional[LogBuffer] = None,
    run_manager: Optional[RunManager] = None,
) -> Any:
    """Point pt_serve at a handler that also serves the log and polling routes."""
    handler = make_extended_handler(getattr(pt_serve, "_Handler"), buffer, run_manager)
    pt_serve._Handler = handler
    return handler


# Back-compat alias for the log-only name.
make_logging_handler = make_extended_handler
install_log_routes = install_routes


# ---------------------------------------------------------------------------
# Front end: public tunnels
# ---------------------------------------------------------------------------
# cloudflared prints its quick-tunnel URL to its own log.
_TRYCLOUDFLARE_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")

CLOUDFLARED_RELEASE = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-{system}-{arch}"
)


def _cloudflared_dest() -> str:
    base = COLAB_WORKDIR if in_colab() else os.path.join(_REPO_ROOT, ".cache")
    return os.path.join(base, "cloudflared")


def _cloudflared_release_url() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    arch = "arm64" if machine in {"aarch64", "arm64"} else "amd64"

    if system == "darwin":
        # macOS ships a .tgz rather than a bare binary.
        return (
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            f"cloudflared-darwin-{arch}.tgz"
        )

    return CLOUDFLARED_RELEASE.format(system="linux", arch=arch)


def ensure_cloudflared(dest: Optional[str] = None, verbose: bool = True) -> str:
    """Return a usable ``cloudflared`` binary, downloading it if necessary.

    Colab does not ship cloudflared; the static release binary is a single file,
    so this needs no package manager.
    """
    found = shutil.which("cloudflared")
    if found:
        return found

    dest = dest or _cloudflared_dest()

    if os.path.isfile(dest) and os.access(dest, os.X_OK):
        return dest

    url = _cloudflared_release_url()

    if url.endswith(".tgz"):
        raise RuntimeError(
            "Automatic cloudflared download is only wired up for Linux. On macOS "
            "install it with `brew install cloudflared` and re-run."
        )

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if verbose:
        print(f"Downloading cloudflared -> {dest}")

    urllib.request.urlretrieve(url, dest)
    os.chmod(dest, 0o755)

    return dest


@dataclass
class Tunnel:
    """A public URL forwarding to the local UI server."""

    url: str
    provider: str
    _stop: Optional[Callable[[], None]] = field(default=None, repr=False)

    def stop(self) -> None:
        if self._stop is not None:
            try:
                self._stop()
            finally:
                self._stop = None

    def __repr__(self) -> str:
        return f"<Tunnel provider={self.provider} url={self.url}>"


def start_cloudflare_tunnel(
    port: int,
    host: str = "127.0.0.1",
    timeout: int = 60,
    binary: Optional[str] = None,
    verbose: bool = True,
) -> Tunnel:
    """Start a Cloudflare quick tunnel (no account; random ``trycloudflare.com`` URL)."""
    binary = binary or ensure_cloudflared(verbose=verbose)

    command = [binary, "tunnel", "--url", f"http://{host}:{port}", "--no-autoupdate"]

    if verbose:
        print("Starting Cloudflare quick tunnel...")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url: Optional[str] = None
    seen: List[str] = []
    deadline = time.time() + timeout

    while time.time() < deadline:
        line = process.stdout.readline() if process.stdout else ""

        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue

        seen.append(line.rstrip())
        match = _TRYCLOUDFLARE_RE.search(line)
        if match:
            url = match.group(0)
            break

    if not url:
        process.terminate()
        raise RuntimeError(
            f"cloudflared did not publish a URL within {timeout}s. Output:\n"
            + "\n".join(seen[-20:])
        )

    def _stop() -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    return Tunnel(url=url, provider="cloudflare", _stop=_stop)


def start_ngrok_tunnel(port: int, token: Optional[str] = None, **kwargs: Any) -> Tunnel:
    """Start an ngrok tunnel (needs ``pip install pyngrok`` and an authtoken)."""
    try:
        from pyngrok import ngrok
    except ImportError as error:
        raise RuntimeError(
            "pyngrok is not installed. Run `pip install pyngrok` and provide an "
            "authtoken (ngrok config, or the NGROK_AUTHTOKEN env var)."
        ) from error

    token = token or os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    tunnel = ngrok.connect(port, "http")

    def _stop() -> None:
        try:
            ngrok.disconnect(tunnel.public_url)
        except Exception:
            pass

    return Tunnel(url=str(tunnel.public_url), provider="ngrok", _stop=_stop)


def start_tunnel(
    port: int,
    provider: str = "auto",
    host: str = "127.0.0.1",
    verbose: bool = True,
    **kwargs: Any,
) -> Optional[Tunnel]:
    """Start a public tunnel for ``port``.

    ``provider``: ``cloudflare`` (default, no account needed), ``ngrok``,
    ``none``/``off`` (skip), or ``auto`` (== cloudflare).
    """
    provider = (provider or "").strip().lower()

    if provider in {"", "none", "off", "no"}:
        return None

    if provider == "auto":
        provider = "cloudflare"

    if provider == "cloudflare":
        return start_cloudflare_tunnel(port, host=host, verbose=verbose, **kwargs)

    if provider == "ngrok":
        return start_ngrok_tunnel(port, **kwargs)

    raise ValueError(
        f"Unknown tunnel provider {provider!r}. Use 'cloudflare', 'ngrok' or 'none'."
    )


# ---------------------------------------------------------------------------
# Front end: serving
# ---------------------------------------------------------------------------
@dataclass
class UIServer:
    """The UI server plus (optionally) its public tunnel."""

    server: Any
    local_url: str
    page: str = "patient"
    tunnel: Optional[Tunnel] = None

    @property
    def url(self) -> str:
        """The best URL to hand out — public if tunnelled, local otherwise."""
        return self.tunnel.url if self.tunnel is not None else self.local_url

    @property
    def page_url(self) -> str:
        """Direct link to the requested page (``/`` = chat, ``/pipeline`` = dashboard)."""
        base = self.url.rstrip("/")
        return base + ("/pipeline" if self.page == "dashboard" else "/")

    @property
    def pages(self) -> Dict[str, str]:
        base = self.url.rstrip("/")
        return {
            "patient": base + "/",
            "dashboard": base + "/pipeline",
        }

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
        try:
            self.server.stop()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<UIServer page={self.page} url={self.page_url}>"


def _display_iframe(ui: UIServer, height: int = 1180, width: str = "100%") -> None:
    """Render the UI inline (Colab kernel-port iframe, else a plain iframe)."""
    path = "/pipeline" if ui.page == "dashboard" else "/"

    try:
        from google.colab import output  # type: ignore

        try:
            output.serve_kernel_port_as_iframe(
                ui.server.port, path=path, height=height, width=width
            )
            return
        except TypeError:
            output.serve_kernel_port_as_iframe(ui.server.port, height=height, width=width)
            return
    except Exception:
        pass

    try:
        from IPython.display import HTML, display  # type: ignore

        display(
            HTML(
                f'<iframe src="{ui.page_url}" width="{width}" height="{height}" '
                'style="border:1px solid #e4e4e1;border-radius:10px;background:#fff">'
                "</iframe>"
            )
        )
    except Exception:
        print(f"UI: {ui.page_url}")


def serve_ui(
    pipeline: Optional[HealthcarePricingPipeline] = None,
    page: str = "patient",
    port: int = 8000,
    host: str = "127.0.0.1",
    tunnel: str = "cloudflare",
    html_path: Optional[str] = None,
    patient_html_path: Optional[str] = None,
    display: Optional[bool] = None,
    height: int = 1180,
    width: str = "100%",
    logs: bool = True,
    log_to_notebook: bool = True,
    polling: bool = True,
    verbose: bool = True,
    **setup_kwargs: Any,
) -> UIServer:
    """Serve the TypeScript front end against this backend, optionally tunnelled.

    Wires ``pipeline.run`` in as the orchestration callable — the server tee's
    its stdout and parses the pipeline's own log lines, so the pages track the
    real backend rather than a reimplementation of it.

    ``polling=True`` adds ``/api/run`` + ``/api/poll``, a request/response
    transport carrying the same information as ``/api/stream``. The built client
    prefers it, because a proxy that buffers the SSE response (Cloudflare quick
    tunnels: no ``Transfer-Encoding: chunked``) strands every frame until the run
    ends, leaving the page blank while the backend works.

    ``logs=True`` adds a backend log pane at ``/logs`` (backed by ``/api/log``,
    a plain polling endpoint), and installs a printer that records every pipeline
    line into it. ``log_to_notebook=True`` also echoes those lines into the cell
    output, so you can watch a run without the browser — pt_serve otherwise only
    mirrors them to the page.

    Returns a `UIServer`; call ``.stop()`` to end the server and the tunnel.
    """
    if pipeline is None:
        pipeline = setup(**setup_kwargs)

    missing = [name for name, ok in check_ui_assets().items() if not ok]

    if missing:
        raise FileNotFoundError(
            f"Front-end assets missing from {UI_DIR}: {missing}. Rebuild them in "
            "the front-end repo (`npm run build`) and copy pt_patient.html, "
            "pt_frontend.html and pt_serve.py back into ui/."
        )

    pt_serve = import_pt_serve()

    buffer = LogBuffer() if logs else None
    run_manager = (
        RunManager(pipeline.run, payload_projection(pt_serve)) if polling else None
    )

    # The pipeline's printer is the one hook that sees every log line before the
    # tee does, so it is what the pane, the notebook and the poller all read from.
    pipeline.printer = log_printer(
        buffer,
        sys.__stdout__ if log_to_notebook else None,
        run_manager.record_line if run_manager is not None else None,
        getattr(pt_serve, "_Tee", None),
    )

    if buffer is not None or run_manager is not None:
        install_routes(pt_serve, buffer, run_manager)

    handle = pt_serve.serve(
        orchestration=pipeline.run,
        html_path=html_path or DASHBOARD_HTML,
        patient_html_path=patient_html_path or PATIENT_HTML,
        mrf_data=pipeline.mrf_data,
        client=getattr(pipeline.retriever, "client", None),
        top_k=pipeline.top_k,
        port=port,
        host=host,
    )

    public = start_tunnel(handle.port, provider=tunnel, host=host, verbose=verbose)

    ui = UIServer(server=handle, local_url=handle.url, page=page, tunnel=public)

    if verbose:
        print(f"\nLocal      : {ui.local_url}")
        if public is not None:
            print(f"Public     : {ui.page_url}   (via {public.provider})")
            print(
                "             ^ public and unauthenticated - anyone with this URL can\n"
                "               run queries on this pipeline. ui.stop() shuts it down."
            )
        else:
            print(f"Page       : {ui.page_url}")
        print(
            f"Pages      : patient={ui.pages['patient']}  "
            f"dashboard={ui.pages['dashboard']}"
        )
        if buffer is not None:
            base = ui.url.rstrip("/")
            print(f"Logs       : {base}/logs")
            print(
                "             ^ live backend log over plain polling, so it still works\n"
                "               when a proxy buffers the SSE answer stream."
            )

    if display is None:
        display = in_colab()

    if display and public is None:
        # Inside Colab the kernel-port iframe is the cheapest path. With a tunnel
        # the public URL is the point, so leave the cell output alone.
        _display_iframe(ui, height=height, width=width)

    return ui


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colab_run.py",
        description=(
            "Run the HPT prototype from a Colab cell or as a script. With no "
            "--query it runs the notebook's validation queries; with --serve-ui "
            "it serves the TypeScript front end (optionally tunnelled)."
        ),
    )

    parser.add_argument("queries", nargs="*", help="Queries to run.")
    parser.add_argument("--query", "-q", action="append", dest="extra_queries", help="Add a query (repeatable).")

    parser.add_argument("--install", action="store_true", help="pip install the runtime packages first.")
    parser.add_argument("--no-mount", action="store_true", help="Skip drive.mount().")
    parser.add_argument("--no-stage", action="store_true", help="Skip copying/extracting the DB snapshot.")

    parser.add_argument(
        "--mrf-csv",
        default=None,
        help=f"Default: PT_MRF_CSV from your .env, else {DRIVE_MRF_CSV}",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help=f"Default: PT_QDRANT_PATH from your .env, else {COLAB_DB_DIR}",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env (default: search the CWD, /content and the repo root).",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-hf-login", action="store_true", help="Skip huggingface_hub.login().")
    parser.add_argument("--json", action="store_true", help="Print each answer as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step logging.")

    ui = parser.add_argument_group("front end")
    ui.add_argument("--serve-ui", action="store_true", help="Serve the TypeScript UI against this backend.")
    ui.add_argument("--ui-page", choices=["patient", "dashboard"], default="patient")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument(
        "--tunnel",
        default="cloudflare",
        choices=["cloudflare", "ngrok", "none"],
        help="Public tunnel provider for --serve-ui (default: cloudflare).",
    )
    ui.add_argument(
        "--serve-once",
        action="store_true",
        help="Serve, run the queries, then exit instead of blocking (kills the tunnel).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    explicit: List[str] = list(args.queries)
    if args.extra_queries:
        explicit.extend(args.extra_queries)

    if explicit:
        queries = explicit
    elif args.serve_ui:
        queries = []          # serving is the job; don't burn three runs first
    else:
        queries = list(DEFAULT_QUERIES)

    pipeline = setup(
        install=args.install,
        mount=not args.no_mount,
        stage=not args.no_stage,
        device=args.device,
        top_k=args.top_k,
        mrf_csv=args.mrf_csv,
        qdrant_path=args.qdrant_path,
        env_file=args.env_file,
    )

    ui: Optional[UIServer] = None

    if args.serve_ui:
        ui = serve_ui(
            pipeline=pipeline,
            page=args.ui_page,
            port=args.port,
            host=args.host,
            tunnel=args.tunnel,
            display=in_colab(),
        )

    for query in queries:
        result = pipeline.run(query, verbose=not args.quiet)

        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "query": query,
                        "categorized": result["categorized"],
                        "output": result["output"],
                        "match_count": result["code_plausible"].get("match_count"),
                        "answer": result["answer"],
                        "total_seconds": result["total_seconds"],
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print("\n>>> FINAL ANSWER:\n" + str(result["answer"]))

    if ui is not None and not args.serve_once:
        print("\nServing. Interrupt to stop (the tunnel dies with this process).")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print()
        finally:
            ui.stop()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
