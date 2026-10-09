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
import os
import platform
import re
import shutil
import subprocess
import sys
import time
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
    provider = (provider or "auto").strip().lower()

    if provider in {"none", "off", "no", ""}:
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
    verbose: bool = True,
    **setup_kwargs: Any,
) -> UIServer:
    """Serve the TypeScript front end against this backend, optionally tunnelled.

    Wires ``pipeline.run`` in as the orchestration callable — the server tee's
    its stdout and parses the pipeline's own log lines, so the pages track the
    real backend rather than a reimplementation of it.

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
