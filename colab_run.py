"""Colab runner for the Healthcare Price Transparency (HPT) prototype.

Single entry point that keeps every Colab-specific bit (the ``!pip install``
cell, ``drive.mount``, the ``!cp`` + ``!tar -xzf`` snapshot staging, dotenv /
Colab-secret token lookup) next to the wiring for models, Qdrant and the MRF.

Two ways to use it in Colab
---------------------------

**As a script** (after cloning the repo)::

    !git clone https://github.com/adi274903/HPT_Prototype /content/HPT_Prototype
    %cd /content/HPT_Prototype
    !pip install -q -r requirements.txt
    !python colab_run.py

**As a module** (calls the same code, keeps the pipeline warm between queries)::

    %cd /content/HPT_Prototype
    import colab_run

    colab_run.mount_drive()               # drive.mount("/content/drive")
    colab_run.stage_db()                  # !cp ... && !tar -xzf ...
    pipeline = colab_run.setup(mount=False, stage=False)

    result = colab_run.run(
        "What might a diagnostic mammogram cost at UPMC Presbyterian "
        "with UPMC Health Plan?",
        pipeline,
    )

Nothing here is imported by the ``pt_healthcare`` package, so the package still
imports cleanly on a machine without ``google.colab``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

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


def load_env(dotenv_path: str = ".env") -> None:
    """Load a ``.env`` file if present (the notebook called ``load_dotenv()``)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv absent — rely on the environment
        return

    load_dotenv(dotenv_path)


def hf_token() -> Optional[str]:
    """Resolve the Hugging Face token from env, ``.env``, or Colab secrets."""
    token = os.getenv("HUGGINGFACE_API_KEY")

    if token:
        return token

    try:
        from google.colab import userdata

        return userdata.get("HUGGINGFACE_API_KEY")
    except Exception:
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
# Wiring
# ---------------------------------------------------------------------------
def default_paths() -> Dict[str, str]:
    """The Colab paths the notebook used, as a dict."""
    return {
        "drive_mount_point": DRIVE_MOUNT_POINT,
        "db_archive": DRIVE_DB_ARCHIVE,
        "qdrant_path": COLAB_DB_DIR,
        "mrf_csv": DRIVE_MRF_CSV,
    }


def build_pipeline(
    mrf_csv: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    device: str = "cuda",
    top_k: Optional[int] = None,
    login: bool = True,
    token: Optional[str] = None,
) -> HealthcarePricingPipeline:
    """Load models + data and return a ready-to-run pipeline."""
    load_env()

    if login:
        login_huggingface(token or hf_token())

    pipe, embed_model = load_models(device=device)

    client = get_qdrant_client(qdrant_path or COLAB_DB_DIR)
    mrf_data = load_mrf_data(mrf_csv or DRIVE_MRF_CSV)

    top_k = top_k or config.TOP_K

    return HealthcarePricingPipeline(
        retriever=CodeRetriever(embed_model, client, top_k=top_k),
        engine=MedGemmaEngine(pipe),
        mrf_data=mrf_data,
        top_k=top_k,
    )


def setup(
    install: bool = False,
    mount: bool = True,
    stage: bool = True,
    device: str = "cuda",
    top_k: Optional[int] = None,
    mrf_csv: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    token: Optional[str] = None,
    verbose: bool = True,
) -> HealthcarePricingPipeline:
    """One-call Colab bootstrap: install deps, mount Drive, stage the DB, wire models.

    Returns the pipeline so you can call it repeatedly::

        pipeline = setup()
        pipeline.run("...")
    """
    if install:
        install_dependencies()

    if mount:
        mount_drive()

    resolved_db = qdrant_path or COLAB_DB_DIR

    if stage:
        resolved_db = stage_db(verbose=verbose)

    return build_pipeline(
        mrf_csv=mrf_csv,
        qdrant_path=resolved_db,
        device=device,
        top_k=top_k,
        token=token,
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
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colab_run.py",
        description=(
            "Run the HPT prototype from a Colab cell or as a script. "
            "With no --query it runs the notebook's validation queries."
        ),
    )

    parser.add_argument("queries", nargs="*", help="Queries to run (defaults to the notebook's queries).")
    parser.add_argument("--query", "-q", action="append", dest="extra_queries", help="Add a query (repeatable).")

    parser.add_argument("--install", action="store_true", help="pip install the runtime packages first.")
    parser.add_argument("--no-mount", action="store_true", help="Skip drive.mount().")
    parser.add_argument("--no-stage", action="store_true", help="Skip copying/extracting the DB snapshot.")

    parser.add_argument("--mrf-csv", default=DRIVE_MRF_CSV)
    parser.add_argument("--qdrant-path", default=None, help=f"Default: {COLAB_DB_DIR}")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-hf-login", action="store_true", help="Skip huggingface_hub.login().")
    parser.add_argument("--json", action="store_true", help="Print each answer as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step logging.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    queries: List[str] = list(args.queries)
    if args.extra_queries:
        queries.extend(args.extra_queries)
    if not queries:
        queries = list(DEFAULT_QUERIES)

    pipeline = setup(
        install=args.install,
        mount=not args.no_mount,
        stage=not args.no_stage,
        device=args.device,
        top_k=args.top_k,
        mrf_csv=args.mrf_csv,
        qdrant_path=args.qdrant_path,
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

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
