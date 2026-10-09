"""Command-line entry point for the HPT prototype.

    pt-healthcare --query "What might a diagnostic mammogram cost at UPMC Presbyterian?"
    python -m pt_healthcare --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Sequence

from . import config
from .data import get_qdrant_client, load_mrf_data, login_huggingface
from .llm import MedGemmaEngine
from .models import load_models
from .pipeline import HealthcarePricingPipeline
from .retrieval import CodeRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pt-healthcare",
        description=(
            "Healthcare Price Transparency prototype: extract entities, retrieve "
            "CPT/HCPCS candidates, filter a hospital MRF and answer a cost question."
        ),
    )

    parser.add_argument(
        "--query",
        "-q",
        help="The patient question to answer.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start a REPL and answer questions until you type 'exit'.",
    )

    parser.add_argument(
        "--mrf-csv",
        default=None,
        help="Path to the cleaned MRF CSV (default: PT_MRF_CSV, else config default).",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Local Qdrant store directory (default: PT_QDRANT_PATH, else config default).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Candidates retrieved per code family (default: PT_TOP_K).",
    )
    parser.add_argument("--device", default="cuda", help="Device for MedGemma (e.g. cuda, cpu).")

    parser.add_argument(
        "--no-hf-login",
        action="store_true",
        help="Skip huggingface_hub.login (use when the models are already cached/unauth).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step logging.")

    return parser


def _json_default(value: Any) -> Any:
    """Make pandas objects / numpy scalars JSON-serializable."""
    try:
        import pandas as pd  # noqa: F401

        if hasattr(value, "to_dict"):
            return value.to_dict(orient="records")
    except Exception:  # pragma: no cover - defensive
        pass

    return str(value)


def run_once(pipeline: HealthcarePricingPipeline, query: str, verbose: bool, as_json: bool) -> Dict[str, Any]:
    """Run one query and emit its result in the requested format."""
    result = pipeline.run(query, verbose=verbose)

    if as_json:
        print(
            json.dumps(
                {
                    "query": query,
                    "categorized": result["categorized"],
                    "output": result["output"],
                    "answer": result["answer"],
                    "match_count": result["code_plausible"].get("match_count"),
                    "total_seconds": result["total_seconds"],
                },
                indent=2,
                default=_json_default,
            )
        )

    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.query and not args.interactive:
        build_parser().print_help()
        return 2

    # Honour a .env (and any PT_* overrides) before resolving any path.
    config.load_dotenv_files()
    config.refresh()

    mrf_csv = args.mrf_csv or config.mrf_csv_path()
    qdrant_path = args.qdrant_path or config.qdrant_dir()
    top_k = config.top_k() if args.top_k is None else args.top_k

    if not args.no_hf_login:
        login_huggingface()

    pipe, embed_model = load_models(device=args.device)

    client = get_qdrant_client(qdrant_path)
    mrf_data = load_mrf_data(mrf_csv)

    retriever = CodeRetriever(embed_model, client, top_k=top_k)
    engine = MedGemmaEngine(pipe)

    pipeline = HealthcarePricingPipeline(
        retriever=retriever,
        engine=engine,
        mrf_data=mrf_data,
        top_k=top_k,
    )

    if args.query:
        run_once(pipeline, args.query, verbose=not args.quiet, as_json=args.json)

    if args.interactive:
        print("Enter a question (or 'exit' to quit).")
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query or query.lower() in {"exit", "quit"}:
                break

            run_once(pipeline, query, verbose=not args.quiet, as_json=args.json)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
