"""Minimal end-to-end example — the notebook's final cells, as a script.

    python examples/run_pipeline.py \
        --mrf-csv /path/to/everyUPMCmrf_clean.csv \
        --qdrant-path /path/to/New_PT_DB

Requires the gated models (HUGGINGFACE_API_KEY) and a CUDA GPU.
"""

from __future__ import annotations

import argparse

from pt_healthcare import config
from pt_healthcare.data import get_qdrant_client, load_mrf_data, login_huggingface
from pt_healthcare.llm import MedGemmaEngine
from pt_healthcare.models import load_models
from pt_healthcare.pipeline import HealthcarePricingPipeline
from pt_healthcare.retrieval import CodeRetriever

DEFAULT_QUERIES = [
    "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?",
    "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?",
    "Best and cheapest hospital for me to get my mammogram done?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrf-csv", default=config.MRF_CSV_PATH)
    parser.add_argument("--qdrant-path", default=config.QDRANT_PATH)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "queries",
        nargs="*",
        default=None,
        help="Queries to run (defaults to the notebook's validation queries).",
    )
    args = parser.parse_args()

    # 1. Auth + models
    login_huggingface()
    pipe, embed_model = load_models(device=args.device)

    # 2. Data
    client = get_qdrant_client(args.qdrant_path)
    mrf_data = load_mrf_data(args.mrf_csv)

    # 3. Pipeline
    pipeline = HealthcarePricingPipeline(
        retriever=CodeRetriever(embed_model, client, top_k=args.top_k),
        engine=MedGemmaEngine(pipe),
        mrf_data=mrf_data,
        top_k=args.top_k,
    )

    for query in args.queries or DEFAULT_QUERIES:
        result = pipeline.run(query, verbose=True)
        print("\n>>> RESULT:", result["answer"])


if __name__ == "__main__":
    main()
