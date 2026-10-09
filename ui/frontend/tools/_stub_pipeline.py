"""
A stand-in for the notebook's pipeline, used only by the test-suite.

It reproduces Copy_of_Try_PT-3.ipynb's *observable behaviour* — the same log
lines, the same return-dict shape, the same `repr()` dumps — but without
MedGemma, Qdrant or the 2.4 GB MRF. That is enough to verify the server's
log-parsing and payload projection against the real contract.

It deliberately prints to stdout exactly as the notebook does, because that
stdout is the interface `pt_serve` reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEMO_PATH = Path(__file__).resolve().parent.parent / "src" / "demo.json"


class Frame:
    """The minimal pandas.DataFrame surface the server actually uses."""

    def __init__(self, rows):
        self._rows = [dict(r) for r in rows]

    def to_dict(self, orient="records"):
        if orient != "records":
            raise ValueError(orient)
        return [dict(r) for r in self._rows]

    def head(self, n):
        return Frame(self._rows[:n])

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return [r.get(key) for r in self._rows]


class Collection:
    def __init__(self, points_count):
        self.points_count = points_count


class Client:
    def get_collection(self, name):
        return Collection({"cpt_medte": 15927, "hcpcs_medte": 9154}.get(name, 0))


def _scenarios() -> list[dict]:
    return json.loads(DEMO_PATH.read_text(encoding="utf-8"))


def _pick(query: str) -> dict:
    q = (query or "").lower()
    scenarios = _scenarios()
    for s in scenarios:
        if q and any(w in q for w in s["query"].lower().split() if len(w) > 6):
            return s
    return scenarios[0]


TOP_K = 10
mrf_data = Frame(
    [row for s in _scenarios() for row in s["code_plausible"]["price_summary"]]
)
client = Client()


def orchestration(user_query, verbose=True):
    """Same signature, same log lines, same return shape as the notebook."""
    sc = _pick(user_query)
    times = {s["key"]: (s["seconds"] or 0.0) for s in sc["stages"]}
    log = print if verbose else (lambda *a, **k: None)

    # 1 — categorize
    log("=" * 80)
    log("STEP 1/7 — Extracting medical, hospital, insurer, and medication entities")
    log(f"User query: {user_query}")
    time.sleep(0.05)
    log("Raw categorizer output:")
    log(repr(sc["raw"]["categorization"]))
    log(f"Categorization completed in {times.get('entities', 0):.2f}s")

    # 2/3 — retrieval
    log("=" * 80)
    log("STEP 2/7 — Retrieving CPT candidates from Qdrant")
    time.sleep(0.05)
    log(f"CPT retrieval completed in {times.get('cpt', 0):.2f}s")

    log("=" * 80)
    log("STEP 3/7 — Retrieving HCPCS candidates from Qdrant")
    time.sleep(0.05)
    log(f"HCPCS retrieval completed in {times.get('hcpcs', 0):.2f}s")

    # 4 — decision
    log("=" * 80)
    log("STEP 4/7 — Asking decision model to select relevant candidate codes")
    time.sleep(0.05)
    log("Raw decision-model output:")
    log(repr(sc["raw"]["decision"]))
    log(f"Decision completed in {times.get('decision', 0):.2f}s")

    # 5 — validation
    log("=" * 80)
    log("STEP 5/7 — Validating selected codes against Qdrant candidates")
    log(f"Validated in {times.get('validation', 0):.2f}s")
    rejected = sc["rejected"]
    if rejected.get("cpt"):
        log(f"Rejected CPT codes not found in retrieval results: {rejected['cpt']!r}")
    if rejected.get("hcpcs"):
        log(f"Rejected HCPCS codes not found in retrieval results: {rejected['hcpcs']!r}")

    # 6 — MRF filter
    log("=" * 80)
    log("STEP 6/7 — Filtering UPMC MRF data by selected codes, payer, and hospital")
    log(f"MRF filtering completed in {times.get('mrf', 0):.2f}s")
    cp = sc["code_plausible"]
    log(f"MRF rows matched: {cp['match_count']}")

    # 7 — answer
    log("=" * 80)
    log("STEP 7/7 — Generating final user-facing answer")
    log(f"Answer generation completed in {times.get('answer', 0):.2f}s")

    return {
        "categorized": dict(sc["categorized"]),
        "cpt_candidates": Frame(sc["cpt_candidates"]),
        "hcpcs_candidates": Frame(sc["hcpcs_candidates"]),
        "output": dict(sc["decision"]),
        "code_plausible": {
            "match_count": cp["match_count"],
            "filters_used": dict(cp["filters_used"]),
            "price_summary": Frame(cp["price_summary"]),
            "matches": Frame(cp["price_summary"]),
        },
        "answer": sc["answer"],
        "total_seconds": sc["total_seconds"],
    }


def exploding_orchestration(user_query, verbose=True):
    """Used to check that a pipeline exception is reported, not swallowed."""
    print("STEP 1/7 — Extracting medical, hospital, insurer, and medication entities")
    time.sleep(0.05)
    raise RuntimeError("CUDA out of memory")


if __name__ == "__main__":
    import sys

    out = orchestration(sys.argv[1] if len(sys.argv) > 1 else "mammogram")
    print("\nkeys:", sorted(out.keys()))
