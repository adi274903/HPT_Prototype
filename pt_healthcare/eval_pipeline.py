"""Evaluate the shipped orchestrator end to end: Qdrant retrieval, LLM generation.

Two phases, because generation is the expensive half and scoring is the half you
re-run while you tune the metrics:

``generate``  runs every eval query through the real ``HealthcarePricingPipeline``
              (Qdrant retriever + the engine, vLLM in process by default) and
              writes one row per case with the results in new columns --
              ``generate_cpt``, ``generate_hcpcs``, ``generate_row_ids``,
              ``generate_rate``, ``generate_answer`` and friends. The ground truth
              columns are copied through untouched. JSONL keeps the types; the CSV
              is the same table flattened for a spreadsheet.

``score``     reads those rows and judges *the generated columns* against the key
              with the metrics the review asked for: Sorensen-Dice and Jaccard over
              CPT/HCPCS codes and over MRF row ids, precision/recall/F1 for the
              entity fields (hospital, insurer, medical, medication), negotiated-rate
              accuracy as an SQL result, and abstention / failure accounting.
              Nothing is regenerated, so a metric change costs seconds.

Colab, after ``colab_run.setup()`` has built the store::

    python -m pt_healthcare.eval_pipeline generate --qdrant-path /content/New_PT_DB \\
        --mrf everyUPMCmrf_clean.csv --out /content/eval_generations.jsonl
    python -m pt_healthcare.eval_pipeline score --rows /content/eval_generations.jsonl \\
        --mrf everyUPMCmrf_clean.csv --report /content/eval_report.json

Generation resumes: rows already in ``--out`` are skipped unless ``--fresh``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from . import config
from .eval_runner import (
    FIELD_ORDER,
    Observation,
    aggregate,
    final_answer,
    find_dataset_dir,
    find_targets,
    format_report,
    load_jsonl,
    score_observation,
)

# The generated side of the table. Everything here is new; nothing overwrites a key.
GENERATED_COLUMNS: Sequence[str] = (
    "generate_cpt",
    "generate_hcpcs",
    "generate_cpt_candidates",
    "generate_hcpcs_candidates",
    "generate_row_ids",
    "generate_matches",
    "generate_rate",
    "generate_cash",
    "generate_hospital",
    "generate_insurer",
    "generate_medical",
    "generate_medication",
    "generate_answer",
    "generate_intent",
    "generate_decision_recovered",
    "generate_seconds",
    "generate_error",
)

# Raw MRF column -> the names the scorer and the collapse key expect.
ROW_KEYS: Dict[str, Sequence[str]] = {
    "row_id": ("row_id",),
    "cpt": ("CPT", "cpt"),
    "hcpcs": ("HCPCS", "hcpcs"),
    "hospital": ("hospital_name", "hospital"),
    "payer": ("payer_name", "payer"),
    "plan": ("plan_name", "plan"),
    "setting": ("setting",),
    "rate": ("standard_charge|negotiated_dollar", "rate"),
    "cash": ("standard_charge|discounted_cash", "cash"),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _cell(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _money(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def describe_rows(frame: Any, limit: int = 50) -> List[Dict[str, Any]]:
    """The MRF rows a price lookup returned, flattened to the scorer's vocabulary."""
    if frame is None:
        return []
    if isinstance(frame, Mapping):  # a JSONL row, not a frame: column -> value(s)
        import pandas as pd

        frame = pd.DataFrame({key: value if isinstance(value, (list, tuple)) else [value]
                              for key, value in frame.items()})
    try:
        records = frame.head(limit).to_dict(orient="records")
    except AttributeError:
        records = list(frame)[:limit]
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        rows.append({
            "row_id": str(_cell(record, ROW_KEYS["row_id"])),
            "cpt": _cell(record, ROW_KEYS["cpt"]),
            "hcpcs": _cell(record, ROW_KEYS["hcpcs"]),
            "hospital": _cell(record, ROW_KEYS["hospital"]),
            "payer": _cell(record, ROW_KEYS["payer"]),
            "plan": _cell(record, ROW_KEYS["plan"]),
            "setting": _cell(record, ROW_KEYS["setting"]),
            "rate": _money(_cell(record, ROW_KEYS["rate"])),
            "cash": _money(_cell(record, ROW_KEYS["cash"])),
        })
    return rows


def _codes(frame: Any) -> List[str]:
    if frame is None:
        return []
    try:
        return [str(code) for code in frame["code"].tolist()]
    except (KeyError, TypeError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# price lookups for the answer key
# ---------------------------------------------------------------------------
class FramePricer:
    """``row_details``/``keyed_rates`` over the MRF frame the pipeline already holds.

    One vectorised scan restricted to the key's row ids, then dict hits. Used when
    the notebook has the 2.43 GB frame in memory and a 509 MB duckdb file would be a
    third copy of the same data.
    """

    def __init__(self, frame: Any, row_ids: Iterable[str]) -> None:
        self.frame = frame
        self._details: Dict[str, Dict[str, Any]] = {}
        self._scope({str(row_id) for row_id in row_ids if str(row_id)})

    def _scope(self, ids: Iterable[str]) -> None:
        pending = {str(i) for i in ids} - set(self._details)
        if not pending or "row_id" not in getattr(self.frame, "columns", []):
            return
        sample = self.frame["row_id"].head(1).tolist()
        wanted: Any = sorted(pending)
        if sample and not isinstance(sample[0], str):
            coerced: List[Any] = []
            for value in wanted:
                try:
                    coerced.append(int(value))
                except ValueError:
                    coerced.append(value)
            wanted = coerced
        subset = self.frame[self.frame["row_id"].isin(wanted)]
        for record in describe_rows(subset, limit=len(subset)):
            self._details[record["row_id"]] = record

    def row_details(self, row_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = [str(row_id) for row_id in row_ids]
        self._scope(ids)
        return [self._details[i] for i in ids if i in self._details]

    def keyed_rates(self, row_ids: Iterable[str]) -> "tuple[set, set]":
        details = self.row_details(row_ids)
        return ({row["rate"] for row in details if row.get("rate") is not None},
                {row["cash"] for row in details if row.get("cash") is not None})


def build_pricer(*, db: Optional[Any] = None, mrf_data: Any = None,
                 mrf_path: Optional[str] = None, row_ids: Iterable[str] = ()) -> Any:
    """duckdb when a database is given, else the frame (loaded if only a path is)."""
    if db:
        from .eval_runner import Pricer

        return Pricer(Path(db))
    if mrf_data is None:
        from .data import load_mrf_data

        mrf_data = load_mrf_data(mrf_path or config.mrf_csv_path())
    return FramePricer(mrf_data, row_ids)


# ---------------------------------------------------------------------------
# engine proxy: keep the raw completion of every step
# ---------------------------------------------------------------------------
class RecordingEngine:
    """Wraps ``MedGemmaEngine`` so the raw text of each step survives the run.

    That text is the generation data: not the parsed JSON, but what the model
    emitted, per step, with its own latency.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.calls: List[Dict[str, Any]] = []

    def _record(self, step: str, method: Callable[..., Any], *args: Any,
                **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            text = method(*args, **kwargs)
        except Exception as error:  # a dead step should not take the case with it
            self.calls.append({"step": step, "text": "", "ms": _ms(started),
                               "error": f"{type(error).__name__}: {error}"})
            raise
        self.calls.append({
            "step": step,
            "text": text if isinstance(text, str) else str(text),
            "ms": _ms(started),
        })
        return text

    def categorize(self, *args: Any, **kwargs: Any) -> Any:
        return self._record("categorize", self.engine.categorize, *args, **kwargs)

    def decide(self, *args: Any, **kwargs: Any) -> Any:
        return self._record("decide", self.engine.decide, *args, **kwargs)

    def answer(self, *args: Any, **kwargs: Any) -> Any:
        return self._record("answer", self.engine.answer, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)


# ---------------------------------------------------------------------------
# result -> row of generate_* columns
# ---------------------------------------------------------------------------
def generate_row(case: Mapping[str, Any], result: Mapping[str, Any],
                 calls: Sequence[Mapping[str, Any]],
                 error: Optional[str] = None) -> Dict[str, Any]:
    """One case, its key untouched, the run's output in ``generate_*`` columns."""
    output = result.get("output") if isinstance(result.get("output"), Mapping) else {}
    output = output or {}
    plausible = result.get("code_plausible")
    plausible = plausible if isinstance(plausible, Mapping) else {}
    categorized = result.get("categorized")
    categorized = categorized if isinstance(categorized, Mapping) else {}
    matches = describe_rows(plausible.get("matches"))

    row: Dict[str, Any] = dict(case)  # the key's columns stay exactly as published
    row.update({
        "generate_cpt": _as_list(output.get("cpt_list")),
        "generate_hcpcs": _as_list(output.get("hcpcs_list")),
        "generate_cpt_candidates": _codes(result.get("cpt_candidates")),
        "generate_hcpcs_candidates": _codes(result.get("hcpcs_candidates")),
        "generate_row_ids": [match["row_id"] for match in matches],
        "generate_matches": matches,
        "generate_rate": sorted({match["rate"] for match in matches
                                 if match.get("rate") is not None}),
        "generate_cash": sorted({match["cash"] for match in matches
                                 if match.get("cash") is not None}),
        "generate_hospital": _as_list(categorized.get("hospital")),
        "generate_insurer": _as_list(categorized.get("insurer")),
        "generate_medical": _as_list(categorized.get("medical")),
        "generate_medication": _as_list(categorized.get("medication")),
        "generate_answer": result.get("answer") or "",
        "generate_intent": result.get("intent") or "",
        "generate_decision_recovered": bool(result.get("decision_recovered")),
        "generate_seconds": round(float(result.get("total_seconds") or 0.0), 3),
        "generate_error": error or result.get("stopped") or "",
        "generate_calls": [dict(call) for call in calls],
        "generate_filters_used": plausible.get("filters_used"),
    })
    return row


def observation_from_row(row: Mapping[str, Any]) -> Observation:
    """Score a ``generate_*`` row without touching the pipeline or the model."""
    codes = _as_list(row.get("generate_cpt")) + _as_list(row.get("generate_hcpcs"))
    ranked = (_as_list(row.get("generate_cpt_candidates"))
              + _as_list(row.get("generate_hcpcs_candidates")))
    matches = row.get("generate_matches")
    matches = [match for match in matches if isinstance(match, Mapping)] \
        if isinstance(matches, (list, tuple)) else []
    candidate_set = set(ranked)
    answer_text = row.get("generate_answer") or ""
    return Observation(
        codes=codes,
        ranked=ranked,
        rows=matches,
        rows_substring=matches,
        text=final_answer(answer_text) if isinstance(answer_text, str) else "",
        categorized={field_name: _as_list(row.get(f"generate_{field_name}"))
                     for field_name in FIELD_ORDER},
        intent=(str(row["generate_intent"]).strip().lower()
                if row.get("generate_intent") else None),
        hallucinated=[code for code in codes if candidate_set and code not in candidate_set],
        latency={"total": round(float(row.get("generate_seconds") or 0.0) * 1000, 1)},
        errors=[str(row["generate_error"])] if row.get("generate_error") else [],
        extra={"filters_used": row.get("generate_filters_used"),
               "n_candidates": len(ranked)},
    )


# ---------------------------------------------------------------------------
# building the real pipeline
# ---------------------------------------------------------------------------
def build_pipeline(*, mrf_path: Optional[str] = None, qdrant_path: Optional[str] = None,
                   backend: Optional[str] = None, model: Optional[str] = None,
                   base_url: Optional[str] = None, device: Optional[str] = "cuda",
                   top_k: Optional[int] = None, gguf: Optional[str] = None,
                   mrf_data: Any = None, pipe: Any = None, embed_model: Any = None,
                   client: Any = None) -> "tuple[Any, RecordingEngine]":
    """Assemble ``HealthcarePricingPipeline`` the way the app does.

    vLLM in process unless asked otherwise: ``load_models(backend="vllm")`` returns
    the generator and the embedder together. Pass ``pipe``/``embed_model``/
    ``mrf_data`` to reuse what the notebook already loaded — one set of weights for
    the UI and the eval, not two.
    """
    from .data import get_qdrant_client, load_mrf_data
    from .llm import MedGemmaEngine
    from .models import load_models
    from .pipeline import HealthcarePricingPipeline
    from .retrieval import CodeRetriever

    if mrf_data is None:
        mrf_data = load_mrf_data(mrf_path or config.mrf_csv_path())
    if pipe is None or embed_model is None:
        loaded_pipe, loaded_embed = load_models(device=device or "cuda", backend=backend,
                                                gguf=gguf, pipe=pipe)
        pipe = pipe if pipe is not None else loaded_pipe
        embed_model = embed_model or loaded_embed
    if client is None:
        client = get_qdrant_client(qdrant_path or config.qdrant_dir())

    retriever = CodeRetriever(embed_model, client, top_k=top_k or config.top_k())
    engine = RecordingEngine(MedGemmaEngine(pipe))
    pipeline = HealthcarePricingPipeline(
        retriever=retriever, engine=engine, mrf_data=mrf_data,
        top_k=top_k or config.top_k(),
        printer=lambda _message: None,
        display_fn=lambda *_args, **_kwargs: None,
    )
    return pipeline, engine


# ---------------------------------------------------------------------------
# case selection
# ---------------------------------------------------------------------------
def select_cases(cases: Sequence[Mapping[str, Any]],
                 targets: Mapping[str, Mapping[str, Any]], *,
                 subtypes: Sequence[str] = (), per_subtype: int = 0,
                 limit: int = 0) -> List[Mapping[str, Any]]:
    wanted = {subtype.upper() for subtype in subtypes if subtype}
    picked: List[Mapping[str, Any]] = []
    seen: Dict[str, int] = {}
    for case in cases:
        subtype = str((targets.get(case["eval_id"]) or {}).get("subtype") or "").upper()
        if wanted and subtype not in wanted:
            continue
        if per_subtype:
            if seen.get(subtype, 0) >= per_subtype:
                continue
            seen[subtype] = seen.get(subtype, 0) + 1
        picked.append(case)
        if limit and len(picked) >= limit:
            break
    return picked


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """The same table, flattened: lists join with ``|``, matches stay JSON."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple)):
                    flat[key] = "|".join(str(item) for item in value)
                elif isinstance(value, Mapping):
                    flat[key] = json.dumps(value, default=str)
                else:
                    flat[key] = value
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# phase 1: generate
# ---------------------------------------------------------------------------
def generate(*, pipeline: Optional[Any] = None, engine: Optional[RecordingEngine] = None,
             dataset: Optional[Any] = None, out: Any = "eval_generations.jsonl",
             csv_out: Optional[Any] = None, mrf_path: Optional[str] = None,
             mrf_data: Any = None, qdrant_path: Optional[str] = None,
             backend: Optional[str] = None, model: Optional[str] = None,
             base_url: Optional[str] = None, device: Optional[str] = "cuda",
             top_k: Optional[int] = None, gguf: Optional[str] = None,
             subtypes: Sequence[str] = (), per_subtype: int = 0, limit: int = 0,
             fresh: bool = False, with_intent: bool = False,
             progress: Optional[Callable[[str], None]] = print) -> List[Dict[str, Any]]:
    """Run the orchestrator over the dataset and return the enriched rows."""
    dataset_dir = find_dataset_dir(dataset)
    cases = load_jsonl(dataset_dir / "pt_eval_v2.jsonl")
    targets = {record["eval_id"]: record for record in load_jsonl(find_targets(dataset_dir))}
    cases = select_cases(cases, targets, subtypes=subtypes,
                         per_subtype=per_subtype, limit=limit)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if not fresh and out_path.exists():
        rows = load_jsonl(out_path)
    done = {row["eval_id"] for row in rows}

    todo = [case for case in cases if case["eval_id"] not in done]
    if progress:
        skipped = len(cases) - len(todo)
        progress(f"{len(cases)} cases, {len(todo)} to generate"
                 + (f", {skipped} already on disk" if skipped else ""))

    if todo and pipeline is None:
        pipeline, engine = build_pipeline(
            mrf_path=mrf_path, qdrant_path=qdrant_path, backend=backend, model=model,
            base_url=base_url, device=device, top_k=top_k, gguf=gguf, mrf_data=mrf_data)
    if engine is None and pipeline is not None:
        # Reuse a pipeline the notebook already built (same Qdrant client, same
        # weights) — just swap in the recorder so the raw text is kept.
        live = getattr(pipeline, "engine", None)
        if live is not None and not isinstance(live, RecordingEngine):
            engine = RecordingEngine(live)
            pipeline.engine = engine
        else:
            engine = live

    intent_client = None
    if with_intent and todo:
        if progress:
            progress("--with-intent: one extra routing call per case "
                     "(this branch has no router stage)")
        from .eval_runner import build_client

        intent_client = build_client(backend or "vllm",
                                     model=model or config.llm_model_name(),
                                     base_url=base_url or config.llm_base_url())

    for index, case in enumerate(todo, start=1):
        calls: List[Dict[str, Any]] = []
        if engine is not None:
            engine.calls = calls
        started = time.perf_counter()
        error: Optional[str] = None
        try:
            result = pipeline.run(case["eval_query"], verbose=False)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            result = {"output": {}, "code_plausible": {}, "categorized": {},
                      "answer": "", "total_seconds": time.perf_counter() - started}
        if intent_client is not None:
            try:
                from .prompts import build_intent_prompt

                reply = intent_client.complete(build_intent_prompt(case["eval_query"]),
                                               max_tokens=8)
                text = (getattr(reply, "text", "") or "").strip()
                if text:
                    result = dict(result)
                    result["intent"] = text.split()[0].lower()
            except Exception:
                pass

        row = generate_row(case, result, calls, error=error)
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        if progress:
            progress(f"[{index}/{len(todo)}] {case['eval_id']} "
                     f"cpt={len(row['generate_cpt'])} hcpcs={len(row['generate_hcpcs'])} "
                     f"rows={len(row['generate_row_ids'])} {row['generate_seconds']}s"
                     + (f" ERROR {error}" if error else ""))

    if csv_out:
        _write_csv(rows, Path(csv_out))
    return rows


# ---------------------------------------------------------------------------
# phase 2: score the generated columns
# ---------------------------------------------------------------------------
def score(*, rows: Any, dataset: Optional[Any] = None, db: Optional[Any] = None,
          mrf_path: Optional[str] = None, mrf_data: Any = None,
          report_path: Optional[Any] = "eval_report.json",
          scores_out: Optional[Any] = "eval_scores.jsonl",
          progress: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Judge the ``generate_*`` columns against the key. No model, no Qdrant."""
    if not isinstance(rows, (list, tuple)):
        rows = load_jsonl(Path(rows))
    dataset_dir = find_dataset_dir(dataset)
    cases = {case["eval_id"]: case for case in load_jsonl(dataset_dir / "pt_eval_v2.jsonl")}
    targets = {record["eval_id"]: record for record in load_jsonl(find_targets(dataset_dir))}

    usable = [row for row in rows if row.get("eval_id") in cases]
    unknown = len(rows) - len(usable)
    if progress:
        progress(f"scoring {len(usable)} generated rows"
                 + (f" ({unknown} unknown eval_id ignored)" if unknown else ""))

    truth_ids = {str(row_id) for row in usable
                 for row_id in cases[row["eval_id"]].get("ground_truth_row_ids") or []}
    pricer = build_pricer(db=db, mrf_data=mrf_data, mrf_path=mrf_path, row_ids=truth_ids)

    if scores_out:
        Path(scores_out).unlink(missing_ok=True)  # scores are derived; never append twice

    records: List[Dict[str, Any]] = []
    for row in usable:
        case = cases[row["eval_id"]]
        record = score_observation(case, targets.get(row["eval_id"], {}),
                                   observation_from_row(row), pricer)
        # not a model error -- the answer still came out of a real decision, but the
        # decision was salvaged from a thought block that never emitted its JSON
        record["decision_recovered"] = bool(row.get("generate_decision_recovered"))
        records.append(record)
        if scores_out:
            path = Path(scores_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")

    meta = {"mode": "orchestrator", "retriever": "qdrant",
            "generated_rows": len(usable), "dataset": str(dataset_dir)}
    report = aggregate(records, meta)
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2, default=str),
                                     encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=None, help="eval dir, or path to pt_eval_v2.jsonl")
    parser.add_argument("--mrf", default=None, help="cleaned MRF csv (default: PT_MRF_CSV)")
    parser.add_argument("--db", default=None, help="mrf_eval.duckdb, for the key's rates")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the orchestrator over pt_eval_v2, then score its columns.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Qdrant retrieval + the LLM over every case")
    _common(gen)
    gen.add_argument("--qdrant-path", default=None)
    gen.add_argument("--backend", default=None, help="vllm (default), remote, llamacpp, hf")
    gen.add_argument("--model", default=None)
    gen.add_argument("--base-url", default=None)
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--gguf", default=None)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--subtypes", default="")
    gen.add_argument("--per-subtype", type=int, default=0)
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--fresh", action="store_true")
    gen.add_argument("--with-intent", action="store_true",
                     help="one extra routing call per case (this branch has no router)")
    gen.add_argument("--out", default="eval_generations.jsonl")
    gen.add_argument("--csv", default=None, help="also write the flattened table")
    gen.add_argument("--quiet", action="store_true")

    scr = sub.add_parser("score", help="metrics over the generate_* columns")
    _common(scr)
    scr.add_argument("--rows", default="eval_generations.jsonl")
    scr.add_argument("--report", default="eval_report.json")
    scr.add_argument("--scores", default="eval_scores.jsonl")
    scr.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    progress = None if getattr(args, "quiet", False) else print

    if args.command == "generate":
        generate(mrf_path=args.mrf, qdrant_path=args.qdrant_path, backend=args.backend,
                 model=args.model, base_url=args.base_url, device=args.device,
                 top_k=args.top_k, gguf=args.gguf, dataset=args.dataset, out=args.out,
                 csv_out=args.csv, subtypes=tuple(filter(None, args.subtypes.split(","))),
                 per_subtype=args.per_subtype, limit=args.limit, fresh=args.fresh,
                 with_intent=args.with_intent, progress=progress)
        return 0

    report = score(rows=args.rows, dataset=args.dataset, db=args.db, mrf_path=args.mrf,
                   report_path=args.report, scores_out=args.scores, progress=progress)
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
