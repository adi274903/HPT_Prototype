"""Tests for the orchestrator eval path: generate_* columns in, metrics out.

The fake pipeline here is a *perfect oracle*: it returns the answer key's own codes
and rows and the sidecar's own entities, so every metric must come out at exactly
1.0. That is the only way to prove the plumbing — if a column is dropped, renamed
or flattened wrongly, a 1.0 collapses and the test names the metric that moved. A
second subject drops codes on purpose, so the numbers are shown to be live.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pt_healthcare.eval_pipeline import (  # noqa: E402
    GENERATED_COLUMNS,
    FramePricer,
    _write_csv,
    generate,
    generate_row,
    observation_from_row,
    score,
)
from pt_healthcare.eval_runner import FIELD_ORDER, find_dataset_dir, load_jsonl  # noqa: E402

DB = REPO / ".cache" / "mrf_eval.duckdb"
DATASET = REPO / "eval"

pytestmark = pytest.mark.skipif(
    not DB.exists(), reason="needs .cache/mrf_eval.duckdb (scripts/prepare_eval_mrf.py)")


def truth_frame(row_ids):
    """The key's MRF rows straight from duckdb, wearing the pipeline's column names."""
    import duckdb

    ids = [int(value) for value in row_ids]
    if not ids:
        return None
    connection = duckdb.connect(str(DB), read_only=True)
    try:
        frame = connection.execute(
            "SELECT row_id, cpt, hcpcs, hospital, payer, rate, cash FROM rows_ext "
            f"WHERE row_id IN ({','.join('?' * len(ids))})", ids).df()
    finally:
        connection.close()
    return frame.rename(columns={
        "cpt": "CPT", "hcpcs": "HCPCS", "hospital": "hospital_name",
        "payer": "payer_name", "rate": "standard_charge|negotiated_dollar",
        "cash": "standard_charge|discounted_cash"})


class OraclePipeline:
    """Answers every query with the key — a perfect run, by construction."""

    def __init__(self, cases, targets, drop_codes=False):
        self.by_query = {case["eval_query"]: case for case in cases}
        self.targets = targets
        self.drop_codes = drop_codes

    def run(self, query, verbose=False):
        case = self.by_query[query]
        target = self.targets.get(case["eval_id"], {})
        rows = truth_frame(case["ground_truth_row_ids"])
        codes = list(case["ground_truth_code_MRF"])
        if self.drop_codes:
            codes = codes[:1]
        if rows is not None:
            best = rows.iloc[0]["standard_charge|negotiated_dollar"]
            answer = f"<unused94>thought\nweighing it<unused95>The negotiated rate is ${best:,.2f}."
        else:
            answer = "<unused94>thought\nnothing to price<unused95>No price is listed for that."
            # D2 codes nothing; D1 codes 28 colonoscopy procedures but has no rows.
            codes = codes if codes else []
        return {
            # the sidecar keeps canonical strings under "canonical", spans under "entities"
            "categorized": {field: list((target.get("canonical") or {}).get(field) or [])
                            for field in FIELD_ORDER},
            "cpt_candidates": None,
            "hcpcs_candidates": None,
            "output": {"cpt_list": codes, "hcpcs_list": []},
            "code_plausible": {
                "matches": rows,
                "match_count": 0 if rows is None else len(rows),
                "filters_used": {"cpt_codes": codes},
            },
            "answer": answer,
            "intent": target.get("intent"),
            "total_seconds": 0.25,
        }


def oracle(drop_codes=False):
    dataset = find_dataset_dir(DATASET)
    cases = load_jsonl(dataset / "pt_eval_v2.jsonl")
    targets = {record["eval_id"]: record
               for record in load_jsonl(dataset / "pt_eval_v2_targets.jsonl")}
    return OraclePipeline(cases, targets, drop_codes=drop_codes)


# ---------------------------------------------------------------------------
# the columns themselves
# ---------------------------------------------------------------------------
def test_generate_row_keeps_key_and_adds_columns():
    case = {"eval_id": "27132_000001_A1", "eval_query": "q",
            "ground_truth_codes": ["45378"], "ground_truth_code_MRF": ["45378"],
            "ground_truth_row_ids": ["6979391"],
            "ground_truth_hospital_name": "Upmc Somerset",
            "ground_truth_insurance_provider": "Highmark BCBS of PA Medicare Advantage"}
    result = {"output": {"cpt_list": ["45378"], "hcpcs_list": []},
              "categorized": {"medical": ["colonoscopy"], "hospital": ["Upmc Somerset"],
                              "insurer": ["Highmark"], "medication": []},
              "code_plausible": {"matches": {"row_id": [6979391], "CPT": ["45378"],
                                             "hospital_name": ["Upmc Somerset"],
                                             "payer_name": ["Highmark"],
                                             "standard_charge|negotiated_dollar": [2845.25],
                                             "standard_charge|discounted_cash": [None]}},
              "answer": "It costs $2,845.25.", "total_seconds": 1.5}
    row = generate_row(case, result, [{"step": "categorize", "text": "{}", "ms": 12.0}])

    for key, value in case.items():  # the key's columns are copied, never rewritten
        assert row[key] == value
    for column in GENERATED_COLUMNS:
        assert column in row, f"missing generated column {column}"

    assert row["generate_cpt"] == ["45378"]
    assert row["generate_row_ids"] == ["6979391"]
    assert row["generate_rate"] == [2845.25]
    assert row["generate_cash"] == []
    assert row["generate_hospital"] == ["Upmc Somerset"]
    assert row["generate_answer"] == "It costs $2,845.25."
    assert row["generate_seconds"] == 1.5


def test_observation_strips_the_thought_block():
    observation = observation_from_row({
        "generate_cpt": ["45378"],
        "generate_answer": "<unused94>thought\nabout it<unused95>The price is $1,240.40."})
    assert "thought" not in observation.text
    assert "1,240.40" in observation.text


# ---------------------------------------------------------------------------
# the metrics on those columns
# ---------------------------------------------------------------------------
def test_perfect_run_scores_one(tmp_path):
    rows = generate(pipeline=oracle(), dataset=DATASET, out=tmp_path / "gen.jsonl",
                    per_subtype=1, progress=None)
    assert len(rows) == 10, "one case per subtype"
    assert len({row["eval_id"] for row in rows}) == 10

    report = score(rows=rows, dataset=DATASET, db=DB,
                   report_path=tmp_path / "rep.json", scores_out=None, progress=None)

    assert report["codes"]["dice_mrf"] == 1.0
    assert report["codes"]["jaccard_mrf"] == 1.0
    assert report["rows"]["dice_raw"] == 1.0
    assert report["rates"]["headline_matches_key"] == 1.0
    assert report["ner"]["macro_f1"] == 1.0
    assert report["abstention"]["correct_silence"] == 1.0
    assert report["intent"]["measured"] is True
    assert report["intent"]["accuracy"] == 1.0
    assert (tmp_path / "rep.json").exists()


def test_dropping_codes_moves_the_metrics(tmp_path):
    rows = generate(pipeline=oracle(drop_codes=True), dataset=DATASET,
                    out=tmp_path / "gen.jsonl", per_subtype=1, progress=None)
    report = score(rows=rows, dataset=DATASET, db=DB, report_path=None,
                   scores_out=None, progress=None)
    assert report["codes"]["dice_mrf"] < 1.0


def test_generate_resumes_without_duplicating(tmp_path):
    out = tmp_path / "gen.jsonl"
    first = generate(pipeline=oracle(), dataset=DATASET, out=out, limit=3, progress=None)
    again = generate(pipeline=oracle(), dataset=DATASET, out=out, limit=3, progress=None)
    assert len(first) == len(again) == 3
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_csv_flattens_lists(tmp_path):
    path = tmp_path / "table.csv"
    _write_csv([{"eval_id": "a", "generate_cpt": ["1", "2"],
                 "generate_matches": [{"row_id": 1}]}], path)
    text = path.read_text(encoding="utf-8")
    assert "generate_cpt" in text and "1|2" in text


def test_score_ignores_rows_without_a_case():
    report = score(rows=[{"eval_id": "not_a_case", "generate_cpt": ["45378"]}],
                   dataset=DATASET, db=DB, report_path=None, scores_out=None, progress=None)
    assert report["n_cases"] == 0


def test_frame_pricer_matches_duckdb():
    import duckdb

    frame = truth_frame(["6979391", "2344710"])
    ids = [str(value) for value in frame["row_id"].tolist()]

    connection = duckdb.connect(str(DB), read_only=True)
    try:
        expected = {float(row[0]) for row in connection.execute(
            "SELECT rate FROM rows_ext WHERE row_id IN ("
            + ",".join("?" * len(ids)) + ")", [int(value) for value in ids]).fetchall()}
    finally:
        connection.close()

    rates, _cash = FramePricer(frame, ids).keyed_rates(ids)
    assert rates == expected


# ---------------------------------------------------------------------------
# A decision salvaged from a truncated thought block is still a real decision
# ---------------------------------------------------------------------------
def test_a_salvaged_decision_is_recorded_but_does_not_touch_the_key():
    case = {
        "eval_id": "77065_000001_A1",
        "eval_query": "mammogram at UPMC Passavant",
        "ground_truth_codes": ["77065"],
        "ground_truth_code_MRF": ["77065"],
        "ground_truth_row_ids": ["12345"],
        "ground_truth_hospital_name": "UPMC Passavant",
        "ground_truth_insurance_provider": "Highmark BCBS",
    }
    before = dict(case)
    result = {
        "output": {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []},
        "decision_recovered": True,
        "decision_error": "Could not find a JSON object in the model response.",
        "code_plausible": {"matches": [], "match_count": 0, "filters_used": {}},
        "categorized": {"medical": ["mammogram"]},
        "answer": "There is a price.",
        "total_seconds": 1.0,
    }

    row = generate_row(case, result, calls=[])

    assert row["generate_decision_recovered"] is True
    assert "generate_decision_recovered" in GENERATED_COLUMNS
    assert case == before                      # the key is copied, never edited
    assert row["generate_cpt"] == ["77065"]


# ---------------------------------------------------------------------------
# with_intent: a bolt-on call that must not shadow module-level names
# ---------------------------------------------------------------------------
def test_generate_with_intent_does_not_shadow_config(monkeypatch, tmp_path):
    """Reproduces the UnboundLocalError: a local `from . import config` made the
    module-level one unbound, and every 600-case run died before case one."""
    from pt_healthcare import eval_runner

    class Reply:
        text = "price"

        def __init__(self):
            self.prompts = []

    class FakeClient:
        def __init__(self):
            self.seen = []

        def complete(self, prompt, max_tokens=None):
            self.seen.append(prompt)
            return Reply()

    fake = FakeClient()
    monkeypatch.setattr(eval_runner, "build_client", lambda *args, **kwargs: fake)

    rows = generate(pipeline=oracle(), dataset=DATASET, out=tmp_path / "g.jsonl",
                    per_subtype=1, with_intent=True, progress=None)

    assert len(rows) == 10
    assert all(row["generate_intent"] == "price" for row in rows)
    assert len(fake.seen) == 10               # one routing call per case, no more
