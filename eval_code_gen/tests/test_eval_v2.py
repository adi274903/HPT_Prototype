"""Schema and provenance invariants for eval_dataset/pt_eval_v2.jsonl.

The provenance tests re-query the raw MRF, so they catch an answer key that was
assembled from stale state or from the deduped combos table alone.

    python -m pytest eval_code_gen/tests -q      # from the repo root
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent.parent          # eval_code_gen/
REPO = CODE.parent                                      # repo root
EVAL = REPO / "eval_dataset" / "pt_eval_v2.jsonl"
META = REPO / "eval_dataset" / "pt_eval_v2_meta.jsonl"
MRF = Path.home() / "Downloads" / "everyUPMCmrf_clean.csv"
SUBTYPES = ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2")
ID_RX = re.compile(r"^(?:\d{5}|\d{4}[A-Z]|[A-Z]\d{4})_\d{6}_(A1|A2|A3|B1|B2|B3|C1|C2|D1|D2)$")
KEYS = ["eval_id", "eval_query", "ground_truth_codes", "ground_truth_code_MRF",
        "ground_truth_row_ids", "ground_truth_hospital_name", "ground_truth_insurance_provider"]

sys.path.insert(0, str(CODE))


def _read(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def cases():
    if not EVAL.exists():
        pytest.skip("run eval_code_gen/build_eval_v2.py first")
    return _read(EVAL)


@pytest.fixture(scope="module")
def meta():
    return {m["eval_id"]: m for m in _read(META)}


@pytest.fixture(scope="module")
def con():
    duckdb = pytest.importorskip("duckdb")
    if not MRF.exists():
        pytest.skip("MRF not available")
    c = duckdb.connect()
    c.execute("PRAGMA threads=4")
    c.execute(
        f"""CREATE VIEW mrf AS
            SELECT row_id, nullif(trim("CPT"),'') AS cpt, nullif(trim("HCPCS"),'') AS hcpcs,
                   hospital_name, payer_name, plan_name, description,
                   TRY_CAST("standard_charge|negotiated_dollar" AS DOUBLE) AS rate
            FROM read_csv_auto('{MRF}', header=true, all_varchar=true, sample_size=200000)"""
    )
    return c


# ── schema ──────────────────────────────────────────────────────────────────
def test_key_set_and_order(cases):
    for case in cases:
        assert list(case) == KEYS, case["eval_id"]


def test_id_format_and_uniqueness(cases):
    ids = [c["eval_id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert ID_RX.match(case["eval_id"]), case["eval_id"]


def test_types(cases):
    for case in cases:
        assert isinstance(case["eval_query"], str) and case["eval_query"].strip()
        for key in ("ground_truth_codes", "ground_truth_code_MRF", "ground_truth_row_ids"):
            assert isinstance(case[key], list), case["eval_id"]
            assert all(isinstance(v, str) for v in case[key]), case["eval_id"]
        for key in ("ground_truth_hospital_name", "ground_truth_insurance_provider"):
            assert case[key] is None or isinstance(case[key], str), case["eval_id"]


def test_mrf_codes_are_a_subset(cases):
    for case in cases:
        assert set(case["ground_truth_code_MRF"]) <= set(case["ground_truth_codes"]), case["eval_id"]


def test_category_codes_all_come_from_the_catalog(cases):
    from categories import load_catalog

    catalog = load_catalog()
    owned = set(catalog["CPT"]) | set(catalog["HCPCS"])
    for case in cases:
        assert set(case["ground_truth_codes"]) <= owned, case["eval_id"]


def test_named_entities_appear_in_the_query(cases, meta):
    for case in cases:
        record = meta[case["eval_id"]]
        if case["ground_truth_hospital_name"]:
            assert case["ground_truth_hospital_name"].lower() in case["eval_query"].lower(), case["eval_id"]
        if case["ground_truth_insurance_provider"]:
            needle = record.get("alias_used") or case["ground_truth_insurance_provider"]
            assert needle.lower() in case["eval_query"].lower(), case["eval_id"]


def test_every_case_has_meta(cases, meta):
    for case in cases:
        assert case["eval_id"] in meta
        assert meta[case["eval_id"]]["subtype"] == case["eval_id"].rsplit("_", 1)[1]


def test_each_subtype_is_fully_populated(cases):
    counts = {}
    for case in cases:
        key = case["eval_id"].rsplit("_", 1)[1]
        counts[key] = counts.get(key, 0) + 1
    assert set(counts) == set(SUBTYPES)
    assert all(n == 60 for n in counts.values()), counts


def test_queries_are_not_duplicated(cases):
    queries = [c["eval_query"] for c in cases]
    assert len(queries) == len(set(queries))


# ── provenance against the raw MRF ──────────────────────────────────────────
def test_row_ids_exist_and_match_the_pivot(cases, con):
    """Every row id must be a real MRF row carrying the eval_id's pivot code."""
    pairs = []
    for case in cases:
        pivot = case["eval_id"].split("_")[0]
        pairs += [(case["eval_id"], pivot, rid) for rid in case["ground_truth_row_ids"]]

    assert pairs, "no row ids to check"
    con.execute("CREATE TEMP TABLE claim(eval_id VARCHAR, pcode VARCHAR, row_id VARCHAR)")
    con.executemany("INSERT INTO claim VALUES (?, ?, ?)", pairs)

    bad = con.execute(
        """SELECT c.eval_id, c.pcode, c.row_id
           FROM claim c LEFT JOIN mrf m ON m.row_id = c.row_id
           WHERE m.row_id IS NULL
              OR (m.cpt IS DISTINCT FROM c.pcode AND m.hcpcs IS DISTINCT FROM c.pcode)"""
    ).fetchall()
    assert not bad, f"{len(bad)} row id(s) do not resolve to the pivot code: {bad[:5]}"

    missing = con.execute(
        "SELECT count(*) FROM claim c LEFT JOIN mrf m USING (row_id) WHERE m.row_id IS NULL"
    ).fetchone()[0]
    assert missing == 0


def test_row_ids_respect_the_stated_entities(cases, meta, con):
    """A keyed row must sit at the hospital/payer the question names."""
    pairs = [(c["eval_id"], rid) for c in cases for rid in c["ground_truth_row_ids"]]
    if not pairs:
        return
    con.execute("CREATE TEMP TABLE claim2(eval_id VARCHAR, row_id VARCHAR)")
    con.executemany("INSERT INTO claim2 VALUES (?, ?)", pairs)
    by_id = {c["eval_id"]: c for c in cases}

    rows = con.execute(
        """SELECT c.eval_id, m.hospital_name, m.payer_name
           FROM claim2 c JOIN mrf m USING (row_id)"""
    ).fetchall()

    for eval_id, hospital, payer in rows:
        case, record = by_id[eval_id], meta[eval_id]
        if case["eval_id"].endswith("_C1"):
            # C1 is a two-hospital comparison: the key spans both hospitals and the
            # cheapest one is the ground-truth hospital name.
            assert hospital in case["eval_query"], eval_id
            assert payer == case["ground_truth_insurance_provider"], eval_id
            continue
        if case["ground_truth_hospital_name"]:
            assert hospital == case["ground_truth_hospital_name"], eval_id
        if case["ground_truth_insurance_provider"]:
            assert payer == case["ground_truth_insurance_provider"], eval_id
        withheld = record.get("withheld_field")
        if withheld == "hospital":
            assert case["ground_truth_hospital_name"] is None
        if withheld == "payer":
            assert case["ground_truth_insurance_provider"] is None


def test_d1_pairs_have_no_rows(cases, con):
    """Abstention cases must genuinely be absent from the file."""
    pairs = [(c["eval_id"].split("_")[0], c["ground_truth_hospital_name"])
             for c in cases if c["eval_id"].endswith("_D1")]
    assert pairs
    con.execute("CREATE TEMP TABLE absent(pcode VARCHAR, hospital VARCHAR)")
    con.executemany("INSERT INTO absent VALUES (?, ?)", pairs)

    hits = con.execute(
        """SELECT a.pcode, a.hospital, count(*)
           FROM absent a JOIN mrf m ON m.hospital_name = a.hospital
                AND (m.cpt = a.pcode OR m.hcpcs = a.pcode)
           GROUP BY 1, 2"""
    ).fetchall()
    assert not hits, f"D1 cases that do have rows: {hits[:5]}"


def test_d2_services_appear_in_no_description(cases, meta, con):
    """An invented service must not be findable in the MRF text."""
    services = [meta[c["eval_id"]]["invented_service"]
                for c in cases if c["eval_id"].endswith("_D2")]
    assert services
    found = []
    for phrase in services:
        words = [w for w in re.findall(r"[a-z]{6,}", phrase.lower())][:2]
        if not words:
            continue
        where = " AND ".join("description ILIKE ?" for _ in words)
        if con.execute(f"SELECT count(*) FROM mrf WHERE {where}",
                       [f"%{w}%" for w in words]).fetchone()[0]:
            found.append(words)
    assert not found, f"invented services with a partial text match: {found[:3]}"


# ── specific knowns from the source data ────────────────────────────────────
def test_colonoscopy_split_is_the_verified_one(cases):
    """31 curated codes, 28 of them present in the MRF (44397/44406/44407 are not)."""
    colon = [c for c in cases if "45378" in c["ground_truth_codes"]]
    assert colon
    for case in colon:
        assert len(case["ground_truth_codes"]) == 31, case["eval_id"]
        assert len(case["ground_truth_code_MRF"]) == 28, case["eval_id"]
        assert not {"44397", "44406", "44407"} & set(case["ground_truth_code_MRF"])


def test_c1_keys_the_cheaper_hospital_minimum(cases, meta):
    for case in cases:
        if not case["eval_id"].endswith("_C1"):
            continue
        record = meta[case["eval_id"]]
        assert record["cheapest_rate"] < record["priciest_rate"], case["eval_id"]
        assert round(record["priciest_rate"] - record["cheapest_rate"], 2) == record["difference"]
        assert case["ground_truth_hospital_name"] == record["cheapest"]
        assert len(record["ranking"]) >= 2


def test_build_is_deterministic(tmp_path):
    """Same seed -> byte-identical output."""
    pytest.importorskip("duckdb")
    common = ["--per-subtype", "3", "--seed", "7", "--db", str(REPO / ".cache" / "mrf_eval.duckdb")]
    outs = []
    for tag in ("a", "b"):
        out, meta_path = tmp_path / f"{tag}.jsonl", tmp_path / f"{tag}.meta"
        subprocess.run([sys.executable, "eval_code_gen/build_eval_v2.py", *common,
                        "--out", str(out), "--meta", str(meta_path)],
                       check=True, capture_output=True, cwd=REPO)
        outs.append(out.read_text())
    assert outs[0] == outs[1]
