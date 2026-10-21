"""Unit tests for the eval harness: metric arithmetic and the derived targets.

The point of these is that the reported numbers are auditable. Dice/Jaccard get
tested on the degenerate inputs that decide whether an aggregate is honest — the
empty/empty case that must not be scored as a hit, and the one-sided case that
must be a zero. The target tests re-derive intent and entities from gazetteers and
assert they agree with the dataset's own ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pt_healthcare.eval_runner import (  # noqa: E402
    amounts,
    collapse,
    dice,
    jaccard,
    matches_any,
    normalize_entity,
    percentile,
    prf1,
    recall_at_k,
    reciprocal_rank,
)

DATASET = REPO / "eval"
DB = REPO / ".cache" / "mrf_eval.duckdb"


def test_dice_identity_and_disjoint():
    assert dice({"a", "b"}, {"a", "b"}) == 1.0
    assert dice({"a"}, {"b"}) == 0.0
    assert dice({"a", "b", "c"}, {"a", "b"}) == pytest.approx(4 / 5)


def test_dice_and_jaccard_agree_on_partial_overlap():
    predicted, truth = {"a", "b", "c"}, {"b", "c", "d"}
    assert dice(predicted, truth) == pytest.approx(4 / 6)
    assert jaccard(predicted, truth) == pytest.approx(2 / 4)


def test_empty_sets_are_not_scored_as_perfect():
    # 120 of the 600 cases have no row: returning nothing there is correct but
    # must not enter the Dice average as a 1.0.
    assert dice(set(), set()) is None
    assert jaccard([], []) is None


def test_one_sided_empty_is_zero():
    assert dice({"a"}, set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_prf1_counts():
    precision, recall, f1 = prf1({"a", "b"}, {"b", "c"})
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_precision_and_recall_differ_when_sets_are_lopsided():
    precision, recall, _ = prf1({"a", "b", "c", "d"}, {"a"})
    assert precision == pytest.approx(0.25)
    assert recall == pytest.approx(1.0)


def test_recall_at_k_and_reciprocal_rank():
    ranked = ["x", "y", "z"]
    assert recall_at_k(ranked, {"y", "z", "w"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, {"x"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"z"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"q"}) == 0.0
    assert reciprocal_rank(ranked, set()) is None


def test_amount_extraction_handles_commas_and_cents():
    assert amounts("the rate is $1,234.56, cash $900") == [1234.56, 900.0]
    assert amounts("no figure here") == []
    assert amounts("") == []


def test_amount_tolerance_is_one_cent():
    assert matches_any(1234.56, [1234.56])
    assert matches_any(1234.56 + 0.0001, [1234.56]), "float artifacts must still match"
    assert matches_any(1234.57, [1234.56]) is False, "a wrong cent is wrong"
    assert matches_any(1234.56 + 0.01, [1234.56]) is False


def test_entity_normalization_makes_articles_neutral():
    assert normalize_entity("a colonoscopy") == normalize_entity("colonoscopy")
    assert normalize_entity("The  MRI ") == "mri"


def test_collapse_reduces_rate_lines_to_one_identity():
    rows = [
        {"cpt": "27130", "hospital": "Upmc Magee", "payer": "Aetna"},
        {"cpt": "27130", "hospital": "Upmc Magee", "payer": "Aetna"},
        {"cpt": "27130", "hospital": "Upmc Magee", "payer": "Highmark BCBS of PA"},
    ]
    assert len(collapse(rows)) == 2


def test_percentile_is_defined_for_small_samples():
    assert percentile([], 0.95) is None
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 3.0


@pytest.mark.skipif(not DB.is_file() or not (DATASET / "pt_eval_v2.jsonl").is_file(),
                    reason="needs the MRF cache and the dataset")
def test_targets_agree_with_the_dataset_ground_truth():
    sys.path.insert(0, str(REPO / "eval_code_gen"))
    from build_targets import build  # type: ignore

    targets, audit = build(DATASET, DB)

    assert len(targets) == 600
    assert audit["hospital_not_in_gt"] == 0, "a question names a hospital the key omits"
    assert audit["payer_not_in_gt"] == 0, "a question names a payer the key omits"
    assert audit["gt_hospital_unnamed"] == 0, "the key names a hospital the question omits"
    assert audit["gt_payer_unnamed"] == 0, "the key names a payer the question omits"
    assert audit["multi_hospital"] == 60, "only the C1 comparison cases name two hospitals"
    assert audit["medical_withheld"] == 12, "only the 12 B1 service-withheld cases have no service"

    # Every case in this set asks what something costs, so the router's expected
    # label is "price" throughout. That makes the intent number a false-positive
    # test (a cost question must never route to lookup or explain), not a
    # three-way F1 — proving the other two labels needs a routing set.
    intents: dict = {}
    for record in targets:
        intents[record["intent"]] = intents.get(record["intent"], 0) + 1
    assert intents == {"price": 600}

    medication = [record for record in targets if record["canonical"]["medication"]]
    assert len(medication) == 54, "the insulin cases are the only ones with a medication entity"
    assert all(record["entities"]["medication"] for record in medication)
