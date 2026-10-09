"""Tests for the closed vocabulary injected into the categorizer prompt."""

import pandas as pd

from pt_healthcare.vocabulary import (
    column_vocabulary,
    drop_placeholders,
    resolve_to_vocabulary,
)

PAYERS = ["Aetna", "Highmark BCBS of PA", "Multiplan", "UPMC Health Plan"]


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hospital_name": [
                "Upmc Altoona",
                "upmc altoona",       # same hospital, different casing
                "Upmc Bedford",
                None,
                "",
                "nan",
            ],
            "payer_name": [
                "UPMC Health Plan",
                "Highmark BCBS of PA",
                None,
                None,
                None,
                None,
            ],
        }
    )


def test_column_vocabulary_dedupes_case_insensitively_and_sorts():
    assert column_vocabulary(make_frame(), "hospital_name") == [
        "Upmc Altoona",
        "Upmc Bedford",
    ]


def test_column_vocabulary_keeps_the_first_spelling_seen():
    frame = pd.DataFrame({"hospital_name": ["UPMC ALTOONA", "Upmc Altoona"]})

    assert column_vocabulary(frame, "hospital_name") == ["UPMC ALTOONA"]


def test_column_vocabulary_handles_a_missing_column_or_frame():
    assert column_vocabulary(make_frame(), "nope") == []
    assert column_vocabulary(None, "hospital_name") == []


def test_column_vocabulary_respects_the_limit():
    frame = pd.DataFrame({"hospital_name": [f"Hospital {i}" for i in range(10)]})

    assert len(column_vocabulary(frame, "hospital_name", limit=3)) == 3


def test_drop_placeholders_removes_bare_category_words():
    assert drop_placeholders(["hospital", "UPMC Presbyterian"], "hospital") == [
        "UPMC Presbyterian"
    ]
    assert drop_placeholders(["my insurance", "Aetna"], "insurer") == ["Aetna"]


def test_drop_placeholders_is_case_and_space_insensitive():
    assert drop_placeholders(["  HOSPITAL  "], "hospital") == []


def test_drop_placeholders_keeps_real_names_and_real_programs():
    assert drop_placeholders(["UPMC Presbyterian"], "hospital") == ["UPMC Presbyterian"]
    # Medicare is a payer that genuinely appears in an MRF, not a category word.
    assert drop_placeholders(["medicare"], "insurer") == ["medicare"]


def test_drop_placeholders_ignores_unknown_kinds():
    assert drop_placeholders(["hospital"], "medication") == ["hospital"]


def test_resolve_exact_match_adopts_the_file_spelling():
    assert resolve_to_vocabulary(["UPMC HEALTH PLAN"], PAYERS) == ["UPMC Health Plan"]


def test_resolve_snaps_a_partial_name_onto_the_full_entry():
    assert resolve_to_vocabulary(["Highmark"], PAYERS) == ["Highmark BCBS of PA"]
    assert resolve_to_vocabulary(
        ["UPMC Presbyterian"], ["Upmc Presbyterian Shadyside"]
    ) == ["Upmc Presbyterian Shadyside"]


def test_resolve_snaps_a_near_miss_onto_the_real_entry():
    """The reported failure: a near miss never substring-matches the real value."""
    assert resolve_to_vocabulary(["Highmark Plan"], PAYERS) == ["Highmark BCBS of PA"]


def test_resolve_leaves_genuinely_ambiguous_terms_alone():
    """Guessing between two UPMC payers is worse than reporting what we got."""
    payers = ["UPMC Health Plan", "UPMC Work Partners"]

    assert resolve_to_vocabulary(["UPMC"], payers) == ["UPMC"]


def test_resolve_without_a_vocabulary_passes_values_through():
    assert resolve_to_vocabulary(["Highmark Plan"], []) == ["Highmark Plan"]


def test_resolve_dedupes_after_snapping():
    assert resolve_to_vocabulary(["Highmark", "Highmark BCBS of PA"], PAYERS) == [
        "Highmark BCBS of PA"
    ]


def test_vocabulary_block_is_gone():
    """The prompt no longer carries the hospital and payer list.

    Keeping it out is the point: it was ~280 tokens, the deterministic resolver
    already maps near misses onto real entries, and the two-listed/two-unlisted
    split is what made the model return an empty medical category.
    """
    from pt_healthcare import vocabulary

    assert not hasattr(vocabulary, "vocabulary_block")
    assert hasattr(vocabulary, "resolve_to_vocabulary")
