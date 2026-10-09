"""Tests for the closed vocabulary injected into the categorizer prompt."""

import pandas as pd

from pt_healthcare.vocabulary import column_vocabulary, vocabulary_block


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


def test_vocabulary_block_is_empty_without_known_values():
    """No MRF loaded means the prompt stays exactly the prototype's."""
    assert vocabulary_block() == ""
    assert vocabulary_block([], []) == ""


def test_vocabulary_block_lists_both_sides():
    block = vocabulary_block(["Upmc Altoona"], ["Aetna"])

    assert "hospitals (1):" in block
    assert "- Upmc Altoona" in block
    assert "insurers (1):" in block
    assert "- Aetna" in block
    # The instruction that stops the model dumping the whole list back.
    assert "never return the list itself" in block


def test_vocabulary_block_omits_an_empty_side():
    block = vocabulary_block([], ["Aetna"])

    assert "insurers (1):" in block
    assert "hospitals" not in block
