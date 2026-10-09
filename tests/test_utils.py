import pandas as pd

from pt_healthcare.utils import as_clean_list, clean_list, contains_any, normalize_series


def test_clean_list_handles_none_string_and_list():
    assert clean_list(None) == []
    assert clean_list("mammogram") == ["mammogram"]
    assert clean_list(["a", None, "  ", "nan", "None", "b"]) == ["a", "b"]


def test_clean_list_preserves_case():
    assert clean_list(["UPMC Health Plan"]) == ["UPMC Health Plan"]


def test_as_clean_list_casefolds_and_dedupes_in_order():
    result = as_clean_list(["UPMC", "upmc", " Aetna ", None, "nan"])
    assert result == ["upmc", "aetna"]


def test_as_clean_list_uppercase_for_codes():
    assert as_clean_list(["g0206", "77065", "77065"], uppercase=True) == ["G0206", "77065"]


def test_normalize_series_casefold_and_upper():
    series = pd.Series(["  Upmc  ", None, "AETNA"])

    assert normalize_series(series).tolist() == ["upmc", "", "aetna"]
    assert normalize_series(series, uppercase=True).tolist() == ["UPMC", "", "AETNA"]


def test_contains_any_is_substring_and_case_insensitive():
    series = pd.Series(["UPMC Presbyterian Shadyside", "UPMC Altoona", "Aetna"])

    assert contains_any(series, ["upmc presbyterian"]).tolist() == [True, False, False]
    assert contains_any(series, ["upmc"]).tolist() == [True, True, False]


def test_contains_any_returns_all_true_when_no_terms():
    series = pd.Series(["a", "b"])

    assert contains_any(series, []).tolist() == [True, True]
