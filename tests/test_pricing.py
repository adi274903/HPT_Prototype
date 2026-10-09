"""Tests for MRF price filtering (``sql_answer``)."""

import pandas as pd
import pytest

from pt_healthcare.pricing import sql_answer


def make_mrf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "description": [
                "MAMMO DIGITAL DX UNI WWO CAD",
                "COLONOSCOPY DIAG W/WO BRUSHING",
                "COLONOSCOPY, FLEXIBLE; DIAGNOSTIC",
                "MRI BRAIN W WO CONTRAST",
            ],
            "billing_class": ["facility"] * 4,
            "setting": ["outpatient"] * 4,
            "hospital_name": [
                "Upmc Presbyterian Shadyside",
                "Upmc Altoona",
                "Upmc Altoona",
                "Upmc Childrens",
            ],
            "payer_name": [
                "UPMC Health Plan",
                "Highmark BCBS of PA",
                "Highmark BCBS of PA",
                "Aetna",
            ],
            "plan_name": ["Commercial", "Commercial - Indemnity", "Managed Care", "Commercial"],
            "modifiers": [None, None, None, None],
            "CPT": ["77065", "45378", "45378", None],
            "HCPCS": ["G0206", None, "G9937", "A9579"],
            "standard_charge|gross": [905.0, 2500.0, 2500.0, 1200.0],
            "standard_charge|discounted_cash": [543.0, 1500.0, 1500.0, 800.0],
            "standard_charge|negotiated_dollar": [228.75, 729.80, 869.21, 400.0],
            "median_amount": [None, None, None, None],
            "10th_percentile": [None, None, None, None],
            "90th_percentile": [None, None, None, None],
            "standard_charge|min": [58.27, 180.50, 180.50, 100.0],
            "standard_charge|max": [769.25, 3124.67, 3124.67, 900.0],
            "standard_charge|methodology": ["fee schedule"] * 4,
        }
    )


def test_requires_mrf_dataframe():
    with pytest.raises(ValueError):
        sql_answer(cpt_list=["77065"])


def test_missing_columns_raise():
    broken = make_mrf().drop(columns=["median_amount"])

    with pytest.raises(ValueError, match="missing expected columns"):
        sql_answer(cpt_list=["77065"], mrf_data=broken)


def test_filters_by_code_payer_and_hospital():
    result = sql_answer(
        cpt_list=["77065", "77066"],
        insurance_list=["UPMC Health Plan"],
        hospital_list=["UPMC Presbyterian"],
        mrf_data=make_mrf(),
    )

    assert result["match_count"] == 1
    assert result["matches"].iloc[0]["CPT"] == "77065"
    assert result["filters_used"]["insurers"] == ["upmc health plan"]
    assert result["filters_used"]["cpt_codes"] == ["77065", "77066"]


def test_lowercase_code_input_is_uppercased():
    """G0206 is a HCPCS code, and CPT/HCPCS are separate columns."""
    as_cpt = sql_answer(cpt_list=["g0206"], mrf_data=make_mrf())

    assert as_cpt["match_count"] == 0
    assert as_cpt["filters_used"]["cpt_codes"] == ["G0206"]

    as_hcpcs = sql_answer(hcpcs_list=["g0206"], mrf_data=make_mrf())

    assert as_hcpcs["match_count"] == 1
    assert as_hcpcs["filters_used"]["hcpcs_codes"] == ["G0206"]
    assert as_hcpcs["matches"].iloc[0]["HCPCS"] == "G0206"


def test_hcpcs_filter():
    result = sql_answer(hcpcs_list=["G9937"], mrf_data=make_mrf())

    assert result["match_count"] == 1
    assert result["matches"].iloc[0]["HCPCS"] == "G9937"


def test_cpt_or_hcpcs_semantics():
    result = sql_answer(cpt_list=["77065"], hcpcs_list=["G9937"], mrf_data=make_mrf())

    assert result["match_count"] == 2


def test_no_code_filter_returns_all_then_payer_narrows():
    all_rows = sql_answer(insurance_list=[], hospital_list=[], mrf_data=make_mrf())
    assert all_rows["match_count"] == 4

    narrowed = sql_answer(insurance_list=["Highmark BCBS"], mrf_data=make_mrf())
    assert narrowed["match_count"] == 2


def test_unknown_code_returns_zero_rows():
    result = sql_answer(cpt_list=["00000"], mrf_data=make_mrf())

    assert result["match_count"] == 0
    assert result["price_summary"].empty


def test_price_columns_are_numeric():
    result = sql_answer(cpt_list=["45378"], mrf_data=make_mrf())

    assert pd.api.types.is_numeric_dtype(result["matches"]["standard_charge|gross"])
    assert pd.api.types.is_numeric_dtype(
        result["matches"]["standard_charge|negotiated_dollar"]
    )


def test_price_summary_is_sorted_and_indexed_from_zero():
    result = sql_answer(cpt_list=["45378"], mrf_data=make_mrf())
    summary = result["price_summary"]

    assert summary.index.tolist() == [0, 1]
    assert list(summary["CPT"]) == ["45378", "45378"]
