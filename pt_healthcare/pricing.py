"""MRF price filtering.

``sql_answer`` is carried over verbatim from notebook cell 20: it filters the
cleaned MRF DataFrame by CPT/HCPCS codes (OR within the code family) plus
case-insensitive payer and hospital substring filters (ANDed on top).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .utils import as_clean_list, contains_any, normalize_series

# Columns that must exist for either filtering or the compact summary.
REQUIRED_COLUMNS = {
    "description",
    "payer_name",
    "plan_name",
    "hospital_name",
    "CPT",
    "HCPCS",
    "standard_charge|gross",
    "standard_charge|discounted_cash",
    "standard_charge|negotiated_dollar",
    "median_amount",
    "standard_charge|min",
    "standard_charge|max",
}

# Columns coerced to numeric before the summary is produced.
PRICE_COLUMNS = [
    "standard_charge|gross",
    "standard_charge|discounted_cash",
    "standard_charge|negotiated_dollar",
    "median_amount",
    "10th_percentile",
    "90th_percentile",
    "standard_charge|min",
    "standard_charge|max",
]

# Compact, answer-focused view handed to the final generation step.
SUMMARY_COLUMNS = [
    "row_id",
    "hospital_name",
    "payer_name",
    "plan_name",
    "description",
    "billing_class",
    "setting",
    "CPT",
    "HCPCS",
    "modifiers",
    "standard_charge|gross",
    "standard_charge|discounted_cash",
    "standard_charge|negotiated_dollar",
    "median_amount",
    "10th_percentile",
    "90th_percentile",
    "standard_charge|min",
    "standard_charge|max",
    "standard_charge|methodology",
]

SORT_COLUMNS = [
    "hospital_name",
    "payer_name",
    "plan_name",
    "CPT",
    "HCPCS",
    "standard_charge|negotiated_dollar",
]


def sql_answer(
    cpt_list: Optional[Sequence[str]] = None,
    hcpcs_list: Optional[Sequence[str]] = None,
    insurance_list: Optional[Sequence[str]] = None,
    hospital_list: Optional[Sequence[str]] = None,
    mrf_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Filter cleaned MRF data for matched CPT/HCPCS codes, payers and hospitals.

    Parameters
    ----------
    cpt_list : list[str] | str | None
        CPT candidates, such as ["77065", "77066"].
    hcpcs_list : list[str] | str | None
        HCPCS candidates, such as ["G0202", "A0428"].
    insurance_list : list[str] | str | None
        Payer/insurance names, such as ["UPMC Health Plan"].
    hospital_list : list[str] | str | None
        Hospital names, such as ["UPMC Presbyterian"].
    mrf_data : pd.DataFrame
        The cleaned hospital MRF DataFrame.

    Returns
    -------
    dict
        {
            "matches": full filtered MRF rows,
            "price_summary": compact price-focused result table,
            "filters_used": normalized input values,
            "match_count": number of MRF rows found,
        }
    """
    if mrf_data is None:
        raise ValueError("Pass your MRF DataFrame using `mrf_data=mrf_data`.")

    missing_columns = REQUIRED_COLUMNS - set(mrf_data.columns)

    if missing_columns:
        raise ValueError(
            f"mrf_data is missing expected columns: {sorted(missing_columns)}\n"
            f"Available columns: {mrf_data.columns.tolist()}"
        )

    # ---------------------------------------------------------------
    # Normalize model/extractor output.
    # CPT and HCPCS are code comparisons, so uppercase is appropriate.
    # Insurer and hospital names use case-insensitive text matching.
    # ---------------------------------------------------------------
    cpt_codes = as_clean_list(cpt_list, uppercase=True)
    hcpcs_codes = as_clean_list(hcpcs_list, uppercase=True)
    insurers = as_clean_list(insurance_list, uppercase=False)
    hospitals = as_clean_list(hospital_list, uppercase=False)

    # Start with all records.
    mask = pd.Series(True, index=mrf_data.index)

    # ---------------------------------------------------------------
    # Code logic: CPT OR HCPCS inside the code condition.
    #
    # Example:
    #   CPT in ["77065", "77066"]
    #   OR HCPCS in ["G0202"]
    #
    # If neither list is supplied, do not constrain the dataset by code.
    # ---------------------------------------------------------------
    if cpt_codes or hcpcs_codes:
        code_mask = pd.Series(False, index=mrf_data.index)

        if cpt_codes:
            code_mask |= normalize_series(
                mrf_data["CPT"],
                uppercase=True,
            ).isin(cpt_codes)

        if hcpcs_codes:
            code_mask |= normalize_series(
                mrf_data["HCPCS"],
                uppercase=True,
            ).isin(hcpcs_codes)

        mask &= code_mask

    # ---------------------------------------------------------------
    # Payer and hospital filters (substring match for MRF naming variation).
    # ---------------------------------------------------------------
    if insurers:
        mask &= contains_any(mrf_data["payer_name"], insurers)

    if hospitals:
        mask &= contains_any(mrf_data["hospital_name"], hospitals)

    matches = mrf_data.loc[mask].copy()

    # Convert relevant price columns to numeric where possible.
    for column in PRICE_COLUMNS:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")

    summary_columns = [column for column in SUMMARY_COLUMNS if column in matches.columns]

    sort_columns = [column for column in SORT_COLUMNS if column in summary_columns]

    if sort_columns:
        price_summary = (
            matches[summary_columns]
            .sort_values(by=sort_columns, na_position="last")
            .reset_index(drop=True)
        )
    else:
        price_summary = matches[summary_columns].reset_index(drop=True)

    return {
        "matches": matches.reset_index(drop=True),
        "price_summary": price_summary,
        "match_count": len(matches),
        "filters_used": {
            "cpt_codes": cpt_codes,
            "hcpcs_codes": hcpcs_codes,
            "insurers": insurers,
            "hospitals": hospitals,
        },
    }
