"""End-to-end pipeline tests with fake retriever/engine (no models, no GPU)."""

import json

import pandas as pd
import pytest

from pt_healthcare.pipeline import HealthcarePricingPipeline, orchestration


def make_mrf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2],
            "description": ["MAMMO DIGITAL DX UNI WWO CAD", "MRI BRAIN"],
            "billing_class": ["facility", "facility"],
            "setting": ["outpatient", "outpatient"],
            "hospital_name": ["Upmc Presbyterian Shadyside", "Upmc Childrens"],
            "payer_name": ["UPMC Health Plan", "Aetna"],
            "plan_name": ["Commercial", "Commercial"],
            "modifiers": [None, None],
            "CPT": ["77065", None],
            "HCPCS": ["G0206", "A9579"],
            "standard_charge|gross": [905.0, 1200.0],
            "standard_charge|discounted_cash": [543.0, 800.0],
            "standard_charge|negotiated_dollar": [228.75, 400.0],
            "median_amount": [None, None],
            "10th_percentile": [None, None],
            "90th_percentile": [None, None],
            "standard_charge|min": [58.27, 100.0],
            "standard_charge|max": [769.25, 900.0],
            "standard_charge|methodology": ["fee schedule", "fee schedule"],
        }
    )


class FakeRetriever:
    def __init__(self, cpt, hcpcs):
        self._cpt = cpt
        self._hcpcs = hcpcs
        self.calls = []

    def retrieve_cpt(self, categorized, top_k=None):
        self.calls.append(("cpt", dict(categorized), top_k))
        return self._cpt

    def retrieve_hcpcs(self, categorized, top_k=None):
        self.calls.append(("hcpcs", dict(categorized), top_k))
        return self._hcpcs


class FakeEngine:
    def __init__(self, categorization, decision):
        self.categorization = categorization
        self.decision = decision
        self.answer_args = None
        self.categorize_args = None

    def categorize(self, user_query, hospitals=(), insurers=()):
        self.categorize_args = (user_query, list(hospitals), list(insurers))
        return json.dumps(self.categorization)

    def decide(self, categorization, user_query):
        return "```json\n" + json.dumps(self.decision) + "\n```"

    def answer(self, user_query, output, code_plausible):
        self.answer_args = (user_query, output, code_plausible)
        return "FINAL ANSWER"


def cpt_candidates():
    return pd.DataFrame(
        [
            {"rank": 1, "score": 0.86, "code": "77065", "text": "Diagnostic mammography"},
            {"rank": 2, "score": 0.90, "code": "76090", "text": "Mammography; unilateral"},
        ]
    )


def hcpcs_candidates():
    return pd.DataFrame(
        [
            {"rank": 1, "score": 0.79, "code": "G0206", "text": "Diagnostic mammography"},
        ]
    )


def make_pipeline(decision, categorization=None, **kwargs):
    categorization = categorization or {
        "medical": ["diagnostic mammogram"],
        "hospital": ["UPMC Presbyterian"],
        "insurer": ["UPMC Health Plan"],
        "medication": [],
    }
    retriever = FakeRetriever(cpt_candidates(), hcpcs_candidates())
    engine = FakeEngine(categorization, decision)
    pipeline = HealthcarePricingPipeline(
        retriever=retriever,
        engine=engine,
        mrf_data=make_mrf(),
        **kwargs,
    )
    return pipeline, retriever, engine


def test_pipeline_returns_expected_keys_and_filters_prices():
    pipeline, retriever, engine = make_pipeline(
        {
            "use_codes": "both",
            "cpt_list": ["77065"],
            "hcpcs_list": ["G0206"],
        }
    )

    result = pipeline.run("diagnostic mammogram at UPMC Presbyterian", verbose=False)

    assert set(result) == {
        "categorized",
        "cpt_candidates",
        "hcpcs_candidates",
        "output",
        "code_plausible",
        "answer",
        "total_seconds",
    }
    assert result["output"]["cpt_list"] == ["77065"]
    assert result["output"]["hcpcs_list"] == ["G0206"]
    assert result["code_plausible"]["match_count"] == 1
    assert result["answer"] == "FINAL ANSWER"
    assert result["total_seconds"] >= 0


def test_pipeline_rejects_hallucinated_codes():
    """A code the model selected but that was never retrieved must be dropped."""
    pipeline, _, _ = make_pipeline(
        {
            "use_codes": "cpt",
            "cpt_list": ["77065", "99999"],
            "hcpcs_list": ["ZZ999"],
        }
    )

    result = pipeline.run("query", verbose=False)

    assert result["output"]["cpt_list"] == ["77065"]
    assert result["output"]["hcpcs_list"] == []


def test_pipeline_with_no_surviving_codes_falls_back_to_payer_and_hospital():
    """Carried over from sql_answer, and deliberate: with no code left to filter
    on, the dataset is not constrained by code at all — only the payer and
    hospital filters apply. So an all-hallucinated decision still returns rows.
    """
    pipeline, _, _ = make_pipeline(
        {
            "use_codes": "none",
            "cpt_list": ["00000"],
            "hcpcs_list": [],
        }
    )

    result = pipeline.run("query", verbose=False)

    assert result["output"]["cpt_list"] == []
    assert result["code_plausible"]["filters_used"]["cpt_codes"] == []
    assert result["code_plausible"]["match_count"] == 1  # the UPMC Presbyterian row


def test_answer_receives_entity_linked_context():
    pipeline, _, engine = make_pipeline(
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []}
    )

    pipeline.run("diagnostic mammogram", verbose=False)

    user_query, output, code_plausible = engine.answer_args
    assert user_query == "diagnostic mammogram"
    assert output["cpt_list"] == ["77065"]
    # The model said "UPMC Presbyterian"; the resolver adopted the file's spelling.
    assert code_plausible["filters_used"]["hospitals"] == ["upmc presbyterian shadyside"]


def test_pipeline_passes_the_mrf_vocabulary_to_the_categorizer():
    """The categorizer picks from what exists, instead of inventing names."""
    pipeline, _, engine = make_pipeline(
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []}
    )

    pipeline.run("diagnostic mammogram", verbose=False)

    user_query, hospitals, insurers = engine.categorize_args
    assert user_query == "diagnostic mammogram"
    assert hospitals == ["Upmc Childrens", "Upmc Presbyterian Shadyside"]
    assert insurers == ["Aetna", "UPMC Health Plan"]


def test_vocabulary_is_computed_once():
    pipeline, _, _ = make_pipeline(
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []}
    )

    assert pipeline.hospital_names() is pipeline.hospital_names()
    assert pipeline.payer_names() is pipeline.payer_names()


def test_category_words_and_near_misses_are_fixed_before_filtering():
    """The reported failure: hospital ['hospital'], insurer a near miss.

    "hospital" is a category word, not a facility, and the filter is a substring
    match — left alone it silently selects an arbitrary slice of the file or
    nothing at all.
    """
    pipeline, _, engine = make_pipeline(
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []},
        categorization={
            "medical": ["colonoscopy"],
            "hospital": ["hospital"],
            "insurer": ["UPMC Plan"],
            "medication": [],
        },
    )

    pipeline.run("colonoscopy", verbose=False)

    _, _, code_plausible = engine.answer_args
    filters = code_plausible["filters_used"]

    assert filters["hospitals"] == []
    assert filters["insurers"] == ["upmc health plan"]


def test_top_k_is_forwarded_to_the_retriever():
    pipeline, retriever, _ = make_pipeline(
        {"use_codes": "cpt", "cpt_list": [], "hcpcs_list": []},
        top_k=3,
    )

    pipeline.run("query", verbose=False)

    assert retriever.calls[0][2] == 3
    assert retriever.calls[1][2] == 3


def test_verbose_uses_the_injected_printer():
    messages = []
    pipeline, _, _ = make_pipeline(
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []},
        printer=messages.append,
    )

    pipeline.run("query", verbose=True)

    assert any("STEP 1/7" in message for message in messages)
    assert any("PIPELINE COMPLETE" in message for message in messages)


def test_quiet_pipeline_prints_nothing():
    messages = []
    pipeline, _, _ = make_pipeline(
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []},
        printer=messages.append,
    )

    pipeline.run("query", verbose=False)

    assert messages == []


def test_orchestration_free_function_matches_notebook_api():
    retriever = FakeRetriever(cpt_candidates(), hcpcs_candidates())
    engine = FakeEngine(
        {
            "medical": ["diagnostic mammogram"],
            "hospital": ["UPMC Presbyterian"],
            "insurer": [],
            "medication": [],
        },
        {"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []},
    )

    result = orchestration(
        "diagnostic mammogram",
        verbose=False,
        retriever=retriever,
        engine=engine,
        mrf_data=make_mrf(),
    )

    assert result["output"]["cpt_list"] == ["77065"]
