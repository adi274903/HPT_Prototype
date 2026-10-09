"""Tests for the prompt builders (pure functions)."""

from pt_healthcare.prompts import (
    build_answer_prompt,
    build_categorizer_prompt,
    build_decision_prompt,
    build_general_prompt,
)


def test_categorizer_prompt_embeds_query_and_schema():
    prompt = build_categorizer_prompt("What might a diagnostic mammogram cost?")

    assert "What might a diagnostic mammogram cost?" in prompt
    assert '"medical": []' in prompt
    assert '"hospital": []' in prompt
    assert '"insurer": []' in prompt
    assert '"medication" : []' in prompt
    assert "Return ONLY valid JSON" in prompt


def test_categorizer_prompt_includes_known_values_when_given():
    prompt = build_categorizer_prompt(
        "colonsocopy with my Higmark BCBS plan",
        hospitals=["Upmc Presbyterian Shadyside"],
        insurers=["Highmark BCBS of PA"],
    )

    assert "- Highmark BCBS of PA" in prompt
    assert "- Upmc Presbyterian Shadyside" in prompt
    assert "hospitals (1):" in prompt
    assert "insurers (1):" in prompt
    assert "colonsocopy with my Higmark BCBS plan" in prompt


def test_categorizer_prompt_is_unchanged_without_known_values():
    """No MRF means the prompt is byte-identical to the prototype's."""
    assert build_categorizer_prompt("q") == build_categorizer_prompt("q", [], [])
    assert "Known entries" not in build_categorizer_prompt("q")


def test_categorizer_prompt_no_longer_asks_for_generic_references():
    """It used to say: include generic references such as "my hospital"."""
    prompt = build_categorizer_prompt("q")

    assert "Include generic references" not in prompt
    assert "Never emit a generic word" in prompt


def test_categorizer_prompt_shows_a_query_that_names_no_facility():
    """The failing case, taught as an example: hospital stays empty."""
    prompt = build_categorizer_prompt("q")

    assert "Which hospital should I go to for a colonoscopy?" in prompt
    assert "Asking *which*" in prompt
    # The old example taught that "Houston Hospital" is a hospital entity.
    assert "Houston Hospital" not in prompt


def test_every_model_call_sets_an_explicit_token_budget():
    """The categorizer must not inherit the pipe's 256-token default.

    MedGemma spends that on its <unused94>thought block before reaching the JSON,
    so a verbose query gets truncated into a response with no JSON at all.
    """
    import inspect

    from pt_healthcare import config
    from pt_healthcare.llm import MedGemmaEngine

    source = inspect.getsource(MedGemmaEngine)

    assert source.count("max_new_tokens=config.max_new_tokens()") == 5, (
        "categorize/decide/answer/classify_intent/explain must all pass a budget"
    )
    assert config.max_new_tokens() > 256


def test_categorizer_prompt_scopes_the_reference_lists():
    """The lists must not read as a whitelist for medical or medication.

    Regression: with the lists present, "What might a diagnostic mammogram cost
    at UPMC Presbyterian with UPMC Health Plan?" came back with ``medical: []``.
    The two categories that had lists were filled; the one without was not.
    """
    prompt = build_categorizer_prompt("q", ["Upmc Altoona"], ["Aetna"])

    assert "hospital and insurer categories ONLY" in prompt
    assert "restrict the medical or medication categories" in prompt


def test_categorizer_prompt_shows_a_fully_populated_price_query():
    """The regression query, taught as a worked example."""
    prompt = build_categorizer_prompt("q")

    assert '"medical": ["diagnostic mammogram"]' in prompt
    assert "Every category is filled from the query text." in prompt


def test_every_model_call_is_deterministic():
    """All three calls must be greedy.

    The categorizer and the answer step both used to sample, so the same query
    produced different entities and different answers on consecutive runs.
    """
    import inspect

    from pt_healthcare.llm import MedGemmaEngine

    source = inspect.getsource(MedGemmaEngine)

    assert source.count("do_sample=False") == 5, (
        "every model call must be deterministic"
    )


def test_intent_prompt_offers_every_tool():
    """It used to offer only price and lookup, so 'explain' was unreachable."""
    from pt_healthcare.prompts import build_intent_prompt

    prompt = build_intent_prompt("what is a colonoscopy?")

    for tool in ("price", "lookup", "explain"):
        assert tool in prompt

    assert "exactly one word" in prompt
    assert "what is a colonoscopy?" in prompt


def test_general_prompt_says_no_lookup_happened():
    """The model must not imply it searched when it did not."""
    prompt = build_general_prompt("what is a colonoscopy?")

    assert "No index lookup was performed" in prompt
    assert "Never guess or assert a specific billing code" in prompt
    assert "what is a colonoscopy?" in prompt


def test_decision_prompt_compacts_and_serializes_candidates():
    categorization = {
        "medical": ["diagnostic mammogram"],
        "cpt_candidates": [
            {"code": "77065", "text": "Diagnostic mammography, unilateral", "score": 0.8605},
            {"code": "77065", "text": "duplicate", "score": 0.1},
        ],
        "hcpcs_candidates": [
            {"code": "G0204", "text": "Diagnostic mammography", "score": 0.7996},
        ],
    }

    prompt = build_decision_prompt(categorization, "query")

    assert 'Procedure: ["diagnostic mammogram"]' in prompt
    assert '"code": "77065"' in prompt
    assert '"code": "G0204"' in prompt
    # The duplicate was removed by compact_candidates.
    assert prompt.count('"code": "77065"') == 1
    assert '"use_codes": "cpt"' in prompt


def test_decision_prompt_caps_candidates_at_eight():
    categorization = {
        "medical": ["x"],
        "cpt_candidates": [
            {"code": str(i), "text": "t", "score": i} for i in range(20)
        ],
        "hcpcs_candidates": [],
    }

    prompt = build_decision_prompt(categorization, "query")

    # 8 CPT codes kept, the 9th (code "8") never appears.
    assert '"code": "7"' in prompt
    assert '"code": "8"' not in prompt.split("HCPCS candidates:")[0]


def test_answer_prompt_includes_query_output_and_candidates():
    prompt = build_answer_prompt(
        user_query="best hospital for a colonoscopy?",
        output={"cpt_list": ["45378"]},
        code_plausible={"match_count": 95},
    )

    assert "best hospital for a colonoscopy?" in prompt
    assert "45378" in prompt
    assert "95" in prompt
    assert "Answer:" in prompt
