"""Tests for the prompt builders (pure functions)."""

from pt_healthcare.prompts import (
    build_answer_prompt,
    build_categorizer_prompt,
    build_decision_prompt,
)


def test_categorizer_prompt_embeds_query_and_schema():
    prompt = build_categorizer_prompt("What might a diagnostic mammogram cost?")

    assert "What might a diagnostic mammogram cost?" in prompt
    assert '"medical": []' in prompt
    assert '"hospital": []' in prompt
    assert '"insurer": []' in prompt
    assert '"medication" : []' in prompt
    assert "Return ONLY valid JSON" in prompt


def test_categorizer_prompt_no_longer_asks_for_generic_references():
    """It used to say: include generic references such as "my hospital"."""
    prompt = build_categorizer_prompt("q")

    assert "Include generic references" not in prompt
    assert "Never emit a bare category word" in prompt


def test_categorizer_prompt_shows_a_query_that_names_no_facility():
    """The failing case, taught as an example: hospital stays empty."""
    prompt = build_categorizer_prompt("q")

    assert "Which hospital should I go to for a colonoscopy?" in prompt
    assert "A query asking which hospital to" in prompt
    # The old example taught that "Houston Hospital" is a hospital entity.
    assert "Houston Hospital" not in prompt


def test_categorizer_prompt_stays_small():
    """A budget, so the size cannot creep back.

    The prompt carried the whole hospital and payer vocabulary inline — roughly 280
    tokens of names — and with two categories holding a list and two not, the model
    filled the listed ones and returned an empty medical category. The names are no
    longer needed there at all: ``resolve_to_vocabulary()`` maps a near miss onto
    the real entry ("UPMC Presby" -> "Upmc Presbyterian Shadyside").
    """
    prompt = build_categorizer_prompt("Cost of colonoscopy at UPMC Presby?")

    assert len(prompt) < 2200, f"categorizer prompt is {len(prompt)} chars"
    # Four worked examples plus the query under test.
    assert prompt.count("Query: ") == 5
    assert prompt.count("\n") < 60


def test_every_model_call_sets_an_explicit_token_budget():
    """The categorizer must not inherit the pipe's 256-token default.

    MedGemma spends that on its <unused94>thought block before reaching the JSON,
    so a verbose query gets truncated into a response with no JSON at all.
    """
    import inspect

    from pt_healthcare import config
    from pt_healthcare.llm import MedGemmaEngine

    source = inspect.getsource(MedGemmaEngine)

    assert source.count("max_new_tokens=config.max_new_tokens()") == 3, (
        "categorize/decide/answer must all pass an explicit budget"
    )
    assert config.max_new_tokens() > 256


def test_categorizer_prompt_is_unchanged_without_known_values():
    """Kept as a guard: there is no second prompt shape any more.

    The vocabulary variant was removed, so this now asserts the signature takes no
    vocabulary at all — a caller passing one gets a TypeError rather than silently
    different output.
    """
    import pytest

    assert build_categorizer_prompt("q") == build_categorizer_prompt("q")

    with pytest.raises(TypeError):
        build_categorizer_prompt("q", ["Upmc Altoona"], ["Aetna"])


def test_categorizer_prompt_shows_a_fully_populated_price_query():
    """The multi-entity case, taught as a worked example."""
    prompt = build_categorizer_prompt("q")

    assert (
        '"medical": ["MRI"], "hospital": ["Houston Methodist"], '
        '"insurer": ["Blue Cross"]' in prompt
    )


def test_every_model_call_is_deterministic():
    """All three calls must be greedy.

    The categorizer and the answer step both used to sample, so the same query
    produced different entities and different answers on consecutive runs.
    """
    import inspect

    from pt_healthcare.llm import MedGemmaEngine

    source = inspect.getsource(MedGemmaEngine)

    assert source.count("do_sample=False") == 3, (
        "categorize/decide/answer must all be deterministic"
    )


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


def test_categorizer_prompt_covers_the_terse_cost_phrasing():
    """The two reported failures, neither of which resembled any other example.

    "Cost of colonoscopy at UPMC Presby?" and "How much would an X-ray cost?" both
    came back with an empty medical list, while the full-sentence examples — and a
    query matching one verbatim — extracted correctly. The prompt now shows the
    terse shape.
    """
    prompt = build_categorizer_prompt("q")

    assert '"medical": ["colonoscopy"]' in prompt
    assert '"medical": ["X-ray"]' in prompt
    assert "short or clipped phrase still counts" in prompt


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
