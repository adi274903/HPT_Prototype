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
