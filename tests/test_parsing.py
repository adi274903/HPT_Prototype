import pytest

from pt_healthcare.parsing import (
    compact_candidates,
    parse_json_output,
    recover_selection,
    split_reasoning,
)


def test_parse_plain_json():
    assert parse_json_output('{"medical": ["MRI"]}') == {"medical": ["MRI"]}


def test_parse_dict_passthrough():
    payload = {"medical": ["MRI"]}
    assert parse_json_output(payload) is payload


def test_parse_fenced_json():
    raw = '```json\n{"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []}\n```'
    assert parse_json_output(raw)["cpt_list"] == ["77065"]


def test_parse_medgemma_thought_wrapper():
    raw = (
        "<unused94>thought\nThe procedure is a diagnostic mammogram.<unused95>"
        '```json\n{"use_codes": "cpt", "cpt_list": ["77065"], "hcpcs_list": []}\n```'
    )
    assert parse_json_output(raw)["use_codes"] == "cpt"


def test_parse_reasoning_then_bare_json():
    raw = 'Reasoning first...\n{"medical": ["colonoscopy"]}\ntrailing noise'
    assert parse_json_output(raw) == {"medical": ["colonoscopy"]}


def test_parse_rejects_none_empty_and_non_string():
    with pytest.raises(ValueError):
        parse_json_output(None)

    with pytest.raises(ValueError):
        parse_json_output("   ")

    with pytest.raises(TypeError):
        parse_json_output(123)


def test_parse_rejects_json_list():
    with pytest.raises(ValueError):
        parse_json_output("[1, 2, 3]")


def test_parse_raises_when_no_json_present():
    with pytest.raises(ValueError):
        parse_json_output("there is no json here")


def test_compact_candidates_dedupes_limits_and_truncates():
    candidates = [
        {"code": "77065", "text": "x" * 500, "score": 0.860501},
        {"code": "77065", "text": "duplicate", "score": 0.5},
        {"code": " 77066 ", "text": "diag bilateral", "score": 0.848398},
        {"code": "", "text": "no code", "score": 0.1},
    ]

    compact = compact_candidates(candidates, max_candidates=8, text_limit=10)

    assert [item["code"] for item in compact] == ["77065", "77066"]
    assert len(compact[0]["text"]) == 10
    assert compact[0]["score"] == 0.8605


def test_compact_candidates_respects_max():
    candidates = [{"code": str(i), "text": "t", "score": i} for i in range(20)]

    assert len(compact_candidates(candidates, max_candidates=3)) == 3


def test_compact_candidates_handles_none():
    assert compact_candidates(None) == []


# ── split_reasoning: the scratchpad must never reach the patient ────────────

def test_split_reasoning_takes_the_thought_out_of_the_answer():
    raw = (
        "<unused94>thought\n"
        "Identify the core request: a price for a colonoscopy.\n"
        "CPT candidates are 45378, 45380, 45385.\n"
        "<unused95>\n"
        "A colonoscopy at UPMC Altoona is $1,250 under Cigna Commercial."
    )

    reasoning, answer = split_reasoning(raw)

    assert reasoning.startswith("Identify the core request")
    assert "45378" in reasoning
    assert "<unused" not in reasoning and "<unused" not in answer
    assert answer == "A colonoscopy at UPMC Altoona is $1,250 under Cigna Commercial."


def test_split_reasoning_without_a_closer_produces_no_answer():
    reasoning, answer = split_reasoning("<unused94>thought\nStill weighing candidates 45378")

    assert "Still weighing candidates" in reasoning
    assert answer == ""


def test_split_reasoning_leaves_a_clean_answer_alone():
    reasoning, answer = split_reasoning("  A knee x-ray is $180 at UPMC Horizon.  ")

    assert reasoning == ""
    assert answer == "A knee x-ray is $180 at UPMC Horizon."


def test_split_reasoning_handles_a_bare_closer():
    reasoning, answer = split_reasoning("weighing options<unused95>The rate is $942.")

    assert reasoning == "weighing options"
    assert answer == "The rate is $942."


def test_split_reasoning_survives_missing_or_odd_input():
    assert split_reasoning(None) == ("", "")
    assert split_reasoning("") == ("", "")


def test_strip_thought_is_the_answer_half_of_split_reasoning():
    from pt_healthcare.parsing import split_reasoning, strip_thought

    raw = "<unused94>thought: weigh it up<unused95>It depends on your plan."
    assert strip_thought(raw) == "It depends on your plan."
    assert strip_thought(raw) == split_reasoning(raw)[1]


# ---------------------------------------------------------------------------
# recover_selection: the salvage rule for a decision step that ran out of budget
# ---------------------------------------------------------------------------
def test_recover_selection_keeps_only_retrieved_codes():
    raw = "<unused94>thought\n45378 colonoscopy, 45367 polypectomy, 77065 mammogram"
    salvaged = recover_selection(raw, {"77065"}, set())
    assert salvaged["cpt_list"] == ["77065"]
    assert salvaged["use_codes"] == "cpt"
    assert salvaged["recovered"] is True


def test_recover_selection_reports_both_families():
    salvaged = recover_selection("77065 and G0206", {"77065"}, {"G0206"})
    assert salvaged["use_codes"] == "both"
    assert salvaged["cpt_list"] == ["77065"] and salvaged["hcpcs_list"] == ["G0206"]


def test_recover_selection_dedupes_and_caps():
    raw = "77065 " * 4 + "76090 76091 76092 76093 76094 76095"
    offered = {"77065", "76090", "76091", "76092", "76093", "76094", "76095"}
    salvaged = recover_selection(raw, offered, set(), limit=5)
    assert salvaged["cpt_list"][0] == "77065"
    assert len(salvaged["cpt_list"]) == 5
    assert len(set(salvaged["cpt_list"])) == 5


def test_recover_selection_returns_none_when_nothing_was_offered():
    assert recover_selection("I am not sure what you mean", {"77065"}, set()) is None
    assert recover_selection("45378 and 99999", {"77065"}, set()) is None
