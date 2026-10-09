import pytest

from pt_healthcare.parsing import compact_candidates, parse_json_output


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
