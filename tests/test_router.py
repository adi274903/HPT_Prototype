"""Tests for intent analysis — the tool selector."""

from pt_healthcare import router


class FakeEngine:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def classify_intent(self, query: str) -> str:
        return self.reply


class BrokenEngine:
    def classify_intent(self, query: str) -> str:
        raise RuntimeError("no GPU")


def test_price_wording_routes_to_price():
    intent = router.classify(
        "what might a diagnostic mammogram cost at UPMC Presbyterian?"
    )

    assert intent.tool == router.PRICE
    assert intent.source == "rules"


def test_bare_code_routes_to_lookup():
    intent = router.classify("what is CPT 45378?")

    assert intent.tool == router.LOOKUP
    assert intent.codes == ["45378"]


def test_price_wording_beats_a_code():
    """Cost of a code is a price question, not a definition."""
    intent = router.classify("how much does 45378 cost at UPMC?")

    assert intent.tool == router.PRICE
    assert intent.codes == ["45378"]


def test_explaining_a_procedure_routes_to_explain():
    """No code named, so there is nothing to ground against — answer directly."""
    intent = router.classify("can you explain me what is diagnostic mammography")

    assert intent.tool == router.EXPLAIN


def test_naming_a_facility_routes_to_price():
    intent = router.classify(
        "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?"
    )

    assert intent.tool == router.PRICE


def test_a_named_facility_routes_to_price_without_the_word_cost():
    intent = router.classify(
        "mammogram at UPMC Presbyterian",
        entities={"hospital": ["UPMC Presbyterian"]},
    )

    assert intent.tool == router.PRICE


def test_word_boundaries_stop_a_false_price_cue():
    """"fee" sits inside "feeling"; a substring match would misroute this."""
    intent = router.classify("I am feeling anxious about getting a colonoscopy")

    assert intent.tool == router.EXPLAIN


def test_no_cue_without_an_engine_falls_back_to_explain():
    intent = router.classify("tell me about colonoscopies")

    assert intent.tool == router.EXPLAIN
    assert intent.source == "default"


def test_a_model_hint_breaks_the_tie():
    intent = router.classify(
        "tell me about colonoscopies",
        engine=FakeEngine("price"),
    )

    assert intent.tool == router.PRICE
    assert intent.source == "model"


def test_a_model_hint_never_overrides_a_rule():
    intent = router.classify("what does it cost", engine=FakeEngine("lookup"))

    assert intent.tool == router.PRICE
    assert intent.source == "rules"


def test_a_broken_model_degrades_to_the_default():
    """A routing failure must not take down the request."""
    intent = router.classify("tell me about colonoscopies", engine=BrokenEngine())

    assert intent.tool == router.EXPLAIN
    assert intent.source == "default"


def test_model_output_naming_no_tool_falls_back():
    intent = router.classify("tell me about colonoscopies", engine=FakeEngine("banana"))

    assert intent.tool == router.EXPLAIN
    assert intent.source == "default"


def test_extract_codes_handles_every_code_shape():
    assert router.extract_codes("what is 45378?") == ["45378"]
    assert router.extract_codes("and G0206?") == ["G0206"]
    assert router.extract_codes("a category II code, 0001F") == ["0001F"]
    assert router.extract_codes("no codes in here") == []


def test_extract_codes_dedupes_and_keeps_order():
    assert router.extract_codes("45378 then 77065 then 45378") == ["45378", "77065"]
