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


def test_a_thought_block_naming_price_does_not_route_to_price():
    """The real failure: the reasoning named 'price' while rejecting it.

    "What is diagnostic mammography?" reached the price path because the thought
    block said "not its price" and matching ran over the whole response.
    """
    reply = (
        "<unused94>thought\nThe question asks what diagnostic mammography is. "
        "It is not a price question, and no code is named, so it is not a lookup."
        "<unused95>explain"
    )

    intent = router.classify("What is diagnostic mammography?", engine=FakeEngine(reply))

    assert intent.tool == router.EXPLAIN
    assert intent.source == "model"


def test_an_ambiguous_model_answer_defers_to_the_default():
    """Naming two tools is not a decision; the caller's default is safer."""
    reply = "<unused94>thought\nweighing lookup against explain<unused95>lookup or explain"

    intent = router.classify("tell me about colonoscopies", engine=FakeEngine(reply))

    assert intent.tool == router.EXPLAIN
    assert intent.source == "default"


def test_a_rambling_reply_naming_two_tools_defers_to_the_default():
    """With no answer marker there is nothing to trust, so the default stands."""
    reply = "<unused94>thought\nperhaps price, perhaps lookup"  # no <unused95>

    intent = router.classify("tell me about colonoscopies", engine=FakeEngine(reply))

    assert intent.source == "default"


def test_strip_thought_keeps_only_what_follows_the_trace():
    from pt_healthcare.parsing import strip_thought

    raw = "<unused94>thought\nweighing price against lookup<unused95>explain"

    assert strip_thought(raw) == "explain"


def test_strip_thought_without_a_marker_just_drops_the_tokens():
    from pt_healthcare.parsing import strip_thought

    assert strip_thought("<unused94>price") == "price"


def test_strip_thought_of_nothing_is_empty():
    from pt_healthcare.parsing import strip_thought

    assert strip_thought(None) == ""
    assert strip_thought("") == ""


def test_extract_codes_handles_every_code_shape():
    assert router.extract_codes("what is 45378?") == ["45378"]
    assert router.extract_codes("and G0206?") == ["G0206"]
    assert router.extract_codes("a category II code, 0001F") == ["0001F"]
    assert router.extract_codes("no codes in here") == []


def test_extract_codes_dedupes_and_keeps_order():
    assert router.extract_codes("45378 then 77065 then 45378") == ["45378", "77065"]
