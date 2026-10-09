"""Tests for the tools the orchestrator can run."""

import pandas as pd

from pt_healthcare import router
from pt_healthcare.events import EventStream
from pt_healthcare.tools import ExplainTool, LookupTool, PriceTool

_COLUMNS = ["rank", "score", "code", "text"]


def exact_frame(codes, known):
    """Mirror the real contract: a row per requested code, empty text if absent."""
    rows = []

    for rank, code in enumerate(codes, start=1):
        text = known.get(code, "")

        rows.append(
            {
                "rank": rank,
                "score": 1.0 if text else 0.0,
                "code": code,
                "text": text,
            }
        )

    return pd.DataFrame(rows, columns=_COLUMNS)


class FakeRetriever:
    """Only the exact lookup: the tool no longer searches semantically."""

    def __init__(self, cpt_known=None, hcpcs_known=None):
        self.cpt_known = cpt_known or {}
        self.hcpcs_known = hcpcs_known or {}

    def lookup_cpt(self, codes):
        return exact_frame(list(codes), self.cpt_known)

    def lookup_hcpcs(self, codes):
        return exact_frame(list(codes), self.hcpcs_known)


class FakeEngine:
    def __init__(self, reply="EXPLAINED"):
        self.reply = reply
        self.explain_args = None

    def explain(self, query, context=None):
        self.explain_args = (query, context)
        return self.reply


def test_lookup_fetches_the_exact_descriptor_for_a_named_code():
    retriever = FakeRetriever(cpt_known={"77065": "Diagnostic mammography, unilateral"})
    engine = FakeEngine()

    result = LookupTool(retriever, engine).run(
        "what is 77065?",
        router.Intent(router.LOOKUP, ["77065"]),
        EventStream(),
    )

    assert result.grounded_on == ["CPT 77065"]
    assert "Diagnostic mammography, unilateral" in engine.explain_args[1]
    # A hit in one collection is not a miss in the other.
    assert result.notes == []


def test_lookup_reports_a_code_that_is_not_in_the_index():
    """Better an honest gap than a definition from memory."""
    retriever = FakeRetriever()
    engine = FakeEngine()

    result = LookupTool(retriever, engine).run(
        "what is A0428?",
        router.Intent(router.LOOKUP, ["A0428"]),
        EventStream(),
    )

    assert "A0428: NOT in the local code index" in engine.explain_args[1]
    assert "A0428" in result.notes[0]
    assert result.grounded_on == []


def test_lookup_without_a_code_degrades_to_a_general_answer():
    """Routing sends codeless questions to ExplainTool; this must not blow up."""
    engine = FakeEngine()

    result = LookupTool(FakeRetriever(), engine).run(
        "explain diagnostic mammography",
        router.Intent(router.LOOKUP, []),
        EventStream(),
    )

    assert engine.explain_args == ("explain diagnostic mammography", None)
    assert result.notes == ["No code was named, so nothing was looked up."]
    assert result.grounded_on == []


def test_lookup_flags_a_code_missing_from_both_collections():
    result = LookupTool(FakeRetriever(), FakeEngine()).run(
        "what is A0428?",
        router.Intent(router.LOOKUP, ["A0428"]),
        EventStream(),
    )

    assert result.notes == [
        "Not in the local index: A0428",
        "The code index returned nothing for this question.",
    ]


def test_explain_answers_without_retrieval():
    """No code named means nothing in the index to ground against."""
    engine = FakeEngine("A mammogram is an X-ray image of the breast.")

    result = ExplainTool(engine).run(
        "explain diagnostic mammography",
        router.Intent(router.EXPLAIN, []),
        EventStream(),
    )

    assert result.answer == "A mammogram is an X-ray image of the breast."
    assert engine.explain_args == ("explain diagnostic mammography", None)
    assert "nothing was looked up" in result.notes[0]


def test_explain_emits_its_answer_as_stage_content():
    stream = EventStream()

    ExplainTool(FakeEngine()).run(
        "what is a colonoscopy?",
        router.Intent(router.EXPLAIN, []),
        stream,
    )

    stages = [event["stage"] for event in stream.of_type("stage_content")]

    assert stages == ["explain"]


def test_lookup_emits_the_entries_as_stage_content():
    retriever = FakeRetriever(cpt_known={"77065": "Diagnostic mammography"})
    stream = EventStream()

    LookupTool(retriever, FakeEngine()).run(
        "what is 77065?",
        router.Intent(router.LOOKUP, ["77065"]),
        stream,
    )

    contents = stream.of_type("stage_content")
    stages = [event["stage"] for event in contents]

    assert "lookup" in stages
    assert contents[0]["content"]["entries"] == [
        ("CPT", "77065", "Diagnostic mammography")
    ]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, query, stream=None):
        self.calls.append(query)

        return {
            "answer": "PRICED",
            "categorized": {"medical": ["mammogram"]},
        }


def test_price_tool_keeps_the_orchestration_as_its_body():
    pipeline = FakePipeline()

    result = PriceTool(pipeline).run(
        "cost of a mammogram",
        router.Intent(router.PRICE),
        EventStream(),
    )

    assert pipeline.calls == ["cost of a mammogram"]
    assert result.answer == "PRICED"
    assert result.payload["categorized"] == {"medical": ["mammogram"]}


def test_price_tool_carries_the_pipeline_payload_through_untouched():
    """The UI projects this dict; the tool must not reshape it."""
    pipeline = FakePipeline()

    result = PriceTool(pipeline).run(
        "cost of a mammogram",
        router.Intent(router.PRICE),
        EventStream(),
    )

    assert set(result.payload) == {"answer", "categorized"}
