"""Tests for dispatching a query to a tool."""

import pytest

from pt_healthcare import events as ev
from pt_healthcare.orchestrator import Orchestrator
from pt_healthcare.tools import ToolResult


class FakeTool:
    def __init__(self, name, answer="OK"):
        self.name = name
        self.answer = answer
        self.calls = []

    def run(self, query, intent, stream):
        self.calls.append((query, intent.tool))
        stream.emit(ev.STAGE_STARTED, stage=self.name)

        return ToolResult(tool=self.name, answer=self.answer)


def test_a_price_question_goes_to_the_price_tool():
    price, lookup = FakeTool("price"), FakeTool("lookup")

    out = Orchestrator([price, lookup]).run(
        "what will a colonoscopy cost at UPMC?"
    )

    assert out["tool"] == "price"
    assert price.calls and not lookup.calls


def test_a_procedure_question_goes_to_the_explain_tool():
    price, lookup, explain = FakeTool("price"), FakeTool("lookup"), FakeTool("explain")

    out = Orchestrator([price, lookup, explain]).run(
        "can you explain me what is diagnostic mammography"
    )

    assert out["tool"] == "explain"
    assert explain.calls and not price.calls and not lookup.calls


def test_a_bare_code_goes_to_the_lookup_tool_with_the_code():
    price, lookup = FakeTool("price"), FakeTool("lookup")

    out = Orchestrator([price, lookup]).run("what is CPT 45378?")

    assert out["tool"] == "lookup"
    assert out["intent"].codes == ["45378"]


def test_the_events_a_ui_needs_to_draw_a_step_are_emitted():
    price, lookup = FakeTool("price"), FakeTool("lookup")
    stream, collected = ev.recorder()

    Orchestrator([price, lookup]).run("what is CPT 45378?", stream=stream)

    types = [event["type"] for event in collected]

    assert types[0] == ev.RUN_STARTED
    assert ev.TOOL_CALLED in types
    assert ev.TOOL_RESULT in types
    assert types[-1] == ev.RUN_FINISHED
    assert stream.tool_names() == ["lookup"]
    assert stream.stage_names() == ["intent", "lookup"]


def test_entities_can_route_to_pricing_without_a_second_extraction():
    price, lookup = FakeTool("price"), FakeTool("lookup")

    out = Orchestrator([price, lookup]).run(
        "mammogram please",
        entities={"hospital": ["UPMC Presbyterian"]},
    )

    assert out["tool"] == "price"


def test_the_orchestrator_says_which_tool_it_picked():
    """Otherwise the explain path is silent, and a Colab log cannot tell you
    whether routing ran at all — which is exactly how a correct run looks
    indistinguishable from nothing happening."""
    lines = []

    Orchestrator(
        [FakeTool("price"), FakeTool("lookup"), FakeTool("explain")],
        printer=lines.append,
    ).run("Explain mammography")

    assert any("tool=explain" in line for line in lines)
    assert not any("STEP" in line for line in lines)


def test_the_orchestrator_reports_a_tools_notes():
    lines = []

    class Noted(FakeTool):
        def run(self, query, intent, stream):
            self.calls.append((query, intent.tool))
            return ToolResult(
                tool=self.name,
                answer="A",
                notes=["Not in the local index: A0428"],
            )

    Orchestrator([Noted("price")], printer=lines.append).run("what does it cost")

    assert any("A0428" in line for line in lines)


def test_a_broken_printer_does_not_break_the_run():
    def boom(line):
        raise RuntimeError("no stdout")

    out = Orchestrator([FakeTool("price")], printer=boom).run("what does it cost")

    assert out["tool"] == "price"


def test_the_orchestrator_is_silent_by_default(capsys):
    """A library caller should not get surprise stdout."""
    Orchestrator([FakeTool("price")]).run("what does it cost")

    assert capsys.readouterr().out == ""


def test_a_missing_tool_falls_back_and_says_so():
    only = FakeTool("lookup")
    stream, collected = ev.recorder()

    out = Orchestrator([only]).run("what does a colonoscopy cost?", stream=stream)

    assert out["tool"] == "lookup"
    assert ev.ERROR in [event["type"] for event in collected]


def test_a_tool_failure_is_emitted_and_reraised():
    class Boom(FakeTool):
        def run(self, query, intent, stream):
            raise RuntimeError("GPU fell over")

    stream, collected = ev.recorder()

    with pytest.raises(RuntimeError):
        Orchestrator([Boom("price")]).run("what will it cost?", stream=stream)

    assert ev.ERROR in [event["type"] for event in collected]
