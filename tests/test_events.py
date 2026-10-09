"""Tests for the event stream that lets a UI render blocks as they fill."""

from pt_healthcare import events as ev


def test_events_are_recorded_and_handed_to_the_sink():
    stream, collected = ev.recorder()

    stream.emit(ev.STAGE_STARTED, stage="lookup")

    assert stream.events == collected
    assert collected[0]["type"] == ev.STAGE_STARTED
    assert collected[0]["stage"] == "lookup"


def test_a_broken_sink_never_breaks_the_run():
    """A dead socket must not take down a two-hundred-second GPU run."""

    def boom(event):
        raise RuntimeError("socket closed")

    stream = ev.EventStream(sink=boom)

    event = stream.emit(ev.STAGE_STARTED, stage="price")

    assert event["stage"] == "price"
    assert len(stream.sink_errors) == 1


def test_stage_names_are_deduped_in_order():
    stream = ev.EventStream()

    stream.emit(ev.STAGE_STARTED, stage="intent")
    stream.emit(ev.STAGE_STARTED, stage="lookup")
    stream.emit(ev.STAGE_STARTED, stage="intent")

    assert stream.stage_names() == ["intent", "lookup"]


def test_tool_names_are_collected_in_call_order():
    stream = ev.EventStream()

    stream.emit(ev.TOOL_CALLED, tool="lookup")
    stream.emit(ev.TOOL_RESULT, tool="lookup")

    assert stream.tool_names() == ["lookup"]


def test_a_silent_stream_still_records():
    """Runs must be inspectable even with no UI attached."""
    stream = ev.EventStream()

    stream.emit(ev.RUN_STARTED, query="q")

    assert stream.events[0]["query"] == "q"
