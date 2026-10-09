"""A structured event stream, alongside the existing logging.

The pipeline logs everything the UI needs and ``ui/pt_serve.py`` parses those log
strings into a timeline. That works, but it only ever yields a *timeline* live:
each step's state, label and timing. Every block's **content** is projected from
``run()``'s return value, so the panels stay empty until the run finishes —
``Raw categorizer output:`` is on stdout at about second 12 of a 200-second run
and nothing surfaces it.

Log strings stay exactly as they are; the vendored UI depends on them, and the
refactor's promise is that they never change. Events are emitted *in addition*, so
a consumer can render a block the moment its stage produces data, and so a tool
call is visible as a step of its own.

A sink is any callable taking one dict. Sinks are not trusted: a broken display
must never take down a run, so a raising sink is recorded and skipped.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Event types, kept as plain strings so a consumer can switch on them without
# importing anything from here.
RUN_STARTED = "run_started"
STAGE_STARTED = "stage_started"
STAGE_CONTENT = "stage_content"
STAGE_FINISHED = "stage_finished"
TOOL_CALLED = "tool_called"
TOOL_RESULT = "tool_result"
RUN_FINISHED = "run_finished"
ERROR = "error"

EVENT_TYPES = (
    RUN_STARTED,
    STAGE_STARTED,
    STAGE_CONTENT,
    STAGE_FINISHED,
    TOOL_CALLED,
    TOOL_RESULT,
    RUN_FINISHED,
    ERROR,
)

Sink = Callable[[Dict[str, Any]], None]


class EventStream:
    """Fan-out for pipeline and tool events.

    Every event is recorded on ``.events`` whatever the sink does, which is what
    makes the stream testable without wiring a UI to it.
    """

    def __init__(self, sink: Optional[Sink] = None) -> None:
        self.sink = sink
        self.events: List[Dict[str, Any]] = []
        self.sink_errors: List[str] = []

    # ------------------------------------------------------------------
    def emit(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """Record an event and hand it to the sink.

        Unknown types are allowed through: a new producer should not have to
        update this module to be observable.
        """
        event: Dict[str, Any] = {"type": event_type}
        event.update(fields)

        self.events.append(event)

        if self.sink is not None:
            try:
                self.sink(event)
            except Exception as exc:  # a display failure is not a run failure
                self.sink_errors.append(repr(exc))

        return event

    # ------------------------------------------------------------------
    def of_type(self, *event_types: str) -> List[Dict[str, Any]]:
        """Recorded events matching any of ``event_types``."""
        wanted = set(event_types)
        return [event for event in self.events if event.get("type") in wanted]

    def stage_names(self) -> List[str]:
        """Stages that started, in order, without repeats."""
        names: List[str] = []
        for event in self.of_type(STAGE_STARTED):
            stage = event.get("stage")
            if stage and stage not in names:
                names.append(stage)
        return names

    def tool_names(self) -> List[str]:
        """Tools that were called, in order."""
        return [
            event["tool"]
            for event in self.of_type(TOOL_CALLED)
            if event.get("tool")
        ]


def recorder() -> "tuple[EventStream, List[Dict[str, Any]]]":
    """An :class:`EventStream` whose sink appends to a plain list.

    Convenience for tests and for callers that want the events as they arrive.
    """
    collected: List[Dict[str, Any]] = []
    stream = EventStream(sink=collected.append)
    return stream, collected
