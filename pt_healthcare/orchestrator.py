"""Dispatch a patient query to the tool that fits it.

The intent analyst picks the tool; each tool owns one kind of answer. The price
path is not special-cased — it is the ``price`` tool, and its 7-step orchestration
is its body, unchanged.

Events are emitted around every call, so a consumer can render a block as soon as
it has data instead of waiting for the run to finish, and a tool call is visible
as a step of its own.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from . import events as ev
from . import router
from .tools import Tool, ToolResult


class Orchestrator:
    """Pick a tool for the query, run it, and report what happened."""

    def __init__(
        self,
        tools: Iterable[Tool],
        engine: Optional[Any] = None,
    ) -> None:
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools}
        #: Used only for the tie-break when the lexical cues find nothing.
        self.engine = engine

    # ------------------------------------------------------------------
    def tool_names(self) -> List[str]:
        return list(self.tools)

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        stream: Optional[ev.EventStream] = None,
        entities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify, dispatch, and return a summary of the run.

        ``entities`` lets a caller that has already extracted entities (the price
        path does) feed them to the classifier, so naming a facility or payer can
        route to pricing without a second extraction.
        """
        stream = stream if stream is not None else ev.EventStream()

        stream.emit(ev.RUN_STARTED, query=query, tools=self.tool_names())

        intent = router.classify(query, engine=self.engine, entities=entities)

        stream.emit(
            ev.STAGE_STARTED,
            stage="intent",
            detail=f"Selected {intent.tool}",
            tool=intent.tool,
            reason=intent.reason,
            source=intent.source,
            codes=intent.codes,
        )

        tool = self.tools.get(intent.tool)

        if tool is None:
            stream.emit(
                ev.ERROR,
                stage="intent",
                message=f"no tool named {intent.tool!r}",
            )
            tool = self.tools.get(router.LOOKUP) or next(iter(self.tools.values()))
            intent = router.Intent(tool.name, intent.codes, "fallback", "default")

        stream.emit(ev.TOOL_CALLED, tool=tool.name, codes=intent.codes)
        stream.emit(ev.STAGE_FINISHED, stage="intent")

        try:
            result = tool.run(query, intent, stream)
        except Exception as exc:
            stream.emit(ev.ERROR, tool=tool.name, message=repr(exc))
            raise

        stream.emit(
            ev.TOOL_RESULT,
            tool=tool.name,
            answer=result.answer,
            grounded_on=result.grounded_on,
            notes=result.notes,
        )
        stream.emit(ev.RUN_FINISHED, tool=tool.name)

        return {
            "intent": intent,
            "tool": tool.name,
            "answer": result.answer,
            "result": result,
        }
