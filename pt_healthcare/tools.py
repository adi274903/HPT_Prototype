"""The capabilities the orchestrator can run, as tools.

Two tools, each answering one kind of question:

* :class:`PriceTool` — what something costs. Its body is
  ``HealthcarePricingPipeline``, the existing 7-step orchestration, untouched.
  Nothing here reimplements it; the tool is a wrapper so the orchestrator can
  treat pricing like any other capability.
* :class:`LookupTool` — what a billing code means, or what a procedure is. One
  tool for both, because they are the same retrieval twice over: descriptors from
  the code index, then an explanation grounded in them.

Each tool emits events as it produces data, so a consumer can render a block the
moment it has content rather than waiting for the run to finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from . import events as ev
from . import router


@dataclass
class ToolResult:
    """What a tool produced."""

    tool: str
    answer: str
    #: The tool's own structured output, for a caller that wants detail.
    payload: Any = None
    #: Short strings describing what the answer was grounded on.
    grounded_on: List[str] = field(default_factory=list)
    #: Anything the reader should know about the answer's limits.
    notes: List[str] = field(default_factory=list)


class Tool:
    """Base class: a named capability that answers one kind of question."""

    name = "tool"
    description = ""

    def run(
        self,
        query: str,
        intent: router.Intent,
        stream: ev.EventStream,
    ) -> ToolResult:
        raise NotImplementedError


class PriceTool(Tool):
    """Price a procedure against the published MRF.

    The 7-step orchestration stays exactly as it is, including its log lines — the
    vendored UI parses those, and the refactor promised they never change. This
    tool adds events alongside them.
    """

    name = router.PRICE
    description = "What a procedure or service will cost at a named facility/payer."

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def run(
        self,
        query: str,
        intent: router.Intent,
        stream: ev.EventStream,
    ) -> ToolResult:
        # The pipeline emits its own per-stage events when handed the stream, so
        # each of the 7 steps fills its block as it produces data.
        result = self.pipeline.run(query, stream=stream)

        answer = ""
        if isinstance(result, dict):
            answer = str(result.get("answer", ""))

        return ToolResult(tool=self.name, answer=answer, payload=result)


class LookupTool(Tool):
    """Explain a billing code, or find and explain the codes for a procedure.

    Two ways in:

    * the query named codes ("what is CPT 45378?") — exact descriptor fetch, so
      we get that code's definition rather than its nearest neighbours;
    * the query described a procedure ("explain diagnostic mammography") —
      semantic search for candidate codes, then explain those.

    A code the index does not hold is reported as absent rather than described
    from memory. That is not pedantry: this store is missing roughly a third of
    its HCPCS codes, and billing codes are reissued every year.
    """

    name = router.LOOKUP
    description = "What a CPT/HCPCS code means, or what a procedure, test or condition is."

    #: How many candidates to explain when searching by description.
    SEARCH_LIMIT = 5

    def __init__(self, retriever: Any, engine: Any) -> None:
        self.retriever = retriever
        self.engine = engine

    # ------------------------------------------------------------------
    def _exact(self, codes: Sequence[str]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """(family, code, text) for codes found, and the codes that were not."""
        found: List[Tuple[str, str, str]] = []
        missing: List[str] = []

        for family, lookup in (
            ("CPT", self.retriever.lookup_cpt),
            ("HCPCS", self.retriever.lookup_hcpcs),
        ):
            frame = lookup(list(codes))

            for record in frame.to_dict(orient="records"):
                code = str(record.get("code", "")).strip()
                text = str(record.get("text", "")).strip()

                if text:
                    found.append((family, code, text))
                else:
                    missing.append(code)

        return found, missing

    # ------------------------------------------------------------------
    def _search(self, query: str) -> List[Tuple[str, str, str]]:
        """Semantic search for codes matching a described procedure."""
        found: List[Tuple[str, str, str]] = []

        for family, retrieve in (
            ("CPT", self.retriever.retrieve_cpt),
            ("HCPCS", self.retriever.retrieve_hcpcs),
        ):
            frame = retrieve({"medical": [query]})

            if frame is None or frame.empty:
                continue

            for record in frame.head(self.SEARCH_LIMIT).to_dict(orient="records"):
                code = str(record.get("code", "")).strip()
                text = str(record.get("text", "")).strip()

                if code or text:
                    found.append((family, code, text))

        return found

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        intent: router.Intent,
        stream: ev.EventStream,
    ) -> ToolResult:
        codes = [str(code).strip().upper() for code in (intent.codes or []) if code]
        stage = "lookup"

        if codes:
            stream.emit(
                ev.STAGE_STARTED,
                stage=stage,
                detail=f"Looking up {', '.join(codes)} in the code index",
            )
            entries, missing = self._exact(codes)
        else:
            stream.emit(
                ev.STAGE_STARTED,
                stage=stage,
                detail="Searching the code index for matching codes",
            )
            entries, missing = self._search(query), []

        lines = [f"- {family} {code}: {text}" for family, code, text in entries]

        for code in missing:
            lines.append(f"- {code}: NOT in the local code index")

        context = "\n".join(lines) if lines else "(nothing retrieved from the index)"

        stream.emit(
            ev.STAGE_CONTENT,
            stage=stage,
            content={"entries": entries, "missing": missing, "context": context},
        )

        notes: List[str] = []

        if missing:
            notes.append(
                "Not in the local index: " + ", ".join(missing)
            )

        if not entries:
            notes.append("The code index returned nothing for this question.")

        answer = self.engine.explain(query, context)

        stream.emit(
            ev.STAGE_CONTENT,
            stage=stage,
            content={"answer": answer},
        )
        stream.emit(ev.STAGE_FINISHED, stage=stage)

        return ToolResult(
            tool=self.name,
            answer=answer,
            payload={"entries": entries, "missing": missing, "context": context},
            grounded_on=[f"{family} {code}" for family, code, _ in entries],
            notes=notes,
        )
