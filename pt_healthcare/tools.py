"""The capabilities the orchestrator can run, as tools.

Three tools, each answering one kind of question:

* :class:`PriceTool` — what something costs. Its body is
  ``HealthcarePricingPipeline``, the existing 7-step orchestration, untouched.
  Nothing here reimplements it; the tool is a wrapper so the orchestrator can
  treat pricing like any other capability.
* :class:`LookupTool` — what a *specific* billing code means. The query named a
  code, so the descriptor is fetched exactly from the index and the answer is
  grounded in it. A code the index does not hold is reported as absent rather than
  described from memory.
* :class:`ExplainTool` — a general question about a procedure, test or condition,
  with no code named. No retrieval: MedGemma knows this material, and there is no
  code to ground against.

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
    """Explain a billing code the query named, grounded in the code index.

    The descriptor is fetched exactly — a payload-filtered point lookup, not a
    semantic search — so we get that code's definition rather than its nearest
    neighbours. A code the index does not hold is reported as absent rather than
    described from memory. That is not pedantry: this store is missing roughly a
    third of its HCPCS codes, and CPT is reissued every January.

    A question with no code in it belongs to :class:`ExplainTool`, not here.
    """

    name = router.LOOKUP
    description = "What a specific CPT/HCPCS code means."

    def __init__(self, retriever: Any, engine: Any) -> None:
        self.retriever = retriever
        self.engine = engine

    # ------------------------------------------------------------------
    def _exact(self, codes: Sequence[str]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """(family, code, text) for codes found, and the codes absent everywhere.

        "Absent" means absent from *both* collections. Checking each collection
        independently would report every legitimate hit as missing from the other
        one — and then hand the model both the descriptor and a claim that there
        is no descriptor.
        """
        found: List[Tuple[str, str, str]] = []
        resolved: set = set()

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
                    resolved.add(code.upper())

        missing = [code for code in codes if code.upper() not in resolved]

        return found, missing

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        intent: router.Intent,
        stream: ev.EventStream,
    ) -> ToolResult:
        codes = [str(code).strip().upper() for code in (intent.codes or []) if code]

        if not codes:
            # Routing sends codeless questions to ExplainTool. If one lands here
            # anyway, answer generally rather than failing the request.
            stream.emit(
                ev.STAGE_STARTED,
                stage=self.name,
                detail="No code in the query — answering without the index",
            )

            answer = self.engine.explain(query)

            stream.emit(ev.STAGE_CONTENT, stage=self.name, content={"answer": answer})
            stream.emit(ev.STAGE_FINISHED, stage=self.name)

            return ToolResult(
                tool=self.name,
                answer=answer,
                notes=["No code was named, so nothing was looked up."],
            )

        stream.emit(
            ev.STAGE_STARTED,
            stage=self.name,
            detail=f"Looking up {', '.join(codes)} in the code index",
        )

        entries, missing = self._exact(codes)

        lines = [f"- {family} {code}: {text}" for family, code, text in entries]

        for code in missing:
            lines.append(f"- {code}: NOT in the local code index")

        context = "\n".join(lines) if lines else "(nothing retrieved from the index)"

        stream.emit(
            ev.STAGE_CONTENT,
            stage=self.name,
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
            stage=self.name,
            content={"answer": answer},
        )
        stream.emit(ev.STAGE_FINISHED, stage=self.name)

        return ToolResult(
            tool=self.name,
            answer=answer,
            payload={"entries": entries, "missing": missing, "context": context},
            grounded_on=[f"{family} {code}" for family, code, _ in entries],
            notes=notes,
        )


class ExplainTool(Tool):
    """Answer a general question straight from the model.

    No retrieval, on purpose. MedGemma is a clinical model and knows this
    material, and a question with no code in it has nothing in the index to
    ground against — so there is no reason to search.

    Skipping the search also removes a failure mode. Embedding a conversational
    question ("can you explain me what is diagnostic mammography") produces a far
    worse query vector than embedding an extracted term would, and the price path
    never embeds raw text for exactly that reason. Nothing here needs either.
    """

    name = router.EXPLAIN
    description = "What a procedure, test or condition is, in general terms."

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run(
        self,
        query: str,
        intent: router.Intent,
        stream: ev.EventStream,
    ) -> ToolResult:
        stream.emit(
            ev.STAGE_STARTED,
            stage=self.name,
            detail="Answering from the model — no index lookup",
        )

        # context=None selects the general prompt, which states plainly that no
        # lookup happened, so the model cannot imply it searched.
        answer = self.engine.explain(query)

        stream.emit(ev.STAGE_CONTENT, stage=self.name, content={"answer": answer})
        stream.emit(ev.STAGE_FINISHED, stage=self.name)

        return ToolResult(
            tool=self.name,
            answer=answer,
            notes=["General information from the model; nothing was looked up."],
        )
