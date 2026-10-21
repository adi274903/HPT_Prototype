"""Robust JSON extraction from LLM responses + candidate compaction.

``parse_json_output`` and ``compact_candidates`` are carried over verbatim
from notebook cells 19 and 16 respectively.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import config

__all__ = ["parse_json_output", "compact_candidates", "recover_selection",
           "split_reasoning"]

_THOUGHT_OPEN = re.compile(r"<unused94>", re.IGNORECASE)
_THOUGHT_CLOSE = re.compile(r"<unused95>", re.IGNORECASE)
_ANY_CONTROL = re.compile(r"</?unused\d+>", re.IGNORECASE)


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_reasoning(raw_output: Any) -> tuple[str, str]:
    """Divide a MedGemma answer into ``(reasoning, answer)``.

    MedGemma opens its scratchpad with ``<unused94>thought`` and closes it with
    ``<unused95>``. Both the tokens and the text between them are reasoning,
    not an answer: read by a patient it sounds like the tool talking to itself.
    The reasoning is kept so the UI can show it in a "Thinking" tab; the answer
    never contains it.

    Either side may be empty. An unclosed reasoning block means generation
    stopped before an answer existed, so the answer comes back empty rather
    than half a thought.
    """
    if raw_output is None:
        return "", ""
    if not isinstance(raw_output, str):
        return "", str(raw_output)

    text = raw_output
    opener = _THOUGHT_OPEN.search(text)

    if opener:
        tail = text[opener.end() :]
        closer = _THOUGHT_CLOSE.search(tail)
        if closer:
            reasoning = tail[: closer.start()]
            answer = tail[closer.end() :]
        else:
            reasoning, answer = tail, ""
        # The sentinel word itself is not reasoning.
        reasoning = re.sub(r"^\s*thought\b", "", reasoning, flags=re.IGNORECASE)
        return _tidy(reasoning), _tidy(answer)

    closer = _THOUGHT_CLOSE.search(text)
    if closer:
        # Some templates consume the opener before we see the text.
        return _tidy(text[: closer.start()]), _tidy(text[closer.end() :])

    return "", _tidy(_ANY_CONTROL.sub("", text))


def strip_thought(raw_output: Any) -> str:
    """Drop the reasoning trace, keeping whatever text follows it.

    The orchestrator path only ever wants the answer half. It deliberately differs
    from :func:`split_reasoning` on one input: an *unclosed* thought block. The
    pipeline must not price a half-finished thought, so it returns an empty answer;
    the router is reading a one-word classification like ``<unused94>price`` and
    wants the word, so here the control tokens are simply stripped.
    """
    text = str(raw_output or "")
    closer = _THOUGHT_CLOSE.search(text)
    if closer:
        return _tidy(text[closer.end() :])
    return _tidy(_ANY_CONTROL.sub("", text))


def parse_json_output(raw_output: Any) -> Dict[str, Any]:
    """Extract and parse one JSON object from an LLM response.

    Handles:
    - plain JSON
    - ```json ... ``` fenced JSON
    - reasoning/text before JSON
    - MedGemma control tokens such as <unused94>, <unused95>
    """
    if isinstance(raw_output, dict):
        return raw_output

    if raw_output is None:
        raise ValueError("Model returned None.")

    if not isinstance(raw_output, str):
        raise TypeError(f"Expected str or dict; got {type(raw_output).__name__}")

    text = raw_output.strip()

    if not text:
        raise ValueError("Model returned an empty response.")

    # Remove visible special tokens like <unused94>, <unused95>.
    text = re.sub(r"<unused\d+>", "", text).strip()

    # ------------------------------------------------------------
    # First choice: extract content inside ```json ... ``` fences.
    # ------------------------------------------------------------
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_match:
        json_text = fenced_match.group(1).strip()

        try:
            return json.loads(json_text)

        except json.JSONDecodeError as error:
            raise ValueError(
                "Found a fenced JSON block, but it is invalid.\n\n"
                f"JSON block:\n{json_text}"
            ) from error

    # ------------------------------------------------------------
    # Second choice: entire output is regular JSON.
    # ------------------------------------------------------------
    try:
        parsed = json.loads(text)

        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object, not a JSON list/value.")

        return parsed

    except json.JSONDecodeError:
        pass

    # ------------------------------------------------------------
    # Last-resort recovery: extract from first { to last }.
    # ------------------------------------------------------------
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start : end + 1]

        try:
            parsed = json.loads(json_text)

            if not isinstance(parsed, dict):
                raise ValueError("Extracted JSON is not an object.")

            return parsed

        except json.JSONDecodeError as error:
            raise ValueError(
                "Found JSON-like text but could not parse it.\n\n"
                f"Extracted candidate:\n{json_text}"
            ) from error

    raise ValueError(
        "Could not find a JSON object in the model response.\n\n"
        f"Raw model output:\n{raw_output}"
    )


#: A bare CPT (5 digits) or HCPCS Level II (letter + 4 digits) token.
_CODE_TOKEN = re.compile(r"\b(?:[0-9]{5}|[A-Z][0-9]{4})\b")


def recover_selection(
    raw_output: str,
    cpt_codes: Iterable[str],
    hcpcs_codes: Iterable[str],
    limit: int = 5,
) -> Optional[Dict[str, Any]]:
    """Salvage a code decision from a decision step that never emitted JSON.

    MedGemma thinks before it answers, and the thought can spend the whole token
    budget enumerating candidates — the JSON then never arrives. Everything it
    named is still in the text, and a code is only usable if retrieval offered it,
    so the salvage is the intersection: codes mentioned **and** retrieved, in the
    order mentioned, capped like the model's own answer.

    Returns ``None`` when the text names nothing that was retrieved. Then there is
    no decision to recover, and inventing one would be worse than saying so.
    """
    text = (raw_output or "").upper()

    offered_cpt = {str(code).strip().upper() for code in cpt_codes}
    offered_hcpcs = {str(code).strip().upper() for code in hcpcs_codes}

    mentioned = list(dict.fromkeys(_CODE_TOKEN.findall(text)))
    cpt = [code for code in mentioned if code in offered_cpt][:limit]
    hcpcs = [code for code in mentioned if code in offered_hcpcs][:limit]

    if not cpt and not hcpcs:
        return None

    if cpt and hcpcs:
        use_codes = "both"
    else:
        use_codes = "cpt" if cpt else "hcpcs"

    return {"use_codes": use_codes, "cpt_list": cpt, "hcpcs_list": hcpcs,
            "recovered": True}


def compact_candidates(
    candidates: Optional[Sequence[Dict[str, Any]]],
    max_candidates: Optional[int] = None,
    text_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Keep the decision prompt short and remove duplicate codes."""
    max_candidates = (
        config.max_candidates_per_family() if max_candidates is None else max_candidates
    )
    text_limit = config.candidate_text_limit() if text_limit is None else text_limit

    compact: List[Dict[str, Any]] = []
    seen_codes = set()

    for item in candidates or []:
        code = str(item.get("code", "")).strip().upper()

        if not code or code in seen_codes:
            continue

        seen_codes.add(code)

        compact.append(
            {
                "code": code,
                "text": str(item.get("text", ""))[:text_limit],
                "score": round(float(item.get("score", 0.0)), 4),
            }
        )

        if len(compact) >= max_candidates:
            break

    return compact
