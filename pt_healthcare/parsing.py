"""Robust JSON extraction from LLM responses + candidate compaction.

``parse_json_output`` and ``compact_candidates`` are carried over verbatim
from notebook cells 19 and 16 respectively.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .config import CANDIDATE_TEXT_LIMIT, MAX_CANDIDATES_PER_FAMILY


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


def compact_candidates(
    candidates: Optional[Sequence[Dict[str, Any]]],
    max_candidates: int = MAX_CANDIDATES_PER_FAMILY,
    text_limit: int = CANDIDATE_TEXT_LIMIT,
) -> List[Dict[str, Any]]:
    """Keep the decision prompt short and remove duplicate codes."""
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
