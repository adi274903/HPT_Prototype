"""Thin wrapper around the MedGemma transformers pipeline.

Splits the notebook's three model-calling functions
(``categorizer_prompt``, ``decision_prompt``, ``answer_user``) into a single
engine object that owns the pipeline and the exact call signatures used in the
prototype.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from . import config
from .prompts import (
    build_answer_prompt,
    build_categorizer_prompt,
    build_decision_prompt,
)


class MedGemmaEngine:
    """Encapsulates MedGemma calls for entity extraction, code choice and answering."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe

    # ------------------------------------------------------------------
    # 1. Entity extraction  (notebook cell 15)
    # ------------------------------------------------------------------
    def categorize(self, user_query: str) -> str:
        """Run the categorizer prompt and return the raw assistant text.

        No vocabulary is injected. The known hospital and payer names used to be
        listed inline — about 280 tokens — which made the model fill the two
        categories that had a list and return ``medical: []`` for a query that
        plainly named a procedure. ``vocabulary.resolve_to_vocabulary()`` does that
        mapping after the fact instead, so the prompt does not have to carry it.

        ``max_new_tokens`` is passed explicitly. MedGemma is a thinking model, and
        the pipe's own default of 256 went entirely on the <unused94>thought block
        before any JSON existed.
        """
        prompt = build_categorizer_prompt(user_query)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"}
                ],
            },
        ]

        output = self.pipe(
            text=messages,
            max_new_tokens=config.max_new_tokens(),
            do_sample=False,
        )

        return output[0]["generated_text"][-1]["content"]

    # ------------------------------------------------------------------
    # 2. Code decision  (notebook cell 17)
    # ------------------------------------------------------------------
    def decide(self, categorization: Mapping[str, Any], user_query: str) -> str:
        """Run the decision prompt and return the raw assistant text."""
        prompt = build_decision_prompt(categorization, user_query)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        output = self.pipe(
            messages,
            max_new_tokens=config.max_new_tokens(),
            do_sample=False,
            return_full_text=True,
        )

        generated = output[0]["generated_text"]

        # In chat mode, generated_text is normally the full message history.
        if isinstance(generated, list):
            return generated[-1]["content"]

        # Fallback if the pipeline/version returns a plain string.
        return generated

    # ------------------------------------------------------------------
    # 3. Patient-facing answer  (notebook cell 18)
    # ------------------------------------------------------------------
    def answer(self, user_query: str, output: Any, code_plausible: Any) -> str:
        """Run the answer prompt and return the raw assistant text."""
        prompt = build_answer_prompt(user_query, output, code_plausible)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"}
                ],
            },
        ]

        output_ = self.pipe(
            text=messages,
            max_new_tokens=config.max_new_tokens(),
            do_sample=False,
        )

        return output_[0]["generated_text"][-1]["content"]
