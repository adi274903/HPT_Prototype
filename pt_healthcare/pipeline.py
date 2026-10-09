"""End-to-end orchestration (notebook cell 21).

The 7-step workflow:

    1. Categorize / extract entities
    2. Retrieve CPT candidates
    3. Retrieve HCPCS candidates
    4. Let the decision model choose candidate codes
    5. Validate selected codes against the retrieved candidates
    6. Filter the UPMC MRF data
    7. Generate the final response

The notebook used IPython's ``display``; here that is a pluggable hook so the
same pipeline works in a notebook, a CLI or a test.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

from . import config
from . import events as ev
from .pricing import sql_answer
from .utils import clean_list
from .vocabulary import column_vocabulary, drop_placeholders, resolve_to_vocabulary

# A display hook receives a DataFrame and returns nothing.
DisplayHook = Callable[[pd.DataFrame], None]


def _default_display(df: pd.DataFrame) -> None:
    print(df.head(10).to_string())


class HealthcarePricingPipeline:
    """Wire a retriever, an LLM engine and an MRF DataFrame into one pipeline."""

    def __init__(
        self,
        retriever: Any,
        engine: Any,
        mrf_data: pd.DataFrame,
        top_k: Optional[int] = None,
        printer: Callable[[str], None] = print,
        display_fn: Optional[DisplayHook] = None,
    ) -> None:
        self.retriever = retriever
        self.engine = engine
        self.mrf_data = mrf_data
        self.top_k = config.top_k() if top_k is None else top_k
        self.printer = printer
        self.display_fn = display_fn or _default_display

        # Distinct hospitals/payers from the MRF, computed on first use and then
        # reused: they are injected into the categorizer prompt as a closed
        # vocabulary.
        self._hospitals: Optional[List[str]] = None
        self._payers: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def hospital_names(self) -> List[str]:
        """Distinct hospital names in the MRF (cached)."""
        if self._hospitals is None:
            self._hospitals = column_vocabulary(self.mrf_data, "hospital_name")
        return self._hospitals

    def payer_names(self) -> List[str]:
        """Distinct payer names in the MRF (cached)."""
        if self._payers is None:
            self._payers = column_vocabulary(self.mrf_data, "payer_name")
        return self._payers

    # ------------------------------------------------------------------
    def run(
        self,
        user_query: str,
        verbose: bool = True,
        stream: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the full workflow and return a result dictionary.

        Pass ``stream`` (a :class:`pt_healthcare.events.EventStream`) to receive
        structured events as each stage produces its output. The log strings are
        identical either way — the vendored UI parses them.
        """
        pipeline_start = time.perf_counter()

        def log(message: str) -> None:
            if verbose:
                self.printer(message)

        def emit(event_type: str, **fields: Any) -> None:
            """A structured event alongside the log lines.

            The log lines only ever give a consumer a *timeline*: every block's
            content is projected from this method's return value, so it arrives
            only when the run ends. These events carry the same content the moment
            a stage produces it.
            """
            if stream is not None:
                stream.emit(event_type, **fields)

        def show_df(df: pd.DataFrame, label: str, max_rows: int = 5) -> None:
            if not verbose:
                return

            print(f"\n{label}")
            print(f"Rows: {len(df):,}")

            if df is None or df.empty:
                print("No rows returned.")
                return

            columns_to_show = [
                column
                for column in ["rank", "score", "code", "text"]
                if column in df.columns
            ]

            if columns_to_show:
                self.display_fn(df[columns_to_show].head(max_rows))
            else:
                self.display_fn(df.head(max_rows))

        # ---------------------------------------------------------
        # 1. Categorize
        # ---------------------------------------------------------
        log("=" * 80)
        log("STEP 1/7 — Extracting medical, hospital, insurer, and medication entities")
        log(f"User query: {user_query}")

        emit(ev.STAGE_STARTED, stage="entities", detail="Extracting entities")

        t0 = time.perf_counter()

        raw_categorization = self.engine.categorize(
            user_query,
            hospitals=self.hospital_names(),
            insurers=self.payer_names(),
        )

        log("\nRaw categorizer output:")
        log(repr(raw_categorization))

        from .parsing import parse_json_output

        categorized = parse_json_output(raw_categorization)

        categorized.setdefault("medical", [])
        categorized.setdefault("hospital", [])
        categorized.setdefault("insurer", [])
        categorized.setdefault("medication", [])

        categorizer_attempts = 1

        # The hospital and insurer categories are handed a closed set of real
        # values; medical and medication are not. The model has repeatedly filled
        # the two categories that have lists and returned `medical: []` for a query
        # that plainly names a procedure — "Cost of colonoscopy at UPMC Presby?"
        # came back as medical [] with the facility extracted correctly. Those runs
        # also finished in ~2.5-2.9s with no reasoning trace, against 16-24s when
        # the model reasoned and extracted correctly.
        #
        # So: when no procedure comes back, ask once more with the reference lists
        # omitted — that list is the one variable that differs between the two
        # categories, and the cost is ~3s against an answer built on the whole
        # facility. This is a hypothesis being tested in production, not a proven
        # fix, so it reports itself in the log and in the result.
        if not categorized["medical"]:
            log("\nNo medical entity extracted — retrying without the reference lists.")

            raw_retry = self.engine.categorize(user_query)
            retry = parse_json_output(raw_retry)

            categorizer_attempts = 2

            if retry.get("medical"):
                log(f"Retry extracted: medical={retry['medical']}")
                categorized["medical"] = retry["medical"]
            else:
                log("Retry extracted nothing either — continuing without a procedure.")

        log(f"\nCategorization completed in {time.perf_counter() - t0:.2f}s")
        log(f"Medical terms: {categorized['medical']}")
        log(f"Hospitals: {categorized['hospital']}")
        log(f"Insurers: {categorized['insurer']}")
        log(f"Medications: {categorized['medication']}")

        emit(ev.STAGE_CONTENT, stage="entities", content=categorized)
        emit(ev.STAGE_FINISHED, stage="entities")

        # ---------------------------------------------------------
        # 2. Retrieve CPT
        # ---------------------------------------------------------
        log("\n" + "=" * 80)
        log("STEP 2/7 — Retrieving CPT candidates from Qdrant")

        emit(ev.STAGE_STARTED, stage="cpt", detail="Retrieving CPT candidates")

        t0 = time.perf_counter()

        cpt_candidates = self.retriever.retrieve_cpt(categorized, top_k=self.top_k)

        log(f"CPT retrieval completed in {time.perf_counter() - t0:.2f}s")

        emit(
            ev.STAGE_CONTENT,
            stage="cpt",
            content=cpt_candidates.to_dict(orient="records"),
        )
        emit(ev.STAGE_FINISHED, stage="cpt")
        show_df(cpt_candidates, "Top CPT candidates:")

        # ---------------------------------------------------------
        # 3. Retrieve HCPCS
        # ---------------------------------------------------------
        log("\n" + "=" * 80)
        log("STEP 3/7 — Retrieving HCPCS candidates from Qdrant")

        emit(ev.STAGE_STARTED, stage="hcpcs", detail="Retrieving HCPCS candidates")

        t0 = time.perf_counter()

        hcpcs_candidates = self.retriever.retrieve_hcpcs(categorized, top_k=self.top_k)

        log(f"HCPCS retrieval completed in {time.perf_counter() - t0:.2f}s")

        emit(
            ev.STAGE_CONTENT,
            stage="hcpcs",
            content=hcpcs_candidates.to_dict(orient="records"),
        )
        emit(ev.STAGE_FINISHED, stage="hcpcs")
        show_df(hcpcs_candidates, "Top HCPCS candidates:")

        # ---------------------------------------------------------
        # 4. Code decision
        # ---------------------------------------------------------
        log("\n" + "=" * 80)
        log("STEP 4/7 — Asking decision model to select relevant candidate codes")

        emit(ev.STAGE_STARTED, stage="decision", detail="Selecting candidate codes")

        decision_context: Mapping[str, Any] = {
            **categorized,
            "cpt_candidates": cpt_candidates.to_dict(orient="records"),
            "hcpcs_candidates": hcpcs_candidates.to_dict(orient="records"),
        }

        t0 = time.perf_counter()

        raw_output = self.engine.decide(
            categorization=decision_context,
            user_query=user_query,
        )

        log("\nRaw decision-model output:")
        log(repr(raw_output))

        output = parse_json_output(raw_output)

        log(f"\nDecision completed in {time.perf_counter() - t0:.2f}s")
        log("Unvalidated decision:")
        log(str(output))

        emit(ev.STAGE_CONTENT, stage="decision", content=output)
        emit(ev.STAGE_FINISHED, stage="decision")

        # ---------------------------------------------------------
        # 5. Validate codes
        # ---------------------------------------------------------
        log("\n" + "=" * 80)
        log("STEP 5/7 — Validating selected codes against Qdrant candidates")

        emit(ev.STAGE_STARTED, stage="validation", detail="Validating selected codes")

        t0 = time.perf_counter()

        retrieved_cpt_codes = set(
            cpt_candidates.get("code", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        retrieved_hcpcs_codes = set(
            hcpcs_candidates.get("code", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        requested_cpt = clean_list(output.get("cpt_list", []))
        requested_hcpcs = clean_list(output.get("hcpcs_list", []))

        selected_cpt = [
            code.upper() for code in requested_cpt if code.upper() in retrieved_cpt_codes
        ]

        selected_hcpcs = [
            code.upper()
            for code in requested_hcpcs
            if code.upper() in retrieved_hcpcs_codes
        ]

        output["cpt_list"] = list(dict.fromkeys(selected_cpt))
        output["hcpcs_list"] = list(dict.fromkeys(selected_hcpcs))

        rejected_cpt = [
            code for code in requested_cpt if code.upper() not in retrieved_cpt_codes
        ]

        rejected_hcpcs = [
            code for code in requested_hcpcs if code.upper() not in retrieved_hcpcs_codes
        ]

        log(f"Validated in {time.perf_counter() - t0:.2f}s")
        log(f"Selected CPT codes: {output['cpt_list']}")
        log(f"Selected HCPCS codes: {output['hcpcs_list']}")

        if rejected_cpt:
            log(f"Rejected CPT codes not found in retrieval results: {rejected_cpt}")

        if rejected_hcpcs:
            log(
                "Rejected HCPCS codes not found in retrieval results: "
                f"{rejected_hcpcs}"
            )

        emit(
            ev.STAGE_CONTENT,
            stage="validation",
            content={
                "cpt_list": output["cpt_list"],
                "hcpcs_list": output["hcpcs_list"],
                "rejected_cpt": rejected_cpt,
                "rejected_hcpcs": rejected_hcpcs,
            },
        )
        emit(ev.STAGE_FINISHED, stage="validation")

        # ---------------------------------------------------------
        # Correct the model's entity names, before they become filters.
        # ---------------------------------------------------------
        # 1. A bare category word ("hospital", "my insurance") is not a name. As a
        #    substring match it can quietly select an arbitrary slice of the file,
        #    so drop it rather than let it constrain the query.
        # 2. That filter is a substring match, so a near miss is a total miss:
        #    "Highmark Plan" does not match "Highmark BCBS of PA", and the query
        #    comes back with no rows. Snap what is recognisable onto the real value.
        raw_hospitals = list(categorized["hospital"] or [])
        raw_insurers = list(categorized["insurer"] or [])

        hospitals = resolve_to_vocabulary(
            drop_placeholders(raw_hospitals, "hospital"),
            self.hospital_names(),
        )
        insurers = resolve_to_vocabulary(
            drop_placeholders(raw_insurers, "insurer"),
            self.payer_names(),
        )

        if hospitals != raw_hospitals:
            log(f"Hospitals resolved to price-file entries: {raw_hospitals} -> {hospitals}")

        if insurers != raw_insurers:
            log(f"Insurers resolved to price-file entries: {raw_insurers} -> {insurers}")

        # ---------------------------------------------------------
        # Guard: with nothing to narrow by, the filter prices the entire file.
        # ---------------------------------------------------------
        # sql_answer() starts from "every row" and only narrows. With no codes, no
        # payer and no hospital it narrows nothing and returns the whole MRF, which
        # the answer step would then present as the cost of whatever was asked. This
        # is reachable in practice: when extraction finds no procedure, step 2 runs
        # on an empty query and the decision has nothing to select.
        #
        # A payer or hospital filter alone is enough to answer usefully, so this
        # fires only when there is genuinely nothing left to price against.
        if (
            not output["cpt_list"]
            and not output["hcpcs_list"]
            and not hospitals
            and not insurers
        ):
            total_time = time.perf_counter() - pipeline_start

            log("\n" + "=" * 80)
            log("NO PRICE ANSWER — nothing in this question could be priced.")
            log("No billing code, payer or hospital was identified, so filtering the")
            log("MRF would match the entire published file. The run stops here")
            log("rather than presenting unrelated prices as an answer.")
            log("=" * 80)

            emit(
                ev.STAGE_CONTENT,
                stage="mrf",
                content={"stopped": "nothing_to_filter", "match_count": 0},
            )
            emit(ev.STAGE_FINISHED, stage="mrf")

            answer = (
                "I could not identify a medical service or procedure in that "
                "question, so there is nothing to price.\n\n"
                "Try naming the test or service directly — for example \"How much "
                "does a diagnostic mammogram cost?\" — and the hospital and "
                "insurance plan if you know them."
            )

            log("\nFINAL ANSWER:")
            log(answer)

            return {
                "categorized": categorized,
                "cpt_candidates": cpt_candidates,
                "hcpcs_candidates": hcpcs_candidates,
                "output": output,
                "code_plausible": {
                    "matches": self.mrf_data.iloc[0:0].copy(),
                    "price_summary": self.mrf_data.iloc[0:0].copy(),
                    "match_count": 0,
                    "filters_used": {
                        "cpt_codes": [],
                        "hcpcs_codes": [],
                        "insurers": insurers,
                        "hospitals": hospitals,
                    },
                },
                "answer": answer,
                "total_seconds": total_time,
                "categorizer_attempts": categorizer_attempts,
                "stopped": "nothing_to_filter",
            }

        # ---------------------------------------------------------
        # 6. Filter UPMC MRF
        # ---------------------------------------------------------
        log("\n" + "=" * 80)
        log("STEP 6/7 — Filtering UPMC MRF data by selected codes, payer, and hospital")

        emit(ev.STAGE_STARTED, stage="mrf", detail="Filtering the price file")

        t0 = time.perf_counter()

        code_plausible = sql_answer(
            cpt_list=output["cpt_list"],
            hcpcs_list=output["hcpcs_list"],
            insurance_list=insurers,
            hospital_list=hospitals,
            mrf_data=self.mrf_data,
        )

        log(f"MRF filtering completed in {time.perf_counter() - t0:.2f}s")

        if isinstance(code_plausible, dict):
            emit(
                ev.STAGE_CONTENT,
                stage="mrf",
                content={
                    "filters_used": code_plausible.get("filters_used"),
                    "match_count": code_plausible.get("match_count"),
                },
            )

        emit(ev.STAGE_FINISHED, stage="mrf")

        if isinstance(code_plausible, dict):
            match_count = code_plausible.get("match_count", "unknown")
            log(f"MRF rows matched: {match_count}")

            if "filters_used" in code_plausible:
                log(f"MRF filters used: {code_plausible['filters_used']}")

            price_summary = code_plausible.get("price_summary")

            if isinstance(price_summary, pd.DataFrame):
                log("\nTop matching MRF price rows:")

                price_columns = [
                    column
                    for column in [
                        "hospital_name",
                        "payer_name",
                        "plan_name",
                        "description",
                        "CPT",
                        "HCPCS",
                        "standard_charge|gross",
                        "standard_charge|discounted_cash",
                        "standard_charge|negotiated_dollar",
                        "median_amount",
                    ]
                    if column in price_summary.columns
                ]

                if price_summary.empty:
                    log("No MRF price rows matched.")
                else:
                    self.display_fn(price_summary[price_columns].head(10))

        # ---------------------------------------------------------
        # 7. Final user answer
        # ---------------------------------------------------------
        log("\n" + "=" * 80)
        log("STEP 7/7 — Generating final user-facing answer")

        emit(ev.STAGE_STARTED, stage="answer", detail="Writing the answer")

        t0 = time.perf_counter()

        answer = self.engine.answer(
            user_query=user_query,
            output=output,
            code_plausible=code_plausible,
        )

        log(f"Answer generation completed in {time.perf_counter() - t0:.2f}s")
        log("\nFINAL ANSWER:")
        log(str(answer))

        emit(ev.STAGE_CONTENT, stage="answer", content={"answer": answer})
        emit(ev.STAGE_FINISHED, stage="answer")

        total_time = time.perf_counter() - pipeline_start

        log("\n" + "=" * 80)
        log(f"PIPELINE COMPLETE — Total time: {total_time:.2f}s")
        log("=" * 80)

        return {
            "categorized": categorized,
            "cpt_candidates": cpt_candidates,
            "hcpcs_candidates": hcpcs_candidates,
            "output": output,
            "code_plausible": code_plausible,
            "answer": answer,
            "total_seconds": total_time,
            "categorizer_attempts": categorizer_attempts,
        }


def orchestration(user_query: str, verbose: bool = True, **kwargs: Any) -> Dict[str, Any]:
    """Backwards-compatible free function matching the notebook's ``orchestration``.

    Requires the caller to pass ``retriever``, ``engine`` and ``mrf_data`` as
    keyword arguments; prefer constructing ``HealthcarePricingPipeline`` directly.
    """
    retriever = kwargs.pop("retriever")
    engine = kwargs.pop("engine")
    mrf_data = kwargs.pop("mrf_data")
    top_k = kwargs.pop("top_k", None)

    pipeline = HealthcarePricingPipeline(
        retriever=retriever,
        engine=engine,
        mrf_data=mrf_data,
        top_k=top_k,
        **kwargs,
    )

    return pipeline.run(user_query, verbose=verbose)
