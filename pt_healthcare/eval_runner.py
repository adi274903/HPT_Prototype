"""Score the pipeline against ``eval_dataset`` through a vLLM endpoint.

The runner is deliberately independent of the Qdrant snapshot and of the 2.4 GB
DataFrame: retrieval is a lexical stand-in over the same catalogs and pricing is
duckdb over the same cleaned rows, so a number here reflects the prompts, the
decision step and the grounding, not a GPU-only setup. Every LLM call goes to an
OpenAI-compatible ``/v1/chat/completions`` endpoint, which is what ``vllm serve``
exposes.

What it reports, and why each one exists:

===================  ==================================================================
codes                Sørensen–Dice and Jaccard of the chosen codes against
                     ``ground_truth_code_MRF`` (headline) and ``ground_truth_codes``
rows                 Dice and Jaccard of the returned MRF row ids, raw and per stratum.
                     Raw alone is misleading: a correct two-row answer to a 48-row
                     question scores 0.08, so the 1-row stratum is the honest number.
collapsed rows       Dice over ``(code, hospital, payer)`` tuples — same grading, immune
                     to how many rate lines a plan happens to carry.
sql exactness        did the price lookup return exactly the keyed row set, under both
                     exact-equality and the pipeline's substring name matching
rate fidelity        every ``$`` in the answer must equal a keyed rate (strict) and a
                     returned rate (faithful), plus right-number-wrong-row
abstention           D1/D2: correct silence, and the false-price rate, which is the
                     number that matters for a tool quoting healthcare prices
NER                  per-field precision/recall/F1 for medical, hospital, insurer,
                     medication, after alias resolution — 20% of the set is a correct
                     empty and must not be scored as a hit
intent               per-class precision/recall/F1 and the confusion matrix
retrieval            recall@k and MRR of the candidate list, so a miss is not blamed on
                     the decision step
ops                  parse-failure rate per step, latency, tokens
===================  ==================================================================

Usage::

    vllm serve google/medgemma-1.5-4b-it --port 8000 &
    python -m pt_healthcare.eval_runner --base-url http://localhost:8000/v1 \\
        --model google/medgemma-1.5-4b-it --limit 20 --out eval_run.jsonl

Nothing here writes to the dataset; a run produces one JSONL of per-case records
plus an aggregate printed to stdout and, with ``--report``, to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import duckdb

from .parsing import parse_json_output
from .prompts import (
    build_answer_prompt,
    build_categorizer_prompt,
    build_decision_prompt,
    build_intent_prompt,
)
from .vocabulary import resolve_to_vocabulary

FIELD_ORDER = ("medical", "hospital", "insurer", "medication")


def safe_parse(text: str) -> Dict[str, Any]:
    """``parse_json_output`` but a malformed reply is a recorded failure, not a crash."""
    try:
        return parse_json_output(text) if text else {}
    except Exception:  # noqa: BLE001 - the eval must survive a bad generation
        return {}
INTENT_LABELS = ("price_lookup", "hospital_comparison", "cash_vs_insurance")
AMOUNT_RX = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
TOLERANCE = 0.005  # half a cent: "equal to the cent", not "close enough"
ROW_STRATA = (("1", lambda n: n == 1), ("2-4", lambda n: 2 <= n <= 4),
              ("5-8", lambda n: 5 <= n <= 8), ("9+", lambda n: n >= 9))


# ---------------------------------------------------------------------------
# pure metric helpers — no I/O, so they are unit-testable in isolation
# ---------------------------------------------------------------------------


def dice(predicted: Iterable[str], truth: Iterable[str]) -> Optional[float]:
    """Sørensen–Dice. ``None`` when both sets are empty — report it, don't fake 1.0."""
    left, right = set(predicted), set(truth)
    if not left and not right:
        return None
    return 2 * len(left & right) / (len(left) + len(right))


def jaccard(predicted: Iterable[str], truth: Iterable[str]) -> Optional[float]:
    left, right = set(predicted), set(truth)
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def prf1(predicted: Iterable[str], truth: Iterable[str]) -> Tuple[float, float, float]:
    left, right = set(predicted), set(truth)
    hit = len(left & right)
    precision = hit / len(left) if left else 0.0
    recall = hit / len(right) if right else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def prf1_sets(predicted: Iterable[str], truth: Iterable[str]) -> Tuple[float, float, float]:
    """Entity-level P/R/F1 where "nothing expected, nothing found" is a success.

    36 of the 40 medical targets in the insulin category are legitimately empty; a
    plain P/R/F1 would score each of them 0/0/0 and drag the field's F1 to near
    zero for a model that is behaving perfectly.
    """
    left, right = set(predicted), set(truth)
    if not left and not right:
        return 1.0, 1.0, 1.0
    return prf1(left, right)


def recall_at_k(ranked: Sequence[str], truth: Set[str], k: int) -> Optional[float]:
    if not truth:
        return None
    return len(set(ranked[:k]) & truth) / len(truth)


def reciprocal_rank(ranked: Sequence[str], truth: Set[str]) -> Optional[float]:
    for position, code in enumerate(ranked, start=1):
        if code in truth:
            return 1.0 / position
    return 0.0 if truth else None


def amounts(text: str) -> List[float]:
    return [float(match.replace(",", "")) for match in AMOUNT_RX.findall(text or "")]


def matches_any(value: float, allowed: Iterable[float], tolerance: float = TOLERANCE) -> bool:
    """Equality to the cent, so a float artifact passes and a wrong cent does not."""
    return any(abs(round(value, 2) - round(candidate, 2)) <= tolerance for candidate in allowed)


def normalize_entity(value: str) -> str:
    text = re.sub(r"^(?:a|an|the|my)\s+", "", str(value).strip().strip(".,;:!?\"'"),
                  flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).casefold().strip()


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


# ---------------------------------------------------------------------------
# vLLM / OpenAI-compatible client
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error: Optional[str] = None


class ChatClient:
    """Minimal stdlib client for a vLLM server's OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str, timeout: float = 180.0,
                 max_tokens: int = 512, retries: int = 2, temperature: float = 0.0,
                 seed: int = 13) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self.temperature = temperature
        self.seed = seed
        self.tokens_in = 0
        self.tokens_out = 0

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> Completion:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens or self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        started = time.perf_counter()
        last_error = "unknown"

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                text = parsed["choices"][0]["message"]["content"] or ""
                usage = parsed.get("usage") or {}
                self.tokens_in += int(usage.get("prompt_tokens") or 0)
                self.tokens_out += int(usage.get("completion_tokens") or 0)
                return Completion(text=text,
                                  prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                  completion_tokens=int(usage.get("completion_tokens") or 0),
                                  latency_ms=(time.perf_counter() - started) * 1000)
            except Exception as exc:  # noqa: BLE001 - any transport or shape error is a
                # recorded failure; a run must never die because one call misbehaved.
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5 * (attempt + 1))

        return Completion(text="", prompt_tokens=0, completion_tokens=0,
                          latency_ms=(time.perf_counter() - started) * 1000,
                          error=last_error)


# ---------------------------------------------------------------------------
# retrieval stand-in and pricing
# ---------------------------------------------------------------------------


class CatalogRetriever:
    """Lexical top-k over a catalog — the harness must run without Qdrant.

    Deliberately simple: IDF-weighted token overlap, deterministic ordering. It is
    not a replacement for MedTE and the report says so; it only has to be good
    enough that the decision step gets plausible candidates.
    """

    def __init__(self, catalog: Mapping[str, Mapping[str, str]]) -> None:
        self.catalog = {family: dict(texts) for family, texts in catalog.items()}
        self.tokens = {family: {code: set(re.findall(r"[a-z0-9]+", text.casefold()))
                                for code, text in self.catalog[family].items()}
                       for family in self.catalog}
        self.df: Dict[str, Dict[str, int]] = {}
        for family, per_code in self.tokens.items():
            counts: Dict[str, int] = {}
            for tokens in per_code.values():
                for token in tokens:
                    counts[token] = counts.get(token, 0) + 1
            self.df[family] = counts
        self.size = {family: max(1, len(self.catalog[family])) for family in self.catalog}

    def retrieve(self, text: str, family: str, k: int = 10) -> List[Dict[str, Any]]:
        query_tokens = set(re.findall(r"[a-z0-9]+", (text or "").casefold()))
        if not query_tokens or family not in self.catalog:
            return []
        idf = {token: math.log(1 + self.size[family] / (1 + self.df[family].get(token, 0)))
               for token in query_tokens}
        scored: List[Tuple[float, str]] = []
        for code, tokens in self.tokens[family].items():
            overlap = query_tokens & tokens
            if not overlap:
                continue
            score = sum(idf[token] for token in overlap) / math.sqrt(len(tokens) or 1)
            scored.append((score, code))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{"code": code, "description": self.catalog[family][code], "score": round(score, 6)}
                for score, code in scored[:k]]


class EmbeddingRetriever:
    """The production retriever: MedTE embeddings against the Qdrant snapshot.

    Constructing this loads the embedding model and opens the snapshot, so it is
    opt-in (``--retriever qdrant``); the lexical stand-in exists so the harness can
    be exercised on a laptop. A report always records which one produced it, because
    recall@k and every downstream number mean different things depending on the answer.
    """

    def __init__(self, qdrant_path: Optional[str] = None, top_k: int = 10,
                 device: Optional[str] = None) -> None:
        from . import config
        from .data import get_qdrant_client
        from .models import load_models
        from .retrieval import CodeRetriever

        _, embed_model = load_models(device=device)
        self.retriever = CodeRetriever(embed_model, get_qdrant_client(qdrant_path), top_k=top_k)
        self.collections = {"CPT": config.cpt_collection(), "HCPCS": config.hcpcs_collection()}

    def retrieve(self, text: str, family: str, k: int = 10) -> List[Dict[str, Any]]:
        frame = self.retriever.retrieve_codes(text, self.collections[family], top_k=k)
        if frame is None or getattr(frame, "empty", True):
            return []
        return [{key: value for key, value in record.items()
                 if key in ("code", "description", "score")}
                for record in frame.to_dict(orient="records")]


class Pricer:
    """The price lookup, over duckdb, mirroring ``pricing.sql_answer``.

    ``pricing.sql_answer`` filters a pandas frame with case-insensitive substring
    matching on payer and hospital. That is reproduced here (``substring_names``)
    and also offered as exact equality, because the ground truth was built with
    exact names — the difference between the two is itself a finding.
    """

    def __init__(self, db: Path) -> None:
        self.db = Path(db)
        self.con = duckdb.connect(str(self.db), read_only=True)
        self.hospitals = sorted({row[0] for row in self.con.execute(
            "SELECT DISTINCT hospital FROM code_hospital WHERE hospital IS NOT NULL").fetchall()})
        self.payers = sorted({row[0] for row in self.con.execute(
            "SELECT DISTINCT payer FROM combos WHERE payer IS NOT NULL").fetchall()})

    def close(self) -> None:
        self.con.close()

    def rows(self, codes: Sequence[str], hospitals: Sequence[str], payers: Sequence[str],
             substring_names: bool = False, limit: int = 400) -> List[Dict[str, Any]]:
        codes = [str(code).strip().upper() for code in codes if str(code).strip()]
        if not codes:
            return []

        where = [f"(upper(trim(cpt)) IN ({','.join('?' * len(codes))})"
                 f" OR upper(trim(hcpcs)) IN ({','.join('?' * len(codes))}))"]
        params: List[Any] = [*codes, *codes]

        if hospitals:
            if substring_names:
                where.append("(" + " OR ".join("contains(lower(hospital), ?)" for _ in hospitals) + ")")
                params += [hospital.casefold() for hospital in hospitals]
            else:
                where.append(f"lower(hospital) IN ({','.join('?' * len(hospitals))})")
                params += [hospital.casefold() for hospital in hospitals]

        if payers:
            if substring_names:
                where.append("(" + " OR ".join("contains(lower(payer), ?)" for _ in payers) + ")")
                params += [payer.casefold() for payer in payers]
            else:
                where.append(f"lower(payer) IN ({','.join('?' * len(payers))})")
                params += [payer.casefold() for payer in payers]

        sql = ("SELECT row_id, hospital, payer, plan, setting, cpt, hcpcs, rate, cash "
               "FROM rows_ext WHERE " + " AND ".join(where) + " AND rate IS NOT NULL "
               "ORDER BY rate, row_id LIMIT ?")
        params.append(limit)
        cursor = self.con.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def keyed_rates(self, row_ids: Sequence[str]) -> Tuple[Set[float], Set[float]]:
        if not row_ids:
            return set(), set()
        placeholders = ",".join("?" * len(row_ids))
        rows = self.con.execute(
            f"SELECT rate, cash FROM rows_ext WHERE row_id IN ({placeholders})",
            [str(row_id) for row_id in row_ids]).fetchall()
        rates = {float(rate) for rate, _ in rows if rate is not None}
        cash = {float(value) for _, value in rows if value is not None}
        return rates, cash


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


@dataclass
class Runner:
    client: ChatClient
    retriever: CatalogRetriever
    pricer: Pricer
    top_k: int = 10
    retrieve_family: str = "both"

    def run_case(self, case: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
        query = str(case["eval_query"])
        truth_codes_mrf = {str(code) for code in case.get("ground_truth_code_MRF") or []}
        truth_codes_all = {str(code) for code in case.get("ground_truth_codes") or []}
        truth_rows = {str(row) for row in case.get("ground_truth_row_ids") or []}

        record: Dict[str, Any] = {
            "eval_id": case["eval_id"], "subtype": target.get("subtype"),
            "intent_target": target.get("intent"), "n_rows_gt": len(truth_rows),
            "n_codes_gt": len(truth_codes_mrf),
            "parse_failures": [], "latency_ms": {},
        }

        # 1. intent -----------------------------------------------------------
        intent_completion = self.client.complete(build_intent_prompt(query), max_tokens=64)
        record["latency_ms"]["intent"] = intent_completion.latency_ms
        intent_parsed = safe_parse(intent_completion.text)
        predicted_intent = str(intent_parsed.get("intent", "")).strip()
        if predicted_intent not in INTENT_LABELS:
            record["parse_failures"].append("intent")
            predicted_intent = "invalid"
        record["intent_pred"] = predicted_intent

        # 2. entity extraction ------------------------------------------------
        extraction = self.client.complete(build_categorizer_prompt(query), max_tokens=256)
        record["latency_ms"]["extract"] = extraction.latency_ms
        categorized = safe_parse(extraction.text)
        if not categorized:
            record["parse_failures"].append("extract")
        categorized = {field_name: [str(value) for value in (categorized.get(field_name) or [])]
                       for field_name in FIELD_ORDER}
        record["extraction"] = categorized

        hospitals = resolve_to_vocabulary(categorized["hospital"], self.pricer.hospitals)
        insurers = resolve_to_vocabulary(categorized["insurer"], self.pricer.payers)
        record["resolved"] = {"hospital": hospitals, "insurer": insurers}

        # 3. retrieval --------------------------------------------------------
        medical_text = " ".join([*categorized["medical"], *categorized["medication"]])
        candidate_cpt = (self.retriever.retrieve(medical_text, "CPT", self.top_k)
                         if self.retrieve_family in ("both", "cpt") else [])
        candidate_hcpcs = (self.retriever.retrieve(medical_text, "HCPCS", self.top_k)
                           if self.retrieve_family in ("both", "hcpcs") else [])
        ranked = [item["code"] for item in candidate_cpt + candidate_hcpcs]
        record["retrieval"] = {
            "recall_at_k": recall_at_k(ranked, truth_codes_mrf, len(ranked)),
            "mrr": reciprocal_rank(ranked, truth_codes_mrf),
            "n_candidates": len(ranked),
        }

        # 4. decision ---------------------------------------------------------
        decision_context = {"medical": categorized["medical"], "hospital": categorized["hospital"],
                            "insurer": categorized["insurer"], "medication": categorized["medication"],
                            "cpt_candidates": candidate_cpt, "hcpcs_candidates": candidate_hcpcs}
        decision = self.client.complete(build_decision_prompt(decision_context, query), max_tokens=256)
        record["latency_ms"]["decide"] = decision.latency_ms
        decided = safe_parse(decision.text)
        if not decided:
            record["parse_failures"].append("decide")
        allowed = {item["code"].upper() for item in candidate_cpt + candidate_hcpcs}
        requested = [str(code).strip().upper()
                     for code in [*(decided.get("cpt_list") or []), *(decided.get("hcpcs_list") or [])]
                     if str(code).strip()]
        predicted_codes = sorted({code for code in requested if code in allowed})
        record["hallucinated_codes"] = sorted({code for code in requested if code not in allowed})
        record["codes_pred"] = predicted_codes

        # 5. price lookup -----------------------------------------------------
        rows = self.pricer.rows(predicted_codes, hospitals, insurers, substring_names=False)
        rows_substring = self.pricer.rows(predicted_codes, hospitals, insurers, substring_names=True)
        predicted_rows = {str(row["row_id"]) for row in rows}
        predicted_rows_substring = {str(row["row_id"]) for row in rows_substring}
        record["rows_pred"] = sorted(predicted_rows)
        record["rates_pred"] = sorted({float(row["rate"]) for row in rows if row["rate"] is not None})
        record["cash_pred"] = sorted({float(row["cash"]) for row in rows if row["cash"] is not None})

        # 6. answer -----------------------------------------------------------
        answer_input = {"rows": rows[:10], "filters_used": {"codes": predicted_codes,
                                                           "hospitals": hospitals, "insurers": insurers}}
        answer = self.client.complete(build_answer_prompt(query, {"cpt_list": predicted_codes}, answer_input),
                                      max_tokens=400)
        record["latency_ms"]["answer"] = answer.latency_ms
        text = answer.text or ""
        if not text:
            record["parse_failures"].append("answer")
        record["answer"] = text

        # ---- scores ---------------------------------------------------------
        pivot = str(case["eval_id"]).split("_")[0].upper()
        record["pivot"] = pivot
        record["codes"] = {
            "dice_mrf": dice(predicted_codes, truth_codes_mrf),
            "jaccard_mrf": jaccard(predicted_codes, truth_codes_mrf),
            "dice_all": dice(predicted_codes, truth_codes_all),
            "jaccard_all": jaccard(predicted_codes, truth_codes_all),
            "returned_unpriced": bool(set(predicted_codes) & (truth_codes_all - truth_codes_mrf)),
            "pivot_hit": pivot in predicted_codes,
            "pivot_retrieved": pivot in {code.upper() for code in ranked},
        }
        record["rows"] = {
            "dice_raw": dice(predicted_rows, truth_rows),
            "jaccard_raw": jaccard(predicted_rows, truth_rows),
            "dice_substring": dice(predicted_rows_substring, truth_rows),
            "sql_exact": predicted_rows == truth_rows,
            "sql_exact_substring": predicted_rows_substring == truth_rows,
            "any_hit": bool(predicted_rows & truth_rows),
            "dice_collapsed": dice(collapse(rows), collapse_rows_from_ids(self.pricer, sorted(truth_rows))),
        }

        keyed_rates, keyed_cash = self.pricer.keyed_rates(sorted(truth_rows))
        allowed_truth = keyed_rates | keyed_cash
        allowed_returned = set(record["rates_pred"]) | set(record["cash_pred"])
        cited = amounts(text)
        unknown = [value for value in cited if not matches_any(value, allowed_truth | allowed_returned)]
        record["rates"] = {
            "cited": cited,
            "n_cited": len(cited),
            "grounded_in_key": all(matches_any(value, allowed_truth) for value in cited) if cited else None,
            "grounded_in_rows": all(matches_any(value, allowed_returned) for value in cited) if cited else None,
            "headline_matches_key": (matches_any(cited[0], allowed_truth) if cited else None),
            "right_number_wrong_row": bool(cited and matches_any(cited[0], allowed_truth)
                                           and not (predicted_rows & truth_rows)),
            "hallucinated": unknown,
        }
        record["ner"] = {}
        record["ner_counts"] = {}
        for field_name in FIELD_ORDER:
            predicted_set = {normalize_entity(value) for value in categorized[field_name]}
            truth_set = set(target.get("canonical", {}).get(field_name, []))
            record["ner"][field_name] = prf1_sets(predicted_set, truth_set)
            record["ner_counts"][field_name] = [len(predicted_set), len(truth_set),
                                                len(predicted_set & truth_set)]
        return record


def collapse(rows: Sequence[Mapping[str, Any]]) -> Set[str]:
    """Row identity at the level a patient cares about: (code, hospital, payer)."""
    out: Set[str] = set()
    for row in rows:
        code = str(row.get("cpt") or row.get("hcpcs") or "").strip().upper()
        out.add(f"{code}|{str(row.get('hospital','')).casefold()}|{str(row.get('payer','')).casefold()}")
    return out


def collapse_rows_from_ids(pricer: Pricer, row_ids: Iterable[str]) -> Set[str]:
    row_ids = list(row_ids)
    if not row_ids:
        return set()
    placeholders = ",".join("?" * len(row_ids))
    rows = pricer.con.execute(
        f"SELECT cpt, hcpcs, hospital, payer FROM rows_ext WHERE row_id IN ({placeholders})",
        [str(row_id) for row_id in row_ids]).fetchall()
    return collapse([{"cpt": cpt, "hcpcs": hcpcs, "hospital": hospital, "payer": payer}
                     for cpt, hcpcs, hospital, payer in rows])


# ---------------------------------------------------------------------------
# aggregate report
# ---------------------------------------------------------------------------


def confusion(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for record in records:
        expected = str(record.get("intent_target"))
        predicted = str(record.get("intent_pred"))
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return matrix


def per_class_f1(records: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        predicted = {str(record["eval_id"]) for record in records if record.get("intent_pred") == label}
        truth = {str(record["eval_id"]) for record in records if record.get("intent_target") == label}
        precision, recall, f1 = prf1(predicted, truth)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": len(truth)}
    return out


def ceiling_dice(records: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Best Dice reachable given a key of the whole category and a top-k shortlist.

    With 28 codes keyed and 10 candidates, even a perfect retriever that returns
    all ten tops out at ``2*10/(10+28) = 0.53``. Reporting the ceiling next to the
    score is the difference between "the model is bad" and "the metric is capped".
    """
    values: List[float] = []
    for record in records:
        key = int(record.get("n_codes_gt") or 0)
        k = int(record.get("retrieval", {}).get("n_candidates") or 0)
        if not key or not k:
            continue
        best = min(k, key)
        values.append(2 * best / (k + key))
    return mean(values)


def aggregate(records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> Dict[str, Any]:
    priced = [record for record in records if record["n_rows_gt"] > 0]
    abstain = [record for record in records if record["n_rows_gt"] == 0]
    codes_scored = [record for record in records if record["codes"]["dice_mrf"] is not None]

    stratum_block: Dict[str, Dict[str, Optional[float]]] = {}
    for name, predicate in ROW_STRATA:
        subset = [record for record in priced if predicate(record["n_rows_gt"])]
        stratum_block[name] = {"n": len(subset),
                               "dice_raw": mean(record["rows"]["dice_raw"] for record in subset),
                               "jaccard_raw": mean(record["rows"]["jaccard_raw"] for record in subset)}

    ner_block: Dict[str, Dict[str, Optional[float]]] = {}
    for field_name in FIELD_ORDER:
        precision_values = [record["ner"][field_name][0] for record in records]
        recall_values = [record["ner"][field_name][1] for record in records]
        f1_values = [record["ner"][field_name][2] for record in records]
        predicted_total = sum(record["ner_counts"][field_name][0] for record in records)
        truth_total = sum(record["ner_counts"][field_name][1] for record in records)
        hit_total = sum(record["ner_counts"][field_name][2] for record in records)
        micro_precision = hit_total / predicted_total if predicted_total else None
        micro_recall = hit_total / truth_total if truth_total else None
        ner_block[field_name] = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                   if micro_precision and micro_recall else 0.0 if truth_total else None),
            "f1_per_case": mean(f1_values),
            "exact_match": ratio(sum(1 for record in records if record["ner"][field_name][2] == 1.0), len(records)),
            "n_predicted": predicted_total,
            "n_truth": truth_total,
        }

    cited_records = [record for record in records if record["rates"]["n_cited"] > 0]
    return {
        "run": dict(meta),
        "n_cases": len(records),
        "codes": {
            "n_scored": len(codes_scored),
            "dice_mrf": mean(record["codes"]["dice_mrf"] for record in codes_scored),
            "jaccard_mrf": mean(record["codes"]["jaccard_mrf"] for record in codes_scored),
            "dice_all": mean(record["codes"]["dice_all"] for record in codes_scored),
            "jaccard_all": mean(record["codes"]["jaccard_all"] for record in codes_scored),
            "returned_unpriced_rate": ratio(sum(1 for record in records
                                                if record["codes"]["returned_unpriced"]), len(records)),
            "hallucinated_codes": sum(len(record["hallucinated_codes"]) for record in records),
            "dice_ceiling": ceiling_dice(records),
            "pivot_hit_rate": ratio(sum(1 for record in records if record["codes"]["pivot_hit"]), len(records)),
            "pivot_retrieved_rate": ratio(sum(1 for record in records
                                             if record["codes"]["pivot_retrieved"]), len(records)),
        },
        "rows": {
            "n_scored": len(priced),
            "dice_raw": mean(record["rows"]["dice_raw"] for record in priced),
            "jaccard_raw": mean(record["rows"]["jaccard_raw"] for record in priced),
            "dice_collapsed": mean(record["rows"]["dice_collapsed"] for record in priced),
            "dice_substring_names": mean(record["rows"]["dice_substring"] for record in priced),
            "sql_exact": ratio(sum(1 for record in priced if record["rows"]["sql_exact"]), len(priced)),
            "sql_exact_substring": ratio(sum(1 for record in priced
                                             if record["rows"]["sql_exact_substring"]), len(priced)),
            "any_hit": ratio(sum(1 for record in priced if record["rows"]["any_hit"]), len(priced)),
            "by_stratum": stratum_block,
        },
        "rates": {
            "n_cited": len(cited_records),
            "grounded_in_key": ratio(sum(1 for record in cited_records
                                         if record["rates"]["grounded_in_key"]), len(cited_records)),
            "grounded_in_rows": ratio(sum(1 for record in cited_records
                                          if record["rates"]["grounded_in_rows"]), len(cited_records)),
            "headline_matches_key": ratio(sum(1 for record in cited_records
                                              if record["rates"]["headline_matches_key"]), len(cited_records)),
            "right_number_wrong_row": sum(1 for record in records
                                          if record["rates"]["right_number_wrong_row"]),
            "hallucinated_figures": sum(len(record["rates"]["hallucinated"]) for record in records),
        },
        "abstention": {
            "n": len(abstain),
            "correct_silence": ratio(sum(1 for record in abstain
                                         if not record["rates"]["cited"] and not record["rows_pred"]),
                                     len(abstain)),
            "false_price": ratio(sum(1 for record in abstain
                                     if record["rates"]["cited"] or record["rows_pred"]), len(abstain)),
        },
        "ner": {"per_field": ner_block,
                "macro_f1": mean(value["f1"] for value in ner_block.values()),
                "macro_f1_per_case": mean(value["f1_per_case"] for value in ner_block.values())},
        "intent": {
            "accuracy": ratio(sum(1 for record in records
                                  if record["intent_pred"] == record["intent_target"]), len(records)),
            "per_class": per_class_f1(records, INTENT_LABELS),
            "macro_f1": mean(value["f1"] for value in per_class_f1(records, INTENT_LABELS).values()
                             if value["support"] > 0),
            "labels_present": sorted(label for label, block in
                                     per_class_f1(records, INTENT_LABELS).items() if block["support"] > 0),
            "confusion": confusion(records),
        },
        "retrieval": {
            "recall_at_k": mean(record["retrieval"]["recall_at_k"] for record in records),
            "mrr": mean(record["retrieval"]["mrr"] for record in records),
            "mean_candidates": mean([float(record["retrieval"]["n_candidates"]) for record in records]),
        },
        "ops": {
            "parse_failure_rate": ratio(sum(1 for record in records if record["parse_failures"]), len(records)),
            "parse_failures_by_step": {
                step: sum(1 for record in records if step in record["parse_failures"])
                for step in ("intent", "extract", "decide", "answer")},
            "latency_ms": {step: {"mean": mean(record["latency_ms"][step] for record in records),
                                  "p95": percentile([record["latency_ms"][step] for record in records], 0.95)}
                           for step in ("intent", "extract", "decide", "answer")},
            "tokens": {"prompt": meta.get("prompt_tokens"), "completion": meta.get("completion_tokens")},
        },
    }


def format_report(report: Mapping[str, Any]) -> str:
    def pct(value: Optional[float]) -> str:
        return "—" if value is None else f"{100 * value:.1f}%"

    def num(value: Optional[float]) -> str:
        return "—" if value is None else f"{value:.3f}"

    lines: List[str] = []
    lines.append(f"cases: {report['n_cases']}   model: {report['run'].get('model')}   "
                 f"retriever: {report['run'].get('retriever')} (top_k={report['run'].get('top_k')})")
    lines.append("")
    lines.append("CODES (set overlap vs the answer key)")
    lines.append(f"  Dice    vs ground_truth_code_MRF : {num(report['codes']['dice_mrf'])}")
    lines.append(f"  Jaccard vs ground_truth_code_MRF : {num(report['codes']['jaccard_mrf'])}")
    lines.append(f"  Dice    vs ground_truth_codes    : {num(report['codes']['dice_all'])}")
    lines.append(f"  Jaccard vs ground_truth_codes    : {num(report['codes']['jaccard_all'])}")
    lines.append(f"  returned a code with no priced row: {pct(report['codes']['returned_unpriced_rate'])}")
    lines.append(f"  codes rejected as not retrieved    : {report['codes']['hallucinated_codes']}")
    lines.append(f"  Dice ceiling for top-k vs category key: {num(report['codes']['dice_ceiling'])}")
    lines.append(f"  pivot code selected / retrieved  : {pct(report['codes']['pivot_hit_rate'])} / "
                 f"{pct(report['codes']['pivot_retrieved_rate'])}")
    lines.append("")
    lines.append("ROWS")
    lines.append(f"  Dice raw / Jaccard raw      : {num(report['rows']['dice_raw'])} / {num(report['rows']['jaccard_raw'])}")
    lines.append(f"  Dice collapsed (code,place,payer): {num(report['rows']['dice_collapsed'])}")
    lines.append(f"  SQL row set exact (exact names)   : {pct(report['rows']['sql_exact'])}")
    lines.append(f"  SQL row set exact (substring names): {pct(report['rows']['sql_exact_substring'])}")
    lines.append(f"  any keyed row returned             : {pct(report['rows']['any_hit'])}")
    for name, block in report["rows"]["by_stratum"].items():
        lines.append(f"    {name:>3} keyed rows (n={block['n']:>3}) : Dice {num(block['dice_raw'])}")
    lines.append("")
    lines.append("NEGOTIATED RATE")
    lines.append(f"  cases citing a figure            : {report['rates']['n_cited']}")
    lines.append(f"  every figure is a keyed rate     : {pct(report['rates']['grounded_in_key'])}")
    lines.append(f"  every figure is a returned rate  : {pct(report['rates']['grounded_in_rows'])}")
    lines.append(f"  headline figure is the keyed rate: {pct(report['rates']['headline_matches_key'])}")
    lines.append(f"  right number, wrong row          : {report['rates']['right_number_wrong_row']}")
    lines.append(f"  hallucinated figures             : {report['rates']['hallucinated_figures']}")
    lines.append("")
    lines.append(f"ABSTENTION (n={report['abstention']['n']})")
    lines.append(f"  correct silence : {pct(report['abstention']['correct_silence'])}")
    lines.append(f"  quoted a price anyway: {pct(report['abstention']['false_price'])}")
    lines.append("")
    lines.append("NER (micro over entity mentions)")
    for field_name, block in report["ner"]["per_field"].items():
        lines.append(f"  {field_name:<11} P {num(block['precision'])}  R {num(block['recall'])}  "
                     f"F1 {num(block['f1'])}  exact-set {pct(block['exact_match'])}  "
                     f"({block['n_truth']} truth)")
    lines.append(f"  macro F1: {num(report['ner']['macro_f1'])}")
    lines.append("")
    lines.append("INTENT")
    lines.append(f"  accuracy : {pct(report['intent']['accuracy'])}")
    lines.append(f"  macro F1 : {num(report['intent']['macro_f1'])}")
    for label, block in report["intent"]["per_class"].items():
        lines.append(f"  {label:<19} P {num(block['precision'])}  R {num(block['recall'])}  "
                     f"F1 {num(block['f1'])}  n={block['support']}")
    lines.append("")
    lines.append("RETRIEVAL")
    lines.append(f"  recall@k : {pct(report['retrieval']['recall_at_k'])}")
    lines.append(f"  MRR      : {num(report['retrieval']['mrr'])}")
    lines.append("")
    lines.append("OPS")
    lines.append(f"  cases with a parse failure: {pct(report['ops']['parse_failure_rate'])}")
    lines.append(f"  by step: {report['ops']['parse_failures_by_step']}")
    for step, block in report["ops"]["latency_ms"].items():
        lines.append(f"  {step:<7} mean {num((block['mean'] or 0) / 1000)}s  p95 {num((block['p95'] or 0) / 1000)}s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the HPT pipeline against eval_dataset.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--dataset", default="eval_dataset")
    parser.add_argument("--db", default=".cache/mrf_eval.duckdb")
    parser.add_argument("--catalog-dir", default=None, help="dir holding cpt.csv/hcpcs.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-subtype", type=int, default=0,
                        help="take the first N cases of every subtype (keeps D cases in the run)")
    parser.add_argument("--subtypes", default="", help="comma separated, e.g. A1,C1,D1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieve", default="both", choices=["both", "cpt", "hcpcs"])
    parser.add_argument("--retriever", default="lexical", choices=["lexical", "qdrant"],
                        help="lexical needs no GPU; qdrant loads MedTE and the snapshot")
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--out", default="eval_run.jsonl")
    parser.add_argument("--report", default="eval_report.json")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_code_gen"))
    try:
        from categories import load_catalog  # type: ignore
    except ImportError:  # development checkout keeps the generators in eval/
        from eval.categories import load_catalog  # type: ignore

    catalog = load_catalog(Path(args.catalog_dir)) if args.catalog_dir else load_catalog()

    client = ChatClient(args.base_url, args.model)
    pricer = Pricer(Path(args.db))
    if args.retriever == "qdrant":
        retriever: Any = EmbeddingRetriever(qdrant_path=args.qdrant_path, top_k=args.top_k)
    else:
        retriever = CatalogRetriever(catalog)
    runner = Runner(client=client, retriever=retriever, pricer=pricer,
                    top_k=args.top_k, retrieve_family=args.retrieve)

    cases = load_jsonl(dataset / "pt_eval_v2.jsonl")
    targets = {record["eval_id"]: record for record in load_jsonl(dataset / "pt_eval_v2_targets.jsonl")}
    wanted = {value.strip().upper() for value in args.subtypes.split(",") if value.strip()}
    if wanted:
        cases = [case for case in cases if str(case["eval_id"]).rsplit("_", 1)[-1] in wanted]
    if args.per_subtype:
        seen: Dict[str, int] = {}
        picked: List[Dict[str, Any]] = []
        for case in cases:
            subtype = str(case["eval_id"]).rsplit("_", 1)[-1]
            if seen.get(subtype, 0) < args.per_subtype:
                seen[subtype] = seen.get(subtype, 0) + 1
                picked.append(case)
        cases = picked
    if args.limit:
        cases = cases[:args.limit]

    records: List[Dict[str, Any]] = []
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            record = runner.run_case(case, targets[case["eval_id"]])
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(cases)}] {case['eval_id']} codes_dice={record['codes']['dice_mrf']} "
                  f"rows_dice={record['rows']['dice_raw']} intent={record['intent_pred']}", flush=True)

    report = aggregate(records, {"base_url": args.base_url, "model": args.model,
                                 "retriever": args.retriever, "top_k": args.top_k,
                                 "prompt_tokens": client.tokens_in, "completion_tokens": client.tokens_out})
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(format_report(report))
    pricer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
