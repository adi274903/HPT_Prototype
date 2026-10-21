#!/usr/bin/env python3
"""Build eval/pt_eval_v2.jsonl: MRF-grounded questions across ten prompt subtypes.

    ~/.hermes/bin/uv run --with duckdb --with pandas --python .venv/bin/python \
        eval/build_eval_v2.py --per-subtype 60 --seed 13

Design rules (do not relax them):
  * Every answer-key value is a lookup result. No model writes a question or a code.
  * A case that cannot be grounded is skipped, never patched with a plausible value.
  * row_ids are *derived from the constraints stated in the question*, so the key and
    the query cannot drift apart: `resolve()` takes exactly the hospital/payer/plan the
    sentence names, and nothing else.
  * Duplicate MRF rows are already collapsed upstream (scripts/prepare_eval_mrf.py).
  * Deterministic: same --seed -> byte-identical output.

Prompt sets
  S1  A1 full specificity (named service | explicit code), B1 withhold one field,
      B2 payer alias, B3 lay | clinical terminology
  S2  A2 ten personas, A3 ten demographic profiles
  S3  C1 cross-hospital ranking, C2 cash vs insurance
  D   D1 real code with no row at a real hospital, D2 invented service
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aliases import INVENTED_SERVICES, PAYER_ALIASES, PERSONAS, PROFILES  # noqa: E402
from categories import CATEGORIES, load_catalog  # noqa: E402

SUBTYPES = ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2")
SET_OF = {"A1": "S1", "B1": "S1", "B2": "S1", "B3": "S1",
          "A2": "S2", "A3": "S2", "C1": "S3", "C2": "S3", "D1": "D", "D2": "D"}
ROW_CAP = 50
GENERATOR_VERSION = "v2.0.0"
CODE_RX = re.compile(r"^(?:\d{5}|\d{4}[A-Z]|[A-Z]\d{4})$")
STOPWORDS = {"with", "without", "and", "the", "for", "per", "each", "visit",
             "session", "hour", "protocol", "using", "under"}

KEY_ORDER = ("eval_id", "eval_query", "ground_truth_codes", "ground_truth_code_MRF",
             "ground_truth_row_ids", "ground_truth_hospital_name",
             "ground_truth_insurance_provider")


# ── text helpers ────────────────────────────────────────────────────────────
def clinical_text(text: str) -> str:
    """The catalog descriptor, lowercased, qualifiers kept."""
    return " ".join(text.strip().rstrip(".").lower().split())


def money(value: Any) -> Optional[float]:
    try:
        out = round(float(value), 2)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def pick(*options: Sequence[str], rng: random.Random) -> str:
    return rng.choice([o for o in options if o])


def gt_from_query(query: str, hospital: Optional[str], payer: Optional[str],
                  plan: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Keep only the entities the sentence actually names.

    Deriving the answer key from the rendered text (rather than from the generator's
    intent) is what keeps the key and the question from drifting apart when a template
    withholds a field or a persona forgets one.
    """
    low = query.lower()
    return (
        hospital if hospital and hospital.lower() in low else None,
        payer if payer and payer.lower() in low else None,
        plan if plan and plan.lower() in low else None,
    )


# ── data loading ────────────────────────────────────────────────────────────
class Universe:
    """Everything the generators need, sliced out of the prepared duckdb file."""

    def __init__(self, db_path: str, seed: int) -> None:
        self.rng = random.Random(seed)
        self.con = duckdb.connect(db_path, read_only=True)

        codes = sorted({c for cat in CATEGORIES.values() for c in cat.codes})
        marks = ",".join("?" * len(codes))
        self.df = self.con.execute(
            f"""SELECT row_id, code, family, hospital, payer, plan, setting,
                       rate, gross, cash, methodology
                FROM combos WHERE code IN ({marks})
                ORDER BY code, family, hospital, payer, plan, setting, row_id""",
            codes,
        ).fetchdf()

        self.df["row_id"] = self.df["row_id"].astype(str)
        for col in ("hospital", "payer", "plan", "setting", "methodology", "code", "family"):
            self.df[col] = self.df[col].fillna("").astype(str)

        self.hospitals_all = sorted(
            r[0] for r in self.con.execute(
                "SELECT DISTINCT hospital FROM code_hospital"
            ).fetchall()
        )
        self.payers_all = sorted(
            r[0] for r in self.con.execute("SELECT DISTINCT payer FROM combos").fetchall()
        )

        # Full-file coverage, keyed by code string only: a hospital that has even one
        # row for the code in either column is not absent.
        cov = self.con.execute(
            "SELECT code, hospital FROM code_hospital"
        ).fetchdf()
        self.coverage: Dict[str, set] = {
            code: set(group["hospital"]) for code, group in cov.groupby("code")
        }
        self.coverage_all_hospitals = set(self.hospitals_all)
        self.descriptions = [
            r[0].lower() for r in self.con.execute(
                "SELECT DISTINCT description FROM rows_ext WHERE description IS NOT NULL"
            ).fetchall()
        ]

        self.by_code: Dict[str, pd.DataFrame] = {
            code: sub.reset_index(drop=True)
            for code, sub in self.df.groupby("code")
        }

    # -- answer-key derivation ---------------------------------------------
    def resolve(self, code: str, hospital: Optional[str] = None,
                payer: Optional[str] = None, plan: Optional[str] = None,
                setting: Optional[str] = None) -> Tuple[List[str], int]:
        """Row ids matching exactly the constraints a question states.

        Lookup is by code string across both MRF code columns: the HCPCS column holds
        Level I CPT codes for 29% of its rows, so a question naming 74182 must see the
        rows filed under either column.
        """
        sub = self.by_code.get(code)
        if sub is None or sub.empty:
            return [], 0

        mask = pd.Series(True, index=sub.index)
        for column, value in (("hospital", hospital), ("payer", payer),
                              ("plan", plan), ("setting", setting)):
            if value is not None:
                mask &= sub[column] == value

        hits = sub[mask]
        total = len(hits)
        if not total:
            return [], 0

        ordered = hits.sort_values(["rate", "row_id"])["row_id"].tolist()[:ROW_CAP]
        return sorted(set(ordered)), total

    def category_pool(self, name: str) -> List[Dict[str, Any]]:
        cat = CATEGORIES[name]
        owned = set(load_catalog()[cat.family])
        full = sorted(c for c in cat.codes if c in owned)
        rows: List[Dict[str, Any]] = []

        for code in sorted(c for c in full if c in self.by_code):
            for record in self.by_code[code].to_dict("records"):
                # "EPO Partners/EPO Select Plan/HMO" reads as three plans inside a
                # sentence, so those rows never enter the prose templates
                if "/" in record["plan"]:
                    continue
                record["category"] = name
                rows.append(record)

        return rows


# ── the builder ─────────────────────────────────────────────────────────────
class Builder:
    def __init__(self, universe: Universe) -> None:
        self.u = universe
        self.rng = universe.rng
        self.cases: List[Dict[str, Any]] = []
        self.meta: List[Dict[str, Any]] = []
        self.skips: Counter = Counter()
        self.seen_queries: set = set()
        self.seen_units: Dict[str, set] = defaultdict(set)
        self.seq = 0

        owned = load_catalog()
        self.codes_full: Dict[str, List[str]] = {}
        self.codes_mrf: Dict[str, List[str]] = {}

        for name, cat in CATEGORIES.items():
            full = sorted(c for c in cat.codes if c in owned[cat.family])
            self.codes_full[name] = full
            self.codes_mrf[name] = sorted(
                c for c in full if c in self.u.by_code
            )

        self.pools: Dict[str, List[Dict[str, Any]]] = {
            name: self._shuffled(self.u.category_pool(name)) for name in CATEGORIES
        }

    def _shuffled(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = list(rows)
        self.rng.shuffle(rows)
        return rows

    # -- emission ----------------------------------------------------------
    def emit(self, subtype: str, pivot: str, query: str, category: str,
             row_ids: List[str], hospital: Optional[str], payer: Optional[str],
             meta: Dict[str, Any], payer_alias: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not CODE_RX.match(pivot):
            self.skips[f"{subtype}:bad_pivot"] += 1
            return None
        if query in self.seen_queries:
            self.skips[f"{subtype}:duplicate_query"] += 1
            return None
        if hospital is not None and hospital.lower() not in query.lower():
            self.skips[f"{subtype}:hospital_not_in_query"] += 1
            return None
        if payer is not None and (payer_alias or payer).lower() not in query.lower():
            self.skips[f"{subtype}:payer_not_in_query"] += 1
            return None

        self.seq += 1
        eval_id = f"{pivot}_{self.seq:06d}_{subtype}"
        # D2 is a service that does not exist, so it has no accepted code at all
        full_codes = [] if subtype == "D2" else self.codes_full[category]
        mrf_codes = [] if subtype == "D2" else self.codes_mrf[category]
        case = {
            "eval_id": eval_id,
            "eval_query": query,
            "ground_truth_codes": full_codes,
            "ground_truth_code_MRF": mrf_codes,
            "ground_truth_row_ids": row_ids,
            "ground_truth_hospital_name": hospital,
            "ground_truth_insurance_provider": payer,
        }
        assert tuple(case) == KEY_ORDER
        assert set(mrf_codes) <= set(full_codes)
        assert len(row_ids) <= ROW_CAP

        self.seen_queries.add(query)
        self.cases.append(case)
        self.meta.append({
            "eval_id": eval_id, "subtype": subtype, "set_id": SET_OF[subtype],
            "category": category, "pivot_code": pivot,
            "generator_version": GENERATOR_VERSION, "n_rows_total": meta.pop("n_rows_total", len(row_ids)),
            **meta,
        })
        return case

    def draw(self, subtype: str, units: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Round-robin across categories, no unit reused within a subtype."""
        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for unit in units:
            by_cat[unit["category"]].append(unit)

        names = [n for n in sorted(by_cat) if (n, len(by_cat[n]))]
        if not names:
            return None
        self.rng.shuffle(names)

        for name in names:
            queue = by_cat[name]
            while queue:
                unit = queue.pop()
                key = unit["key"]
                if key in self.seen_units[subtype]:
                    continue
                self.seen_units[subtype].add(key)
                return unit

        self.skips[f"{subtype}:pool_exhausted"] += 1
        return None

    def units_from_pool(self) -> List[Dict[str, Any]]:
        out = []
        for name, rows in self.pools.items():
            for r in rows:
                out.append({
                    "category": name, "code": r["code"], "family": CATEGORIES[name].family,
                    "row": r,
                    "key": (r["code"], r["hospital"], r["payer"], r["plan"], r["setting"]),
                })
        return out

    # ── S1 ─────────────────────────────────────────────────────────────────
    def gen_a1(self, units, index: int) -> None:
        unit = self.draw("A1", units)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        housing = row.get("setting") or "outpatient"
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)

        if index % 2 == 0:  # named service, no code
            query = pick(
                f"I need {lay} at {hospital}. My insurance is {payer} {plan} — what's the negotiated rate?",
                f"What does {hospital} charge for {lay} under {payer} {plan}?",
                f"{lay.capitalize()} at {hospital} with {payer} {plan}: what price did they agree on?",
                rng=self.rng,
            )
        else:  # explicit code
            query = pick(
                f"My doctor's order says CPT {code} for {lay} at {hospital}. My plan is {payer} {plan}. What did they agree on?",
                f"CPT {code}, {lay}, at {hospital}, billed to {payer} {plan} — what is the negotiated amount?",
                rng=self.rng,
            )

        g_hospital, g_payer, _ = gt_from_query(query, hospital, payer, plan)
        row_ids, total = self.u.resolve(code, g_hospital, g_payer, plan)
        self.emit("A1", code, query, unit["category"], row_ids, g_hospital, g_payer,
                  {"plan": plan, "methodology": row["methodology"], "rate": money(row["rate"]),
                   "gross": money(row["gross"]), "cash": money(row["cash"]),
                   "setting": housing, "variant": "named" if index % 2 == 0 else "code",
                   "n_rows_total": total})

    def gen_b1(self, units, index: int) -> None:
        unit = self.draw("B1", units)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)
        withheld = ("hospital", "payer", "plan", "service", "setting")[index % 5]

        if withheld == "hospital":
            query = pick(
                f"How much is {lay} with {payer} {plan}?",
                f"I'm on {payer} {plan} and need {lay} — what's the agreed price?",
                rng=self.rng,
            )
        elif withheld == "payer":
            query = pick(
                f"What does {hospital} charge for {lay} on my {plan} plan? I can't find my card.",
                f"{lay.capitalize()} at {hospital} on my {plan} plan — what's the negotiated rate?",
                rng=self.rng,
            )
        elif withheld == "plan":
            query = pick(
                f"How much is {lay} at {hospital} with {payer}? I can't remember which of their plans I'm on.",
                f"What does {hospital} charge for {lay} under {payer}? My plan tier is somewhere in the portal.",
                rng=self.rng,
            )
        elif withheld == "service":
            query = pick(
                f"{hospital} billed me under {payer} {plan}. What do they charge for the procedure on my order?",
                f"My order is for {hospital} with {payer} {plan} — what's the negotiated price?",
                rng=self.rng,
            )
        else:  # setting withheld
            query = pick(
                f"How much is {lay} at {hospital} with {payer} {plan}? I don't know if it's outpatient or admitted.",
                f"{lay.capitalize()} at {hospital}, {payer} {plan} — is there a different price for day surgery?",
                rng=self.rng,
            )

        g_hospital, g_payer, g_plan = gt_from_query(query, hospital, payer, plan)
        row_ids, total = self.u.resolve(code, g_hospital, g_payer, g_plan)
        if total > ROW_CAP:
            # a withheld field can leave thousands of rows: keep only keys that are
            # complete, so ground_truth_row_ids is the whole answer set
            self.skips["B1:key_truncated"] += 1
            return
        self.emit("B1", code, query, unit["category"], row_ids, g_hospital, g_payer,
                  {"plan": g_plan,
                   "withheld_field": withheld, "methodology": row["methodology"],
                   "rate": money(row["rate"]), "n_rows_total": total})

    def gen_b2(self, units, index: int) -> None:
        candidates = [u for u in units if u["row"]["payer"] in PAYER_ALIASES]
        unit = self.draw("B2", candidates)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        alias = self.rng.choice(PAYER_ALIASES[payer])
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)

        query = pick(
            f"How much is {lay} at {hospital}? I'm on {alias} {plan}.",
            f"{lay.capitalize()} at {hospital} — my coverage is {alias}, {plan} plan. What's the agreed rate?",
            f"I have {alias} ({plan}). What does {hospital} charge for {lay}?",
            rng=self.rng,
        )
        row_ids, total = self.u.resolve(code, hospital, payer, plan)
        self.emit("B2", code, query, unit["category"], row_ids, hospital, payer,
                  {"plan": plan, "alias_used": alias, "methodology": row["methodology"],
                   "rate": money(row["rate"]), "n_rows_total": total}, payer_alias=alias)

    def gen_b3(self, units, index: int) -> None:
        unit = self.draw("B3", units)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        text = load_catalog()[cat.family].get(code, cat.clinical)
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)

        if index % 2 == 0:
            query = pick(
                f"What will {lay} cost me at {hospital} with {payer} {plan}?",
                f"I'm getting {lay} at {hospital}. My plan is {payer} {plan} — what's the rate?",
                rng=self.rng,
            )
            variant = "lay"
        else:
            query = pick(
                f"For {clinical_text(text)} at {hospital} under {payer} {plan}, what is the negotiated amount?",
                f"{hospital} is billing {clinical_text(text)} to {payer} {plan}. What's the agreed price?",
                rng=self.rng,
            )
            variant = "clinical"

        row_ids, total = self.u.resolve(code, hospital, payer, plan)
        self.emit("B3", code, query, unit["category"], row_ids, hospital, payer,
                  {"plan": plan, "variant": variant, "descriptor": text,
                   "methodology": row["methodology"], "rate": money(row["rate"]),
                   "n_rows_total": total})

    # ── S2 ─────────────────────────────────────────────────────────────────
    def gen_a2(self, units, index: int) -> None:
        unit = self.draw("A2", units)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        persona = PERSONAS[index % len(PERSONAS)]
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)
        forgets_hospital = "forgets the hospital" in persona["recall"]
        forgets_plan = "remembers the hospital" in persona["recall"] and "plan" in persona["recall"]

        if persona["id"] == "P05":
            query = f"{lay} {hospital} {payer} {plan} price?"
        elif persona["id"] == "P06":
            query = (f"So my gastroenterologist wants me to get {lay}, and I've been putting it off for a year, "
                     f"and now they want to schedule it at {hospital}. My coverage is {payer} {plan}. "
                     f"What exactly did they agree to charge?")
        elif persona["id"] == "P07":
            query = f"What's the cash price for {lay} at {hospital}? I have {payer} {plan} but I'd rather just pay."
        elif persona["id"] == "P08":
            query = (f"I'm calling for my mother. She's scheduled for {lay} and her card says {payer} {plan}. "
                     f"What's the negotiated rate?")
        elif persona["id"] == "P09":
            query = (f"I've seen three different numbers for {lay}. What did {hospital} actually agree to with "
                     f"{payer} {plan}?")
        elif persona["id"] == "P10":
            alias = self.rng.choice(PAYER_ALIASES.get(payer, [payer]))
            query = f"What does {hospital} charge for {lay}? I think my plan is {alias}? Maybe {plan}?"
        elif forgets_hospital:
            query = f"I need {lay} and my plan is {payer} {plan}. What should it cost?"
        elif forgets_plan:
            query = f"How much is {lay} at {hospital} with {payer}?"
        else:
            query = (f"I'm a little nervous about this — I need {lay} at {hospital} and my plan is {payer} {plan}. "
                     f"Can you tell me what they agreed to charge?")

        low = query.lower()
        hospital_gt = hospital if hospital and hospital.lower() in low else None
        plan_gt = plan if plan and plan.lower() in low else None
        payer_alias = None
        if persona["id"] == "P10":
            payer_alias = self.rng.choice(PAYER_ALIASES.get(payer, [payer]))
            payer_alias = payer_alias if payer_alias.lower() in low else None

        row_ids, total = self.u.resolve(code, hospital_gt, payer, plan_gt)
        self.emit("A2", code, query, unit["category"], row_ids, hospital_gt, payer,
                  {"plan": plan_gt, "persona": persona["id"], "persona_label": persona["label"],
                   "alias_used": payer_alias,
                   "methodology": row["methodology"], "rate": money(row["rate"]),
                   "n_rows_total": total}, payer_alias=payer_alias)

    def gen_a3(self, units, index: int) -> None:
        unit = self.draw("A3", units)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        profile = PROFILES[index % len(PROFILES)]
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)

        query = pick(
            f"I'm a {profile['age']}-year-old {profile['context']}. I have {payer} and my plan is {plan}. "
            f"What would {lay} cost at {hospital}?",
            f"{profile['age']}-year-old {profile['context']} here, on {payer} {plan}. {hospital} quoted me "
            f"something for {lay} — what's the negotiated rate?",
            rng=self.rng,
        )
        row_ids, total = self.u.resolve(code, hospital, payer, plan)
        self.emit("A3", code, query, unit["category"], row_ids, hospital, payer,
                  {"plan": plan, "profile": profile["id"], "profile_context": profile["context"],
                   "profile_framing": profile["framing"], "methodology": row["methodology"],
                   "rate": money(row["rate"]), "n_rows_total": total})

    # ── S3 ─────────────────────────────────────────────────────────────────
    def c1_units(self) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []

        for name, cat in CATEGORIES.items():
            codes = self.codes_mrf[name]
            if not codes:
                continue
            sub = self.u.df[self.u.df["code"].isin(codes)]
            if sub.empty:
                continue
            per_hospital = (sub.groupby(["code", "payer", "hospital"])["rate"].min()
                            .reset_index())

            for (code, payer), group in per_hospital.groupby(["code", "payer"]):
                if len(group) < 2 or group["rate"].nunique() < 2:
                    continue  # ranking with tied minima is a vacuous question
                spread = float(group["rate"].max()) - float(group["rate"].min())
                if spread < 5.0 or spread / float(group["rate"].min()) < 0.05:
                    continue  # $8.58 vs $8.98 is not a reasoning question
                low = group.loc[group["rate"].idxmin()]
                high = group.loc[group["rate"].idxmax()]
                units.append({
                    "category": name, "code": code, "family": cat.family, "payer": payer,
                    "low_hospital": low["hospital"], "high_hospital": high["hospital"],
                    "low_rate": money(low["rate"]), "high_rate": money(high["rate"]),
                    "diff": round(float(high["rate"]) - float(low["rate"]), 2),
                    "ranking": group.sort_values("rate")[["hospital", "rate"]]
                                      .to_dict("records"),
                    "key": (code, payer),
                })

        return units

    def gen_c1(self, units, index: int) -> None:
        unit = self.draw("C1", units)
        if not unit:
            return
        cat = CATEGORIES[unit["category"]]
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)
        low_h, high_h = unit["low_hospital"], unit["high_hospital"]

        query = pick(
            f"I have {unit['payer']}. Is {lay} cheaper at {low_h} or {high_h}? How much more is the expensive one?",
            f"Comparing {low_h} and {high_h} for {lay} on {unit['payer']}: which is cheaper, and by how much?",
            rng=self.rng,
        )

        # answer key: the rows achieving each hospital's minimum rate for this payer
        row_ids: List[str] = []
        for hospital in (low_h, high_h):
            sub = self.u.by_code.get(unit["code"])
            if sub is None:
                continue
            hits = sub[(sub["hospital"] == hospital) & (sub["payer"] == unit["payer"])]
            if hits.empty:
                continue
            floor = hits["rate"].min()
            row_ids += sorted(hits[hits["rate"] == floor]["row_id"].tolist())
        row_ids = sorted(dict.fromkeys(row_ids))[:ROW_CAP]

        meta_ranking = [{"hospital": r["hospital"], "min_rate": money(r["rate"])}
                        for r in unit["ranking"]]
        self.emit("C1", unit["code"], query, unit["category"], row_ids, low_h, unit["payer"],
                  {"plan": None, "ranking": meta_ranking,
                   "cheapest": low_h, "cheapest_rate": unit["low_rate"],
                   "priciest": high_h, "priciest_rate": unit["high_rate"],
                   "difference": unit["diff"], "n_hospitals": len(meta_ranking),
                   "n_rows_total": len(row_ids)})

    def c2_units(self) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []

        for name, cat in CATEGORIES.items():
            codes = self.codes_mrf[name]
            if not codes:
                continue
            sub = self.u.df[self.u.df["code"].isin(codes)]
            sub = sub[(sub["cash"].fillna(0) > 0) & (sub["rate"] > 0)]
            sub = sub[~sub["plan"].str.contains("/", regex=False)]
            for record in sub.to_dict("records"):
                units.append({"category": name, "code": record["code"],
                              "family": cat.family, "row": record,
                              "key": (record["code"], record["hospital"], record["payer"],
                                      record["plan"], record["setting"])})

        return units

    def gen_c2(self, units, index: int) -> None:
        unit = self.draw("C2", units)
        if not unit:
            return
        cat, code, row = CATEGORIES[unit["category"]], unit["code"], unit["row"]
        hospital, payer, plan = row["hospital"], row["payer"], row["plan"]
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)

        query = pick(
            f"Is it cheaper to pay cash or use my {payer} {plan} for {lay} at {hospital}?",
            f"{hospital} quoted {lay} two ways for me — self-pay versus {payer} {plan}. Which one is lower?",
            rng=self.rng,
        )
        row_ids, total = self.u.resolve(code, hospital, payer, plan)
        self.emit("C2", code, query, unit["category"], row_ids, hospital, payer,
                  {"plan": plan, "cash": money(row["cash"]), "negotiated": money(row["rate"]),
                   "gross": money(row["gross"]), "methodology": row["methodology"],
                   "missing_field": None, "setting": row["setting"], "n_rows_total": total})

    # ── D ──────────────────────────────────────────────────────────────────
    def d1_units(self) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        hospitals = self.u.coverage_all_hospitals

        for name, cat in CATEGORIES.items():
            for code in sorted(c for c in cat.codes if c in self.u.coverage):
                present = self.u.coverage[code]
                if len(present) < 2:
                    continue
                absence = sorted(hospitals - present)
                for hospital in absence:
                    units.append({"category": name, "code": code, "family": cat.family,
                                  "hospital": hospital, "n_hospitals": len(present),
                                  "key": (code, hospital)})

        return units

    def gen_d1(self, units, index: int) -> None:
        unit = self.draw("D1", units)
        if not unit:
            return
        sub = self.u.by_code.get(unit["code"])
        if sub is None or sub.empty:
            self.skips["D1:code_has_no_rate_rows"] += 1
            return
        cat = CATEGORIES[unit["category"]]
        lay = pick(cat.lay, *cat.aliases, rng=self.rng)

        # pick a payer/plan that is real for this code somewhere else, so the question
        # is plausible while the row genuinely does not exist
        payer = self.rng.choice(sorted(sub["payer"].unique()))
        plan_rows = sub[sub["payer"] == payer]
        plan = self.rng.choice(sorted(plan_rows["plan"].unique()))
        hospital = unit["hospital"]

        query = pick(
            f"How much is {lay} at {hospital}? My insurance is {payer} {plan}.",
            f"I'm scheduled for {lay} at {hospital} and my plan is {payer} {plan}. What's the negotiated rate?",
            rng=self.rng,
        )
        row_ids, total = self.u.resolve(unit["code"], hospital, payer, plan)
        if row_ids or hospital in self.u.coverage.get(unit["code"], set()):
            self.skips["D1:not_actually_absent"] += 1
            return

        self.emit("D1", unit["code"], query, unit["category"], row_ids, hospital, payer,
                  {"plan": plan, "abstention": "no_row_for_code_at_hospital",
                   "code_hospitals_with_rows": unit["n_hospitals"],
                   "n_rows_total": total})

    def gen_d2(self, units, index: int) -> None:
        unit = self.draw("D2", units)
        if not unit:
            return
        service, hospital, payer, plan, pivot = (
            unit["service"], unit["hospital"], unit["payer"], unit["plan"], unit["code"])
        query = pick(
            f"How much is {service} at {hospital}? My insurance is {payer} {plan}.",
            f"{hospital} is offering {service} — my plan is {payer} {plan}. What's the negotiated price?",
            rng=self.rng,
        )
        self.emit("D2", pivot, query, unit["category"], [], hospital, payer,
                  {"plan": plan, "abstention": "invented_service",
                   "invented_service": service, "pivot_code_source": "sampled_row",
                   "n_rows_total": 0})


# ── D2 eligibility: no MRF description contains all substantive words ───────
def d2_eligible(descriptions: Sequence[str], label: str) -> bool:
    words = [w for w in re.findall(r"[a-z]{5,}", label.lower()) if w not in STOPWORDS]
    if not words:
        return False
    return not any(all(w in d for w in words) for d in descriptions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=".cache/mrf_eval.duckdb")
    ap.add_argument("--out", default="eval_dataset/pt_eval_v2.jsonl")
    ap.add_argument("--meta", default="eval_dataset/pt_eval_v2_meta.jsonl")
    ap.add_argument("--per-subtype", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    universe = Universe(args.db, args.seed)
    builder = Builder(universe)
    units = builder.units_from_pool()

    d2_units: List[Dict[str, Any]] = []
    services = [s for s in INVENTED_SERVICES if d2_eligible(universe.descriptions, s)]
    builder.skips["D2:service_not_eligible"] = len(INVENTED_SERVICES) - len(services)
    rows = universe.df.sample(frac=1.0, random_state=args.seed)

    for service in services:
        for _, row in rows.iterrows():
            d2_units.append({
                "category": row["family"] == "CPT" and "COLONOSCOPY" or "INSULIN",
                "code": row["code"], "service": service, "hospital": row["hospital"],
                "payer": row["payer"], "plan": row["plan"],
                "key": (service, row["hospital"], row["payer"], row["plan"]),
            })
            if len(d2_units) > 800:
                break

    plan = {
        "A1": (builder.gen_a1, units),
        "A2": (builder.gen_a2, units),
        "A3": (builder.gen_a3, units),
        "B1": (builder.gen_b1, units),
        "B2": (builder.gen_b2, units),
        "B3": (builder.gen_b3, units),
        "C1": (builder.gen_c1, builder._shuffled(builder.c1_units())),
        "C2": (builder.gen_c2, builder._shuffled(builder.c2_units())),
        "D1": (builder.gen_d1, builder._shuffled(builder.d1_units())),
        "D2": (builder.gen_d2, d2_units),
    }

    for subtype in SUBTYPES:
        fn, pool = plan[subtype]
        target = args.per_subtype
        made, attempts = 0, 0
        while made < target and attempts < target * 30:
            attempts += 1
            before = len(builder.cases)
            fn(pool, made)
            if len(builder.cases) > before:
                made += 1
        if made < target:
            print(f"WARN {subtype}: only {made}/{target} cases "
                  f"(skips: {dict((k, v) for k, v in builder.skips.items() if k.startswith(subtype))})")

    out, meta = Path(args.out), Path(args.meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for case in builder.cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    with meta.open("w", encoding="utf-8") as f:
        for record in builder.meta:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    per = Counter(c["eval_id"].rsplit("_", 1)[1] for c in builder.cases)
    print(f"wrote {out} — {len(builder.cases)} cases")
    print(f"wrote {meta} — {len(builder.meta)} records")
    print("per subtype:", dict(sorted(per.items())))
    print("skips:", dict(sorted(builder.skips.items())))

    for subtype in SUBTYPES:
        mine = [c for c in builder.cases if c["eval_id"].endswith(f"_{subtype}")]
        if not mine:
            continue
        codes = {c["eval_id"].split("_")[0] for c in mine}
        hospitals = {c["ground_truth_hospital_name"] for c in mine if c["ground_truth_hospital_name"]}
        payers = {c["ground_truth_insurance_provider"] for c in mine if c["ground_truth_insurance_provider"]}
        with_rows = sum(1 for c in mine if c["ground_truth_row_ids"])
        print(f"  {subtype}: {len(mine)} cases · {len(codes)} codes · "
              f"{len(hospitals)} hospitals · {len(payers)} payers · "
              f"{with_rows} with row ids")

    return 0


if __name__ == "__main__":
    sys.exit(main())
