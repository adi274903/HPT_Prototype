"""Curated CPT/HCPCS categories for the eval generator.

Every code must exist in the vendored catalog (data/codes/*.csv, falling back to
~/Downloads/*_embed_ready.csv). The lists are hand-curated on purpose: a text regex
over the master file pulls in anesthesia, QA/measurement and add-on codes that are
not the procedure a patient asks about (``COLONOSCOP`` matches 51 codes including
anesthesia 00810, AI code 0261U and QA code 0529F).

    python3 eval/categories.py --suggest COLONOSCOP   # candidates from catalog text
    python3 eval/categories.py --verify               # every code in its catalog?
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO / "data" / "codes"


def load_catalog(directory: Path = CATALOG_DIR) -> Dict[str, Dict[str, str]]:
    """``{family: {code: text}}`` from the vendored catalogs, else ~/Downloads."""
    out: Dict[str, Dict[str, str]] = {}

    for family, filename, fallback in (
        ("CPT", "cpt.csv", Path.home() / "Downloads" / "cpt_embed_ready.csv"),
        ("HCPCS", "hcpcs.csv", Path.home() / "Downloads" / "hcpcs_embed_ready.csv"),
    ):
        path = directory / filename
        if not path.is_file():
            path = fallback
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            out[family] = {r["code"].strip(): r["text"].strip() for r in csv.DictReader(f)}

    return out


@dataclass(frozen=True)
class Category:
    name: str
    family: str
    codes: List[str]
    lay: str
    clinical: str
    aliases: List[str] = field(default_factory=list)


CATEGORIES: Dict[str, Category] = {
    "COLONOSCOPY": Category(
        name="COLONOSCOPY",
        family="CPT",
        codes=[
            "45378", "45379", "45380", "45381", "45382", "45384", "45385", "45386",
            "45388", "45389", "45390", "45391", "45392", "45393", "45398",
            "44388", "44389", "44390", "44391", "44392", "44393", "44394", "44397",
            "44401", "44402", "44403", "44404", "44405", "44406", "44407", "44408",
        ],
        lay="a colonoscopy",
        clinical="a flexible colonoscopy, diagnostic",
        aliases=["a colon scope", "getting my colon checked with a camera", "a lower GI scope"],
    ),
    "EGD": Category(
        name="EGD",
        family="CPT",
        codes=[
            "43235", "43236", "43237", "43238", "43239", "43240", "43241", "43242",
            "43243", "43244", "43245", "43246", "43247", "43248", "43249", "43250",
            "43251", "43252", "43253", "43254", "43255", "43257", "43258", "43259",
        ],
        lay="an upper endoscopy",
        clinical="an esophagogastroduodenoscopy",
        aliases=["a scope down my throat", "an upper GI scope", "EGD"],
    ),
    "CHEST_XRAY": Category(
        name="CHEST_XRAY",
        family="CPT",
        codes=["71020", "71021", "71022", "71023", "71030", "71034", "71035",
               "71045", "71046", "71047", "71048"],
        lay="a chest x-ray",
        clinical="a chest radiograph",
        aliases=["a chest film", "an x-ray of my lungs", "chest imaging"],
    ),
    "CMP": Category(
        name="CMP",
        family="CPT",
        codes=["80047", "80048", "80050", "80053", "80069"],
        lay="a metabolic panel blood test",
        clinical="a comprehensive metabolic panel",
        aliases=["routine blood work", "a CMP", "a chem panel"],
    ),
    "MAMMOGRAM": Category(
        name="MAMMOGRAM",
        family="CPT",
        codes=["77051", "77052", "77053", "77054", "77055", "77056", "77057",
               "77061", "77062", "77063", "77065", "77066", "77067"],
        lay="a mammogram",
        clinical="a screening mammogram, bilateral",
        aliases=["a breast x-ray", "my yearly breast screening", "mammo"],
    ),
    "CT_ABDOMEN": Category(
        name="CT_ABDOMEN",
        family="CPT",
        codes=["74150", "74160", "74170", "74175", "74176", "74177", "74178",
               "74181", "74182", "74183"],
        lay="a CT scan of my belly",
        clinical="a computed tomography, abdomen",
        aliases=["an abdominal CAT scan", "a belly scan"],
    ),
    "MRI_BRAIN": Category(
        name="MRI_BRAIN",
        family="CPT",
        codes=["70551", "70552", "70553", "70554", "70555"],
        lay="an MRI of my head",
        clinical="a magnetic resonance imaging, brain",
        aliases=["a brain scan", "head MRI"],
    ),
    "KNEE_ARTHROSCOPY": Category(
        name="KNEE_ARTHROSCOPY",
        family="CPT",
        codes=["29850", "29851", "29855", "29856", "29860", "29861", "29862",
               "29863", "29866", "29867", "29868", "29870", "29871", "29873",
               "29874", "29875", "29876", "29877", "29879", "29880", "29881",
               "29882", "29883", "29884", "29885", "29886", "29887", "29888"],
        lay="knee scope surgery",
        clinical="a knee arthroscopy",
        aliases=["an arthroscopy of my knee", "keyhole knee surgery"],
    ),
    "KNEE_REPLACEMENT": Category(
        name="KNEE_REPLACEMENT",
        family="CPT",
        codes=["27437", "27438", "27440", "27441", "27442", "27443", "27445",
               "27446", "27447", "27448", "27486", "27487", "27488"],
        lay="a knee replacement",
        clinical="a total knee arthroplasty",
        aliases=["new knee surgery", "knee resurfacing"],
    ),
    "HIP_REPLACEMENT": Category(
        name="HIP_REPLACEMENT",
        family="CPT",
        codes=["27125", "27130", "27132", "27134", "27137", "27138"],
        lay="a hip replacement",
        clinical="a total hip arthroplasty",
        aliases=["new hip surgery", "hip resurfacing"],
    ),
    "PHYSICAL_THERAPY": Category(
        name="PHYSICAL_THERAPY",
        family="CPT",
        codes=["97110", "97112", "97116", "97140", "97161", "97162", "97163", "97164"],
        lay="physical therapy",
        clinical="a therapeutic exercise session",
        aliases=["PT", "physio", "rehab sessions"],
    ),
    "INSULIN": Category(
        name="INSULIN",
        family="HCPCS",
        codes=["J1815", "J1817", "J1823", "J1826"],
        lay="insulin pen injections",
        clinical="insulin, per 5 units",
        aliases=["my insulin shots", "the insulin my doctor prescribed"],
    ),
}

HCPCS_CATEGORIES = {k for k, v in CATEGORIES.items() if v.family == "HCPCS"}


def verify(catalog: Dict[str, Dict[str, str]]) -> int:
    """Every curated code must be in its catalog; every category needs >= 2 codes."""
    bad = 0

    for name, cat in CATEGORIES.items():
        owned = catalog[cat.family]
        missing = [c for c in cat.codes if c not in owned]
        if len(cat.codes) < 2:
            print(f"FAIL {name}: only {len(cat.codes)} code(s)")
            bad += 1
        if missing:
            print(f"FAIL {name}: not in {cat.family} catalog: {missing}")
            bad += len(missing)
        else:
            print(f"ok   {name:16} {cat.family} codes={len(cat.codes)}")

    return bad


def suggest(catalog: Dict[str, Dict[str, str]], pattern: str,
            family: Optional[str] = None) -> None:
    rx = re.compile(pattern, re.I)
    for fam, codes in catalog.items():
        if family and fam != family:
            continue
        for code, text in sorted(codes.items()):
            if rx.search(text):
                print(f"{fam:5} {code:6} {text[:100]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suggest", metavar="REGEX")
    ap.add_argument("--family", choices=["CPT", "HCPCS"])
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    catalog = load_catalog()
    if a.suggest:
        suggest(catalog, a.suggest, a.family)
        return 0
    if a.verify:
        return 1 if verify(catalog) else 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
