/**
 * Grounding: bind the dollar figures in an answer back to the MRF rows they
 * came from.
 *
 * The model writes prose; the price table holds the facts. Rather than trusting
 * the model to cite, we match every `$` figure it produced against the actual
 * values in the matched rows. A figure that matches gets a citation to the rows
 * that produced it. A figure that matches *nothing* is flagged as unverified —
 * which is exactly the signal a patient needs when a model makes a number up.
 *
 * This is deliberately conservative: exact match to the cent, no fuzzy
 * comparison, because a near-miss on a price is worse than no citation at all.
 */

import type { PriceRow } from "../types";

export interface PriceField {
  key: string;
  /** Patient-facing name. */
  label: string;
  /** One-line explanation, in plain language. */
  plain: string;
  /** What the patient actually pays, closest first. */
  patient?: boolean;
}

export const PRICE_FIELDS: ReadonlyArray<PriceField> = [
  {
    key: "standard_charge|negotiated_dollar",
    label: "Negotiated rate",
    plain: "the price your insurer has already agreed with this hospital",
    patient: true,
  },
  {
    key: "standard_charge|discounted_cash",
    label: "Cash price",
    plain: "the discount price if you pay yourself, without insurance",
  },
  {
    key: "standard_charge|gross",
    label: "List price",
    plain: "the hospital's full sticker price before any discount",
  },
  {
    key: "median_amount",
    label: "Typical allowed amount",
    plain: "the midpoint of what this payer has actually allowed recently",
  },
  {
    key: "10th_percentile",
    label: "Low end (10th percentile)",
    plain: "the lower end of recent allowed amounts",
  },
  {
    key: "90th_percentile",
    label: "High end (90th percentile)",
    plain: "the upper end of recent allowed amounts",
  },
  {
    key: "standard_charge|min",
    label: "Lowest published",
    plain: "the lowest charge published for this code",
  },
  {
    key: "standard_charge|max",
    label: "Highest published",
    plain: "the highest charge published for this code",
  },
];

/** How the hospital says it charges — per-visit / per-unit / case rate. */
export function methodologyWords(m: unknown): string {
  const s = String(m ?? "").trim().toLowerCase();
  if (!s) return "not published";
  if (s.includes("fee schedule")) return "a set fee schedule";
  if (s.includes("percent") || s.includes("percentage")) return "a percentage of billed charges";
  if (s.includes("per diem") || s.includes("per day")) return "a daily (per-diem) rate";
  if (s.includes("case rate")) return "a flat case rate";
  if (s.includes("capitation")) return "a capitated rate";
  return String(m);
}

export interface Source {
  /** 1-based number shown as the citation marker. */
  n: number;
  rowId: string;
  hospital: string;
  payer: string;
  plan: string;
  code: string;
  description: string;
  /** Which price field this citation is about. */
  field: PriceField;
  value: number;
  methodology: string;
  /** Raw row, for the "see the published record" disclosure. */
  raw: PriceRow;
}

export interface Figure {
  /** The literal text as it appeared, e.g. "$201.26". */
  text: string;
  value: number;
  /** Citation numbers to render after the figure. */
  citations: number[];
  /** True when no published value matches this figure. */
  unverified: boolean;
}

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "figure"; figure: Figure };

export interface Grounding {
  segments: Segment[];
  sources: Source[];
  /** Values the answer stated that appear nowhere in the published data. */
  unverifiedValues: number[];
}

const FIGURE_RE = /\$\s?([\d][\d,]*(?:\.\d{1,2})?)/g;

function toNumber(s: string): number {
  return Number(s.replace(/,/g, ""));
}

function num(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/** Exact to the cent — a near-miss on a price must not produce a citation. */
function close(a: number, b: number): boolean {
  return Math.abs(a - b) < 0.005;
}

function codeOf(row: PriceRow): string {
  const cpt = row["CPT"] != null && row["CPT"] !== "" ? String(row["CPT"]) : "";
  const hcpcs = row["HCPCS"] != null && row["HCPCS"] !== "" ? String(row["HCPCS"]) : "";
  return [cpt, hcpcs].filter(Boolean).join(" / ") || "—";
}

/** Lower sorts first: the negotiated rate is the most patient-relevant value. */
function fieldRank(field: PriceField): number {
  return field.patient ? 0 : 1 + PRICE_FIELDS.findIndex((f) => f.key === field.key);
}

/**
 * Match every dollar figure in `answer` to the published rows behind it.
 *
 * Returns the answer re-segmented so figures can be rendered with citation
 * markers, plus the ordered source list those markers point at.
 */
export function buildGrounding(answer: string, rows: PriceRow[]): Grounding {
  const sources: Source[] = [];
  const byValue = new Map<number, Source[]>();
  const segments: Segment[] = [];
  const unverifiedValues: number[] = [];

  const text = String(answer ?? "");
  let cursor = 0;
  let match: RegExpExecArray | null;

  FIGURE_RE.lastIndex = 0;
  while ((match = FIGURE_RE.exec(text)) !== null) {
    const rawText = match[0];
    const value = toNumber(match[1]);

    // Text between the previous figure and this one.
    if (match.index > cursor) {
      segments.push({ kind: "text", text: text.slice(cursor, match.index) });
    }
    cursor = match.index + rawText.length;

    let matched = byValue.get(value);
    if (!matched) {
      matched = [];

      // Collect every (row, field) pair that publishes this exact value.
      const candidates: { row: PriceRow; field: PriceField }[] = [];
      for (const row of rows) {
        for (const field of PRICE_FIELDS) {
          const v = num(row[field.key]);
          if (v != null && close(v, value)) candidates.push({ row, field });
        }
      }

      // One source per (hospital, payer, plan, field) — the same rate repeated
      // across plan variants is one fact, not twenty.
      // Prefer the negotiated rate when a figure matches several fields, so a
      // patient sees "your insurer's rate" rather than the sticker price.
      candidates.sort((a, b) => fieldRank(a.field) - fieldRank(b.field));

      const seen = new Set<string>();
      for (const c of candidates) {
        const key = [c.row["hospital_name"], c.row["payer_name"], c.row["plan_name"], c.field.key].join("|");
        if (seen.has(key)) continue;
        seen.add(key);

        const src: Source = {
          n: sources.length + 1,
          rowId: String(c.row["row_id"] ?? ""),
          hospital: String(c.row["hospital_name"] ?? "Unknown hospital"),
          payer: String(c.row["payer_name"] ?? "Unknown payer"),
          plan: String(c.row["plan_name"] ?? "—"),
          code: codeOf(c.row),
          description: String(c.row["description"] ?? ""),
          field: c.field,
          value,
          methodology: methodologyWords(c.row["standard_charge|methodology"]),
          raw: c.row,
        };
        sources.push(src);
        matched.push(src);
        if (sources.length >= 60) break;
      }

      byValue.set(value, matched);
    }

    const unverified = matched.length === 0;
    if (unverified && !unverifiedValues.includes(value)) unverifiedValues.push(value);

    segments.push({
      kind: "figure",
      figure: { text: rawText, value, citations: matched.map((s) => s.n), unverified },
    });
  }

  if (cursor < text.length) segments.push({ kind: "text", text: text.slice(cursor) });

  return { segments, sources, unverifiedValues };
}

/* ── Choosing what to show a patient ──────────────────────────────────── */

export interface HospitalOffer {
  hospital: string;
  payer: string;
  plan: string;
  code: string;
  negotiated: number;
  cash: number | null;
  list: number | null;
  methodology: string;
  rowId: string;
}

/**
 * The cheapest published rate per hospital for the codes that were matched.
 *
 * A patient asking "where should I go" needs one number per hospital, not one
 * per payer/plan combination, so this reduces to the lowest negotiated rate
 * each hospital published for the code and keeps the methodology attached.
 */
export function hospitalOffers(rows: PriceRow[]): HospitalOffer[] {
  const best = new Map<string, HospitalOffer>();

  for (const row of rows) {
    const negotiated = num(row["standard_charge|negotiated_dollar"]);
    if (negotiated == null) continue;

    const hospital = String(row["hospital_name"] ?? "Unknown hospital");
    const current = best.get(hospital);
    if (current && current.negotiated <= negotiated) continue;

    best.set(hospital, {
      hospital,
      payer: String(row["payer_name"] ?? "—"),
      plan: String(row["plan_name"] ?? "—"),
      code: codeOf(row),
      negotiated,
      cash: num(row["standard_charge|discounted_cash"]),
      list: num(row["standard_charge|gross"]),
      methodology: methodologyWords(row["standard_charge|methodology"]),
      rowId: String(row["row_id"] ?? ""),
    });
  }

  return [...best.values()].sort((a, b) => a.negotiated - b.negotiated);
}
