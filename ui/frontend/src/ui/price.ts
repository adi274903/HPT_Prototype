/** Step 6: the MRF price table — the part that actually answers the question. */

import { esc, int, money, plural } from "../format";
import { download } from "../icons";
import type { AppState } from "../state";
import type { PriceRow, Stage } from "../types";

interface Col {
  key: string;
  label: string;
  kind: "text" | "money";
  clip?: boolean;
  code?: boolean;
  width?: string;
}

export const PRICE_COLS: ReadonlyArray<Col> = [
  { key: "hospital_name", label: "Hospital", kind: "text", width: "180px" },
  { key: "payer_name", label: "Payer", kind: "text", width: "160px" },
  { key: "plan_name", label: "Plan", kind: "text", width: "150px" },
  { key: "description", label: "Descriptor", kind: "text", clip: true },
  { key: "__code", label: "Code", kind: "text", code: true, width: "72px" },
  { key: "standard_charge|gross", label: "Gross", kind: "money", width: "92px" },
  { key: "standard_charge|discounted_cash", label: "Cash", kind: "money", width: "92px" },
  { key: "standard_charge|negotiated_dollar", label: "Negotiated $", kind: "money", width: "106px" },
  { key: "standard_charge|min", label: "Min", kind: "money", width: "88px" },
  { key: "standard_charge|max", label: "Max", kind: "money", width: "88px" },
  { key: "standard_charge|methodology", label: "Methodology", kind: "text", width: "150px" },
];

export function codeOf(row: PriceRow): string {
  const cpt = row["CPT"];
  const hcpcs = row["HCPCS"];
  const a = cpt != null && cpt !== "" ? String(cpt) : "";
  const b = hcpcs != null && hcpcs !== "" ? String(hcpcs) : "";
  return [a, b].filter(Boolean).join(" / ") || "—";
}

export function cellValue(row: PriceRow, key: string): string | number | null {
  return key === "__code" ? codeOf(row) : (row[key] ?? null);
}

function numeric(v: string | number | null): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/**
 * Sort with nulls pinned to the bottom in both directions — a missing
 * negotiated rate is "unknown", not "cheapest", so it must never lead the
 * ascending sort that users naturally read as "best deal".
 */
export function sortRows(rows: PriceRow[], col: Col, dir: "asc" | "desc"): PriceRow[] {
  const out = rows.slice();
  const sign = dir === "asc" ? 1 : -1;

  out.sort((x, y) => {
    const a = cellValue(x, col.key);
    const b = cellValue(y, col.key);

    if (col.kind === "money") {
      const an = numeric(a);
      const bn = numeric(b);
      if (an == null && bn == null) return 0;
      if (an == null) return 1;
      if (bn == null) return -1;
      return an === bn ? 0 : (an < bn ? -1 : 1) * sign;
    }

    const as = String(a ?? "").trim().toLowerCase();
    const bs = String(b ?? "").trim().toLowerCase();
    if (!as && !bs) return 0;
    if (!as) return 1;
    if (!bs) return -1;
    return as.localeCompare(bs) * sign;
  });

  return out;
}

function lowestNegotiated(rows: PriceRow[]): { row: PriceRow; value: number } | null {
  let best: { row: PriceRow; value: number } | null = null;
  for (const r of rows) {
    const v = numeric(r["standard_charge|negotiated_dollar"]);
    if (v == null) continue;
    if (!best || v < best.value) best = { row: r, value: v };
  }
  return best;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const s = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function filtersBar(state: AppState): string {
  const f = state.result?.code_plausible?.filters_used;
  if (!f) return "";
  const groups: Array<[string, string[]]> = [
    ["CPT", f.cpt_codes ?? []],
    ["HCPCS", f.hcpcs_codes ?? []],
    ["insurer", f.insurers ?? []],
    ["hospital", f.hospitals ?? []],
  ];
  return `<div class="filters">
    <span class="ferret"><span class="k">filters used</span></span>
    ${groups
      .map(
        ([k, vals]) =>
          `<span class="ferret"><span class="k">${k}</span>${
            vals.length
              ? vals.map((v) => `<span class="v">${esc(v)}</span>`).join(" ")
              : `<span class="v faint">any</span>`
          }</span>`,
      )
      .join("")}
  </div>`;
}

export function csvOf(rows: PriceRow[]): string {
  const cols = PRICE_COLS.map((c) => c.key);
  const header = PRICE_COLS.map((c) => c.label).join(",");
  const q = (v: string | number | null) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const body = rows.map((r) => cols.map((k) => q(cellValue(r, k))).join(","));
  return [header, ...body].join("\n");
}

export function priceCard(state: AppState, stages: Stage[]): string {
  const stage = stages.find((s) => s.key === "mrf");
  const r = state.result;
  const cp = r?.code_plausible;

  const head = `
  <div class="card-head">
    <span class="step-n">6</span>
    <h2>MRF price filter</h2>
    <span class="head-meta">
      ${cp ? `<span class="faint">${plural(cp.match_count, "row")} matched</span> &middot;` : ""}
      ${stage?.state === "active" ? `<span class="badge accent"><span class="dot"></span>running</span>` : ""}
      ${stage?.seconds != null ? `<span class="t">${stage.seconds.toFixed(2)}s</span>` : ""}
    </span>
  </div>`;

  if (!r || !cp) {
    return `<section class="card" id="step-mrf">${head}
      <div class="card-body"><div class="empty">Validated codes are priced against the UPMC MRF here.</div></div>
    </section>`;
  }

  const rows = cp.price_summary ?? [];
  if (!rows.length) {
    return `<section class="card" id="step-mrf">${head}
      <div class="card-body">
        ${filtersBar(state)}
        <div class="empty">No MRF rows matched these codes, payer and hospital.
        That usually means the payer is not contracted at that facility under any published
        plan, or the descriptor carries a different code than the one selected.</div>
      </div>
    </section>`;
  }

  const col = PRICE_COLS.find((c) => c.key === state.sort.col) ?? PRICE_COLS[7];
  const sorted = sortRows(rows, col, state.sort.dir);

  const best = lowestNegotiated(rows);
  const negs = rows
    .map((x) => numeric(x["standard_charge|negotiated_dollar"]))
    .filter((v): v is number => v != null);
  const med = median(negs);

  const callout =
    best && rows.length > 1
      ? `<div class="callout">
           <div>
             <div class="label">Lowest negotiated rate</div>
             <div class="big">${money(best.value)}</div>
           </div>
           <div style="flex:1;min-width:0">
             <div style="font-weight:600">${esc(best.row["hospital_name"] ?? "—")}</div>
             <div class="muted">${esc(best.row["payer_name"] ?? "—")} &middot; ${esc(best.row["plan_name"] ?? "—")}</div>
             <div class="faint" style="margin-top:2px">
               methodology: ${esc(best.row["standard_charge|methodology"] ?? "not published")}
             </div>
           </div>
           ${
             med != null
               ? `<div style="text-align:right">
                    <div class="label">Median</div>
                    <div class="num" style="font-size:14px">${money(med)}</div>
                    <div class="faint">${negs.length} of ${rows.length} priced</div>
                  </div>`
               : ""
           }
         </div>`
      : "";

  const headCells = PRICE_COLS.map((c) => {
    const isSorted = c.key === col.key;
    return `<th class="sortable${isSorted ? " sorted" : ""}" data-sort="${esc(c.key)}"
                style="width:${c.width ?? "auto"};${c.kind === "money" ? "text-align:right" : ""}"
                title="Sort by ${esc(c.label)}">
              ${esc(c.label)}<span class="arrow">${isSorted ? (state.sort.dir === "asc" ? "↑" : "↓") : "↕"}</span>
            </th>`;
  }).join("");

  const bodyRows = sorted
    .map((row) => {
      const isBest = best != null && row === best.row;
      const tds = PRICE_COLS.map((c) => {
        const v = cellValue(row, c.key);
        if (c.kind === "money") {
          return `<td class="num" ${v == null ? 'style="text-align:right;color:var(--faint)"' : ""}>${
            v == null ? "—" : money(v)
          }</td>`;
        }
        if (c.code) return `<td class="code">${esc(v ?? "—")}</td>`;
        const text = v == null ? "" : String(v);
        if (!text) return `<td class="faint">—</td>`;
        if (c.clip) return `<td><span class="cell-clip" title="${esc(text)}">${esc(text)}</span></td>`;
        return `<td>${esc(text)}</td>`;
      }).join("");
      return `<tr${isBest ? ' class="best"' : ""} data-row-json="${esc(JSON.stringify(row))}">${tds}</tr>`;
    })
    .join("");

  return `<section class="card" id="step-mrf">
    ${head}
    <div class="card-body">
      <p class="muted" style="margin-bottom:13px">
        Exact-key lookup over the CMS machine-readable file. Every row carries its charging
        methodology because a negotiated rate is not comparable without it &mdash; per-visit,
        per-unit and case rates for the same code differ by roughly 500% on real MRFs.
      </p>
      ${filtersBar(state)}
      ${callout}
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">
        <span class="label">Price detail</span>
        <span class="faint" style="font-size:11.5px">
          showing ${sorted.length} of ${int(cp.matched_rows_total ?? cp.match_count)} matched
        </span>
        <button class="icon-btn btn-sm" id="export-csv" type="button" style="margin-left:auto"
                title="Download the sorted table as CSV">${download}<span>CSV</span></button>
      </div>
      <div class="table-wrap scroll-box">
        <table class="grid"><thead><tr>${headCells}</tr></thead><tbody>${bodyRows}</tbody></table>
      </div>
    </div>
  </section>`;
}
