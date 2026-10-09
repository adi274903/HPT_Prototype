/** Step 7: the patient-facing answer, with its reasoning trace split out. */

import { esc, mdToHtml, money, splitAnswer, stripTokens } from "../format";
import type { AppState } from "../state";
import type { PriceRow, Stage } from "../types";

function groundingSummary(state: AppState): string {
  const rows: PriceRow[] = state.result?.code_plausible?.price_summary ?? [];
  if (!rows.length) return "";

  const neg = rows
    .map((r) => ({
      hospital: String(r["hospital_name"] ?? ""),
      payer: String(r["payer_name"] ?? ""),
      plan: String(r["plan_name"] ?? ""),
      v: typeof r["standard_charge|negotiated_dollar"] === "number"
        ? (r["standard_charge|negotiated_dollar"] as number)
        : Number(r["standard_charge|negotiated_dollar"]),
      m: String(r["standard_charge|methodology"] ?? ""),
    }))
    .filter((x) => Number.isFinite(x.v))
    .sort((a, b) => a.v - b.v);

  if (!neg.length) return "";

  const seen = new Set<string>();
  const uniq = neg.filter((x) => {
    const k = `${x.hospital}|${x.payer}|${x.plan}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });

  return `
  <details class="disclosure" open>
    <summary>Rows this answer was grounded on (${uniq.length} distinct payer/plan combinations, cheapest first)</summary>
    <div class="table-wrap" style="margin-top:9px">
      <table class="grid">
        <thead><tr>
          <th>Hospital</th><th>Payer</th><th>Plan</th>
          <th style="text-align:right;width:110px">Negotiated $</th>
          <th style="width:160px">Methodology</th>
        </tr></thead>
        <tbody>
          ${uniq
            .slice(0, 12)
            .map(
              (x) => `<tr>
                <td>${esc(x.hospital)}</td>
                <td>${esc(x.payer)}</td>
                <td>${esc(x.plan)}</td>
                <td class="num">${money(x.v)}</td>
                <td class="muted">${esc(x.m || "—")}</td>
              </tr>`,
            )
            .join("")}
        </tbody>
      </table>
    </div>
    ${uniq.length > 12 ? `<p class="faint" style="margin-top:7px">+ ${uniq.length - 12} more in the price table above.</p>` : ""}
  </details>`;
}

export function answerCard(state: AppState, stages: Stage[]): string {
  const stage = stages.find((s) => s.key === "answer");
  const r = state.result;

  const head = `
  <div class="card-head">
    <span class="step-n">7</span>
    <h2>Final answer</h2>
    <span class="head-meta">
      ${stage?.state === "active" ? `<span class="badge accent"><span class="dot"></span>running</span>` : ""}
      ${stage?.seconds != null ? `<span class="t">${stage.seconds.toFixed(2)}s</span>` : ""}
    </span>
  </div>`;

  if (!r) {
    return `<section class="card" id="step-answer">${head}
      <div class="card-body"><div class="empty">The grounded answer appears here.</div></div>
    </section>`;
  }

  const { answer, reasoning } = splitAnswer(r.answer ?? "");
  const rawCat = stripTokens(r.raw?.categorization ?? "");

  return `<section class="card" id="step-answer">
    ${head}
    <div class="card-body">
      <div class="answer">${mdToHtml(answer || r.answer || "_(empty answer)_")}</div>

      ${groundingSummary(state)}

      ${
        reasoning
          ? `<details class="disclosure" ${state.showReasoning ? "open" : ""}>
               <summary>Model reasoning trace (stripped from the answer above)</summary>
               <pre class="raw">${esc(reasoning)}</pre>
             </details>`
          : ""
      }
      ${
        rawCat
          ? `<details class="disclosure">
               <summary>Raw entity-extraction output</summary>
               <pre class="raw">${esc(rawCat)}</pre>
             </details>`
          : ""
      }
    </div>
  </section>`;
}
