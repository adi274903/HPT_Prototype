/** Step cards: entities, retrieval candidates, decision, validation. */

import { esc, money, score, seconds, stripTokens } from "../format";
import { STEP_META, type AppState } from "../state";
import type { Candidate, Entities, PipelineResult, Stage } from "../types";

function cardHead(n: number, title: string, metaRight: string): string {
  return `
  <div class="card-head">
    <span class="step-n">${n}</span>
    <h2>${esc(title)}</h2>
    <span class="head-meta">${metaRight}</span>
  </div>`;
}

function blurb(key: string): string {
  const m = STEP_META.find((s) => s.key === key);
  return m ? `<p class="muted" style="margin-bottom:13px">${esc(m.blurb)}</p>` : "";
}

function stageOf(key: string, stages: Stage[]): Stage | undefined {
  return stages.find((s) => s.key === key);
}

function timing(stage: Stage | undefined): string {
  if (!stage) return "";
  if (stage.state === "active") return `<span class="badge accent"><span class="dot"></span>running</span>`;
  return `<span class="t">${stage.seconds == null ? "" : seconds(stage.seconds)}</span>`;
}

/** Placeholder body while a step has not produced data yet. */
function pending(stage: Stage | undefined, what: string): string {
  if (stage?.state === "active") return `<div class="empty">working…</div>`;
  return `<div class="empty">${esc(what)}</div>`;
}

/* ── 1. Entities ──────────────────────────────────────────────────────── */

const ENTITY_FIELDS: ReadonlyArray<{ key: keyof Entities; label: string }> = [
  { key: "medical", label: "Medical" },
  { key: "hospital", label: "Hospital / provider" },
  { key: "insurer", label: "Insurer / payer" },
  { key: "medication", label: "Medication" },
];

function entityValues(values: string[] | undefined, accent: boolean): string {
  const list = (values ?? []).filter((v) => v && String(v).trim());
  if (!list.length) return `<span class="tag none">none found</span>`;
  return list
    .map((v) => `<span class="tag${accent ? " accent" : ""}">${esc(v)}</span>`)
    .join("");
}

export function entitiesCard(state: AppState, stages: Stage[]): string {
  const stage = stageOf("entities", stages);
  const r: PipelineResult | null = state.result;
  const body = r
    ? `<div class="entities">
         ${ENTITY_FIELDS.map(
           (f, i) => `
           <div class="entity">
             <span class="label">${f.label}</span>
             <div class="vals">${entityValues(r.categorized?.[f.key], i === 0)}</div>
           </div>`,
         ).join("")}
       </div>`
    : pending(stage, "Run a query to extract entities.");

  return `<section class="card" id="step-entities">
    ${cardHead(1, "Entity extraction", timing(stage))}
    <div class="card-body">${r ? blurb("entities") : ""}${body}</div>
  </section>`;
}

/* ── 2/3. Retrieval candidates ────────────────────────────────────────── */

function candidateTable(rows: Candidate[] | undefined): string {
  const list = rows ?? [];
  if (!list.length) return `<div class="empty">No candidates returned.</div>`;

  // Bars are scaled against the strongest hit so rank differences stay legible
  // even though every cosine score sits in a narrow high band.
  const top = Math.max(...list.map((c) => c.score), 0.0001);

  return `
  <div class="table-wrap">
    <table class="grid">
      <thead><tr>
        <th style="width:44px">Rank</th>
        <th style="width:120px;text-align:right">Score</th>
        <th style="width:92px">Code</th>
        <th>Descriptor</th>
      </tr></thead>
      <tbody>
        ${list
          .map(
            (c) => `
        <tr>
          <td class="num" style="text-align:left">${c.rank}</td>
          <td>
            <div class="scored">
              <span class="num">${score(c.score)}</span>
              <span class="bar"><i style="width:${Math.max(4, Math.round((c.score / top) * 100))}%"></i></span>
            </div>
          </td>
          <td class="code">${esc(c.code)}</td>
          <td>${esc(c.text)}</td>
        </tr>`,
          )
          .join("")}
      </tbody>
    </table>
  </div>`;
}

function retrievalCard(
  n: number,
  key: "cpt" | "hcpcs",
  title: string,
  rows: Candidate[] | undefined,
  state: AppState,
  stages: Stage[],
  collection: string,
): string {
  const stage = stageOf(key, stages);
  const r = state.result;
  const body = r ? `${blurb(key)}${candidateTable(rows)}` : pending(stage, "Candidates appear after retrieval.");
  const count = rows?.length ? `<span class="faint">${rows.length} candidates</span> &middot; ` : "";
  return `<section class="card" id="step-${key}">
    ${cardHead(n, title, `${count}<span class="faint">${esc(collection)}</span> &middot; ${timing(stage)}`)}
    <div class="card-body${r ? " tight" : ""}">${r ? body : `<div style="padding:15px">${body}</div>`}</div>
  </section>`;
}

export function cptCard(state: AppState, stages: Stage[]): string {
  return retrievalCard(2, "cpt", "CPT retrieval", state.result?.cpt_candidates, state, stages, "cpt_medte");
}

export function hcpcsCard(state: AppState, stages: Stage[]): string {
  return retrievalCard(3, "hcpcs", "HCPCS retrieval", state.result?.hcpcs_candidates, state, stages, "hcpcs_medte");
}

/* ── 4. Decision ──────────────────────────────────────────────────────── */

export function decisionCard(state: AppState, stages: Stage[], showReasoning: boolean): string {
  const stage = stageOf("decision", stages);
  const r = state.result;

  if (!r) {
    return `<section class="card" id="step-decision">
      ${cardHead(4, "Code decision", timing(stage))}
      <div class="card-body">${pending(stage, "The decision model selects from the retrieved candidates.")}</div>
    </section>`;
  }

  const d = r.decision ?? { use_codes: "none", cpt_list: [], hcpcs_list: [] };
  const chipRow = (label: string, codes: string[], cls: string) => `
    <div class="entity">
      <span class="label">${label}</span>
      <div class="vals">${
        codes.length
          ? codes.map((c) => `<span class="tag ${cls}">${esc(c)}</span>`).join("")
          : `<span class="tag none">none</span>`
      }</div>
    </div>`;

  const rawDecision = stripTokens(r.raw?.decision ?? "");

  return `<section class="card" id="step-decision">
    ${cardHead(4, "Code decision", `<span class="faint">use_codes: <code>${esc(d.use_codes)}</code></span> &middot; ${timing(stage)}`)}
    <div class="card-body">
      ${blurb("decision")}
      <div class="entities">
        ${chipRow("Selected CPT", d.cpt_list ?? [], "accent")}
        ${chipRow("Selected HCPCS", d.hcpcs_list ?? [], "accent")}
      </div>
      ${
        rawDecision
          ? `<details class="disclosure" ${showReasoning ? "open" : ""}>
               <summary>Raw decision-model output (with reasoning trace)</summary>
               <pre class="raw">${esc(rawDecision)}</pre>
             </details>`
          : ""
      }
    </div>
  </section>`;
}

/* ── 5. Validation ────────────────────────────────────────────────────── */

export function validationCard(state: AppState, stages: Stage[]): string {
  const stage = stageOf("validation", stages);
  const r = state.result;

  if (!r) {
    return `<section class="card" id="step-validation">
      ${cardHead(5, "Code validation", timing(stage))}
      <div class="card-body">${pending(stage, "Selected codes are intersected with the retrieved set.")}</div>
    </section>`;
  }

  const rejected = [...(r.rejected?.cpt ?? []), ...(r.rejected?.hcpcs ?? [])];
  const kept = [...(r.decision?.cpt_list ?? []), ...(r.decision?.hcpcs_list ?? [])];

  return `<section class="card" id="step-validation">
    ${cardHead(5, "Code validation", `<span class="faint">${kept.length} kept &middot; ${rejected.length} rejected</span> &middot; ${timing(stage)}`)}
    <div class="card-body">
      ${blurb("validation")}
      ${
        rejected.length
          ? `<div class="table-wrap">
               <table class="grid">
                 <thead><tr><th style="width:120px">Code</th><th style="width:110px">Status</th><th>Why</th></tr></thead>
                 <tbody>
                   ${kept
                     .map(
                       (c) => `<tr><td class="code">${esc(c)}</td>
                         <td><span class="badge live">kept</span></td>
                         <td>present in the retrieved candidate set</td></tr>`,
                     )
                     .join("")}
                   ${rejected
                     .map(
                       (c) => `<tr><td class="strike">${esc(c)}</td>
                         <td><span class="badge demo">rejected</span></td>
                         <td>proposed by the model but never retrieved &mdash; dropped before pricing</td></tr>`,
                     )
                     .join("")}
                 </tbody>
               </table>
             </div>`
          : `<div class="empty">All ${kept.length} selected code${kept.length === 1 ? "" : "s"} passed validation
               &mdash; every code the model proposed was present in the retrieved set.</div>`
      }
    </div>
  </section>`;
}

export { money };
