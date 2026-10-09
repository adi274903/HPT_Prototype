/** Header, hero/query bar, status strip and footer. */

import { esc, int, seconds } from "../format";
import { moon, play, spinner, sun } from "../icons";
import { overallState, type AppState } from "../state";
import { ASSETS } from "../assets.gen";

export const EXAMPLES: ReadonlyArray<string> = [
  "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?",
  "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?",
  "What might a diagnostic mammogram cost at UPMC Presbyterian",
];

const MODEL_LABEL = "google/medgemma-1.5-4b-it";
const EMBED_LABEL = "MohammadKhodadad/MedTE-cl15-step-8000";

export function renderHeader(state: AppState): string {
  const dark = state.theme === "dark";
  return `
<header class="top"><div class="wrap top-inner">
  <div class="brand">
    <div class="logos">
      <img class="logo-cmu logo-on-light" src="${ASSETS.cmuLight}" alt="Carnegie Mellon University">
      <img class="logo-cmu logo-on-dark"  src="${ASSETS.cmuDark}"  alt="Carnegie Mellon University">
      <span class="sep" aria-hidden="true"></span>
      <img class="logo-nist logo-on-light" src="${ASSETS.nistLight}" alt="National Institute of Standards and Technology">
      <img class="logo-nist logo-on-dark"  src="${ASSETS.nistDark}"  alt="National Institute of Standards and Technology">
    </div>
    <div class="title-block">
      <span class="t">AIMSEC</span>
      <span class="s">CMU&ndash;NIST AI Measurement Science &amp; Engineering</span>
    </div>
  </div>
  <div class="top-right">
    <a class="icon-btn" href="./pt_patient.html" title="Patient-facing chat view"
       style="text-decoration:none">Patient view</a>
    <button id="theme-toggle" class="icon-btn" type="button"
            title="Switch to ${dark ? "light" : "dark"} mode"
            aria-label="Switch to ${dark ? "light" : "dark"} mode">
      ${dark ? sun : moon}
    </button>
  </div>
</div></header>`;
}

export function renderHero(state: AppState): string {
  const busy = state.status === "running";
  return `
<section class="hero wrap">
  <div class="eyebrow">Hospital price transparency &middot; UPMC MRF</div>
  <h1>What will this procedure actually cost?</h1>
  <p class="lede">
    A natural-language question is turned into billing codes, the codes are validated against a
    retrieval index, and the validated codes are priced against the hospital&rsquo;s CMS
    machine-readable file &mdash; with the negotiated rate, cash rate and charging methodology kept
    attached to every number.
  </p>

  <form class="querybar" id="query-form" autocomplete="off">
    <label class="sr-only" for="q">Your question</label>
    <input id="q" type="text" name="q" spellcheck="false"
           placeholder="e.g. What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?"
           value="${esc(state.query)}">
    <button class="btn btn-primary" id="run" type="submit" ${busy ? "disabled" : ""}>
      ${busy ? spinner.replace("<svg", '<svg class="spin"') : play}
      <span>${busy ? "Running" : "Run pipeline"}</span>
    </button>
  </form>

  <div class="examples">
    <span class="label">Try</span>
    ${EXAMPLES.map(
      (q, i) =>
        `<button class="chip" type="button" data-example="${i}" title="${esc(q)}">${esc(
          q.length > 62 ? q.slice(0, 62) + "…" : q,
        )}</button>`,
    ).join("")}
  </div>
</section>`;
}

export function renderStatus(state: AppState): string {
  const { done, total, active } = overallState(state.stages);
  const pctDone = state.status === "done" ? 100 : Math.round((done / total) * 100);
  const totalSecs =
    state.result?.total_seconds ??
    state.stages.reduce((a, s) => a + (s.seconds ?? 0), 0);

  const stateLabel =
    state.status === "running"
      ? active
        ? `${active.n}/${total} &middot; ${esc(active.title)}`
        : "starting…"
      : state.status === "done"
        ? "complete"
        : state.status === "error"
          ? "failed"
          : "idle";

  const dotColor =
    state.status === "running"
      ? "var(--accent)"
      : state.status === "done"
        ? "var(--ok)"
        : state.status === "error"
          ? "var(--err)"
          : "var(--faint)";

  const modeBadge = state.live
    ? `<span class="badge live" title="Connected to the Colab kernel — runs the real pipeline"><span class="dot"></span>Live kernel</span>`
    : `<span class="badge demo" title="No kernel endpoint reachable — showing a bundled sample run"><span class="dot"></span>Offline sample</span>`;

  const h = state.health;

  return `
<div class="wrap">
  <div class="status" role="status" aria-live="polite">
    ${modeBadge}
    <span class="grp">
      <span class="dot" style="background:${dotColor}"></span>
      <span class="muted">${stateLabel}</span>
    </span>
    <span class="progress" aria-hidden="true"><i style="width:${pctDone}%"></i></span>
    <span class="grp"><span class="faint">elapsed</span><span class="val">${seconds(totalSecs)}</span></span>
    ${
      h
        ? `<span class="sep"></span>
           <span class="grp"><span class="faint">MRF rows</span><span class="val">${int(h.mrf_rows)}</span></span>
           <span class="grp"><span class="faint">CPT&nbsp;idx</span><span class="val">${int(h.collections?.cpt_medte)}</span></span>
           <span class="grp"><span class="faint">HCPCS&nbsp;idx</span><span class="val">${int(h.collections?.hcpcs_medte)}</span></span>
           <span class="grp"><span class="faint">top-k</span><span class="val">${int(h.top_k)}</span></span>`
        : ""
    }
  </div>
  ${
    state.error
      ? `<div class="callout" style="border-color:color-mix(in srgb, var(--err) 35%, transparent);background:var(--err-soft);margin-top:12px;color:var(--err)">
           <span style="font-weight:600">Run failed</span><span>${esc(state.error)}</span>
         </div>`
      : ""
  }
</div>`;
}

export function renderLog(state: AppState): string {
  if (state.status !== "running" && !state.log.length) return "";
  return `
<div class="wrap" style="margin-top:14px">
  <div class="card">
    <div class="card-head">
      <span class="step-n">&#9679;</span>
      <h2>Kernel output</h2>
      <span class="head-meta"><span class="t">${state.log.length} lines</span></span>
    </div>
    <div class="card-body">
      <pre class="log" id="log">${esc(state.log.join("\n")) || "waiting for the first line…"}</pre>
    </div>
  </div>
</div>`;
}

export function renderFooter(state: AppState): string {
  const r = state.result;
  return `
<footer class="bot"><div class="wrap">
  <div class="foot-grid">
    <span><span class="k">models</span> <span class="v">${MODEL_LABEL}</span> &middot; <span class="v">${EMBED_LABEL}</span></span>
    <span><span class="k">vector store</span> <span class="v">Qdrant</span></span>
    <span><span class="k">rates</span> <span class="v">CMS MRF v2.2/v3.0</span></span>
    ${r ? `<span><span class="k">generated</span> <span class="v">${esc(r.created_at)}</span></span>` : ""}
    <span><span class="k">transport</span> <span class="v">${state.live ? "SSE /api/stream" : "bundled sample"}</span></span>
  </div>
  <p class="disclaimer">
    <strong>Not medical, billing or coverage advice.</strong>
    Codes shown are model-selected candidates validated against a retrieval index, not confirmed
    billing codes. Negotiated rates are payer-specific contract terms published under the CMS
    hospital price-transparency rule; a rate is only comparable when its charging methodology
    matches (per-visit, per-unit and case rates differ by roughly 500% for identical codes).
    Confirm costs with the provider&rsquo;s billing office and your insurer before scheduling care.
  </p>
</div></footer>`;
}
