/** Chat surface for the patient-facing page. */

import { esc, mdToHtml, money, plural } from "../format";
import { alert, moon, send, sun } from "../icons";
import { ASSETS } from "../assets.gen";
import type { PriceRow, Stage } from "../types";
import type { Figure, Grounding, HospitalOffer, Source } from "./cite";

/* ── Message model ────────────────────────────────────────────────────── */

export interface ChatMessage {
  id: string;
  role: "user" | "bot";
  /** User question, or the assistant's answer once done. */
  text: string;
  grounding?: Grounding;
  offers?: HospitalOffer[];
  /** Every row the filter matched — backs the "All prices" table. */
  rows?: PriceRow[];
  steps?: Stage[];
  status: "thinking" | "done" | "error";
  error?: string;
  /** Rows the filter returned, for the "no prices found" case. */
  matchCount?: number;
  seconds?: number;
}

/* ── Evidence pane state ──────────────────────────────────────────────── */

export type PaneTab = "sources" | "compare" | "prices";

export interface PaneState {
  tab: PaneTab;
  /** Mobile only — on wide screens the pane is always visible. */
  open: boolean;
  /** Which answer the pane is describing; null = the most recent one. */
  focusedId: string | null;
  sortDir: "asc" | "desc";
}

export function emptyPane(): PaneState {
  return { tab: "sources", open: false, focusedId: null, sortDir: "asc" };
}

/* ── Patient language for pipeline internals ──────────────────────────── */

export const PATIENT_STEPS: ReadonlyArray<{ key: string; label: string }> = [
  { key: "entities", label: "Understanding your question" },
  { key: "cpt", label: "Finding the billing code for that procedure" },
  { key: "hcpcs", label: "Checking the supply and drug code list" },
  { key: "decision", label: "Confirming the most likely code" },
  { key: "validation", label: "Verifying that code is a real one" },
  { key: "mrf", label: "Reading the hospital's published price file" },
  { key: "answer", label: "Writing this up in plain language" },
];

export const SUGGESTIONS: ReadonlyArray<string> = [
  "What might a diagnostic mammogram cost at UPMC Presbyterian with UPMC Health Plan?",
  "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?",
  "What might a diagnostic mammogram cost at UPMC Presbyterian",
];

/* ── Chrome ───────────────────────────────────────────────────────────── */

export function renderTopbar(theme: "light" | "dark", pipelineHref: string): string {
  return `
<header class="topbar"><div class="topbar-inner">
  <div class="logos">
    <img class="logo-cmu logo-on-light" src="${ASSETS.cmuLight}" alt="Carnegie Mellon University">
    <img class="logo-cmu logo-on-dark"  src="${ASSETS.cmuDark}"  alt="Carnegie Mellon University">
    <span class="sep" aria-hidden="true"></span>
    <img class="logo-nist logo-on-light" src="${ASSETS.nistLight}" alt="National Institute of Standards and Technology">
    <img class="logo-nist logo-on-dark"  src="${ASSETS.nistDark}"  alt="National Institute of Standards and Technology">
  </div>
  <div class="brand-sub"><b>Price assistant</b>CMU&ndash;NIST AIMSEC</div>
  <div class="topbar-actions">
    <button class="ghost-btn" id="about-btn" type="button" data-about
            title="What this is, where the prices come from, and what it cannot do">About</button>
    <a class="ghost-btn" href="${esc(pipelineHref)}" title="See the retrieval, validation and pricing steps behind an answer">How it works</a>
    <button class="ghost-btn" id="theme-toggle" type="button"
            aria-label="Switch to ${theme === "dark" ? "light" : "dark"} mode"
            title="Switch to ${theme === "dark" ? "light" : "dark"} mode">
      ${theme === "dark" ? sun : moon}
    </button>
  </div>
</div></header>`;
}

export function renderIntro(): string {
  return `
<div class="intro">
  <h1>What will this procedure cost you?</h1>
  <p>
    Ask about a test, scan or procedure and where you might have it. Answers come from the
    prices hospitals are required to publish &mdash; and every figure is linked back to the
    record it came from, so you can check it yourself.
  </p>
</div>`;
}

export function renderSuggestions(): string {
  return `
<div class="suggestions">
  ${SUGGESTIONS.map(
    (q, i) => `<button class="suggestion" type="button" data-ask="${i}">${esc(q)}</button>`,
  ).join("")}
</div>`;
}

/* ── Figures and citations ────────────────────────────────────────────── */

function figureHtml(f: Figure): string {
  if (f.unverified) {
    return (
      `<span class="amount unsourced" title="This figure does not appear anywhere in the hospital's published price file.">${esc(f.text)}</span>` +
      `<span class="unsourced-flag" title="This number does not match any published price">not in the published file</span>`
    );
  }
  const label =
    f.citations.length <= 3
      ? f.citations.join(",")
      : `${f.citations[0]}–${f.citations[f.citations.length - 1]}`;
  return (
    `<span class="amount grounded" data-cite="${f.citations.join(",")}" title="See the record in the evidence pane">${esc(f.text)}</span>` +
    `<sup class="cite" data-cite="${f.citations.join(",")}" title="Jump to the published record">${label}</sup>`
  );
}

/**
 * Render the answer with citation markers.
 *
 * Figures are swapped for placeholder tokens before markdown runs, then
 * substituted back — so the prose keeps its formatting and the markers survive
 * escaping intact.
 */
export function renderGroundedAnswer(grounding: Grounding): string {
  const figures: string[] = [];
  let text = "";

  for (const seg of grounding.segments) {
    if (seg.kind === "text") {
      text += seg.text;
    } else {
      figures.push(figureHtml(seg.figure));
      text += `@@FIG${figures.length - 1}@@`;
    }
  }

  return mdToHtml(text).replace(/@@FIG(\d+)@@/g, (_m, i: string) => figures[Number(i)] ?? "");
}

/* ── Source records ───────────────────────────────────────────────────── */

function rawField(row: PriceRow, key: string): string {
  const v = row[key];
  if (v == null || v === "") return "—";
  return String(v);
}

function sourceCard(s: Source): string {
  return `
<div class="src" data-src="${s.n}">
  <div class="n">${s.n}</div>
  <div class="src-main">
    <div class="place">${esc(s.hospital)}</div>
    <div class="who-pays">${esc(s.payer)}${s.plan && s.plan !== "—" ? ` &middot; ${esc(s.plan)}` : ""}</div>

    <div class="fig">
      <span class="v">${money(s.value)}</span>
      <span class="what">${esc(s.field.label)}</span>
    </div>
    <div class="explain">${esc(s.field.plain)}.</div>

    <div class="chip-row">
      <span class="chip"><span>code</span> <b>${esc(s.code)}</b></span>
      <span class="chip" title="How this hospital says it charges. A rate is only comparable to another rate when the method matches.">
        <span>charged as</span> <b>${esc(s.methodology)}</b>
      </span>
    </div>
    ${s.description ? `<div class="chip-row"><span class="chip"><span>${esc(s.description.slice(0, 52))}</span></span></div>` : ""}

    <details class="raw-row">
      <summary>See the published record</summary>
      <dl class="raw-grid">
        <dt>record id</dt><dd>${esc(s.rowId || "—")}</dd>
        <dt>hospital</dt><dd>${esc(s.hospital)}</dd>
        <dt>payer</dt><dd>${esc(s.payer)}</dd>
        <dt>plan</dt><dd>${esc(s.plan)}</dd>
        <dt>billing code</dt><dd>${esc(s.code)}</dd>
        <dt>description</dt><dd>${esc(s.description || "—")}</dd>
        <dt>${esc(s.field.label.toLowerCase())}</dt><dd>${money(s.value)}</dd>
        <dt>methodology</dt><dd>${esc(rawField(s.raw, "standard_charge|methodology"))}</dd>
        <dt>list price</dt><dd>${money(s.raw["standard_charge|gross"])}</dd>
        <dt>cash price</dt><dd>${money(s.raw["standard_charge|discounted_cash"])}</dd>
        <dt>billing class</dt><dd>${esc(rawField(s.raw, "billing_class"))}</dd>
        <dt>setting</dt><dd>${esc(rawField(s.raw, "setting"))}</dd>
      </dl>
    </details>
  </div>
</div>`;
}

function sourcesTab(grounding: Grounding | undefined): string {
  const sources = grounding?.sources ?? [];
  if (!sources.length) {
    return `<p class="pane-empty">No figures in this answer were traced to a published
      record. Everything the model stated is either prose or a number that could not be
      matched — see the warning in the answer.</p>`;
  }
  return `<div class="src-list">${sources.map(sourceCard).join("")}</div>`;
}

/* ── Comparison tab ───────────────────────────────────────────────────── */

function compareTab(offers: HospitalOffer[] | undefined): string {
  const list = offers ?? [];
  if (list.length < 2) {
    return `<p class="pane-empty">This answer covers a single hospital, so there is nothing
      to compare. Ask about the procedure without naming a hospital to rank everyone who
      publishes a price for it.</p>`;
  }

  const methods = new Set(list.map((o) => o.methodology));
  return `
${list
  .map(
    (o, i) => `
  <div class="offer${i === 0 ? " cheapest" : ""}">
    <div class="rank">${i + 1}</div>
    <div class="mid">
      <div class="h">${esc(o.hospital)}</div>
      <div class="s">${esc(o.payer)}${o.plan && o.plan !== "—" ? ` &middot; ${esc(o.plan)}` : ""} &middot; charged as ${esc(o.methodology)}</div>
    </div>
    <div class="price">
      <div class="v">${money(o.negotiated)}</div>
      <div class="t">${i === 0 ? "cheapest" : "negotiated"}</div>
    </div>
  </div>`,
  )
  .join("")}
${
  methods.size > 1
    ? `<div class="method-note">
         ${alert}
         <span><b>Careful comparing these.</b> These hospitals report charges differently
         (${[...methods].slice(0, 3).map(esc).join(", ")}). A per-visit and a per-unit rate for the
         same code can differ by several hundred percent, so the cheapest number is not always the
         cheapest care. The method next to each price is part of the price.</span>
       </div>`
    : ""
}`;
}

/* ── All-prices tab ───────────────────────────────────────────────────── */

function numeric(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function priceTable(rows: PriceRow[], matchCount: number, dir: "asc" | "desc"): string {
  if (!rows.length) {
    return `<p class="pane-empty">No rows from the hospital's price file matched this
      question, so there is nothing to list.</p>`;
  }

  // Only rows that actually publish a negotiated rate can be ranked; the rest
  // keep their place at the bottom so a missing rate never reads as "cheapest".
  const sorted = rows.slice().sort((a, b) => {
    const an = numeric(a["standard_charge|negotiated_dollar"]);
    const bn = numeric(b["standard_charge|negotiated_dollar"]);
    if (an == null && bn == null) return 0;
    if (an == null) return 1;
    if (bn == null) return -1;
    return (an - bn) * (dir === "asc" ? 1 : -1);
  });

  return `
<div class="table-scroll">
  <table class="ptable">
    <thead><tr>
      <th>Hospital</th><th>Payer</th><th>Code</th>
      <th class="sortable" data-sort-neg>Negotiated ${dir === "asc" ? "↑" : "↓"}</th>
      <th>Cash</th><th>Method</th>
    </tr></thead>
    <tbody>
      ${sorted
        .map(
          (r) => `<tr>
        <td>${esc(String(r["hospital_name"] ?? "—"))}</td>
        <td class="dim">${esc(String(r["payer_name"] ?? "—"))}</td>
        <td class="mono">${esc(
          [r["CPT"], r["HCPCS"]].filter((v) => v != null && v !== "").join("/") || "—",
        )}</td>
        <td class="num ${numeric(r["standard_charge|negotiated_dollar"]) == null ? "dim" : ""}">${money(
          r["standard_charge|negotiated_dollar"],
        )}</td>
        <td class="num dim">${money(r["standard_charge|discounted_cash"])}</td>
        <td class="dim">${esc(String(r["standard_charge|methodology"] ?? "—"))}</td>
      </tr>`,
        )
        .join("")}
    </tbody>
  </table>
</div>
<p class="pane-foot">${plural(rows.length, "row")} shown${
    matchCount > rows.length ? ` of ${matchCount} matched` : ""
  } &middot; click <b>Negotiated</b> to re-sort.</p>`;
}

/* ── The pane ─────────────────────────────────────────────────────────── */

function focusedPair(
  messages: ChatMessage[],
  pane: PaneState,
): { bot: ChatMessage | null; question: string } {
  const bots = messages.filter((m) => m.role === "bot" && m.status === "done");
  const bot =
    (pane.focusedId && bots.find((m) => m.id === pane.focusedId)) ?? bots[bots.length - 1] ?? null;
  if (!bot) return { bot: null, question: "" };

  const idx = messages.findIndex((m) => m.id === bot.id);
  for (let i = idx - 1; i >= 0; i--) {
    if (messages[i].role === "user") return { bot, question: messages[i].text };
  }
  return { bot, question: "" };
}

export function renderSourcePane(messages: ChatMessage[], pane: PaneState): string {
  const { bot, question } = focusedPair(messages, pane);

  if (!bot) {
    return `
<aside class="srcpane empty" id="srcpane">
  <div class="pane-head">
    <span class="pane-title">Evidence</span>
  </div>
  <div class="pane-body">
    <p class="pane-empty">Ask a question and the published records behind the answer will
      appear here &mdash; the hospital, the payer, the price and how it is charged, straight
      from the hospital's own price file.</p>
  </div>
</aside>`;
  }

  const grounding = bot.grounding;
  const sources = grounding?.sources.length ?? 0;
  const offers = bot.offers?.length ?? 0;
  const rows = bot.rows?.length ?? 0;

  const tab = (key: PaneTab, label: string, count: number) =>
    `<button class="ptab${pane.tab === key ? " on" : ""}" type="button" data-tab="${key}"
             ${count === 0 ? "disabled" : ""}>${label}${count ? ` <i>${count}</i>` : ""}</button>`;

  const body =
    pane.tab === "compare"
      ? compareTab(bot.offers)
      : pane.tab === "prices"
        ? priceTable(bot.rows ?? [], bot.matchCount ?? rows, pane.sortDir)
        : sourcesTab(grounding);

  return `
<aside class="srcpane${pane.open ? " open" : ""}" id="srcpane">
  <button class="pane-head pane-toggle" type="button" data-pane-toggle
          aria-expanded="${pane.open ? "true" : "false"}">
    <span class="pane-title">Evidence</span>
    <span class="pane-count">${sources ? plural(sources, "record") : "no records"}</span>
    <span class="chev" aria-hidden="true">▾</span>
  </button>
  ${question ? `<div class="pane-q" title="${esc(question)}">${esc(question)}</div>` : ""}
  <div class="pane-tabs" role="tablist">
    ${tab("sources", "Sources", sources)}
    ${tab("compare", "Compare", offers)}
    ${tab("prices", "All prices", rows)}
  </div>
  <div class="pane-body">${body}</div>
</aside>`;
}

/* ── Notices ──────────────────────────────────────────────────────────── */

function unsourcedNotice(grounding: Grounding): string {
  if (!grounding.unverifiedValues.length) return "";
  const n = grounding.unverifiedValues.length;
  const list = grounding.unverifiedValues.slice(0, 4).map((v) => money(v)).join(", ");
  return `
<div class="notice warn">
  ${alert}
  <span><b>${plural(n, "figure")} could not be traced.</b>
  ${esc(list)} ${n === 1 ? "does" : "do"} not match any price in the hospital's published file for
  this procedure. Treat ${n === 1 ? "it" : "them"} with caution and check with the hospital before
  relying on ${n === 1 ? "it" : "them"}.</span>
</div>`;
}

const FOOTER_NOTE = `
<div class="notice info">
  ${alert}
  <span>These are <b>published list and negotiated prices, not your bill</b>. What you actually pay
  also depends on your plan's deductible, whether the hospital is in network, and exactly what is
  done on the day. Confirm with the hospital's billing office and your insurer before you schedule.</span>
</div>`;

function noPricesNotice(message: ChatMessage): string {
  if (message.status !== "done") return "";
  if ((message.matchCount ?? 0) > 0) return "";
  return `
<div class="notice warn">
  ${alert}
  <span><b>No published prices matched that combination.</b> The hospital may not be contracted
  with that insurer under any published plan, or the procedure may be recorded under a different
  code. Try naming the procedure and the hospital slightly differently, or ask about the procedure
  on its own to see who publishes a price for it.</span>
</div>`;
}

/* ── Bubbles ──────────────────────────────────────────────────────────── */

function stepsList(steps: Stage[] | undefined): string {
  const byKey = new Map((steps ?? []).map((s) => [s.key, s]));
  return `
  <div class="steps-list">
    ${PATIENT_STEPS.map((p) => {
      const st = byKey.get(p.key);
      const state = st?.state ?? "pending";
      return `<div class="step-line ${state === "done" ? "done" : state === "active" ? "active" : ""}">
          <span class="tick">${state === "done" ? "✓" : ""}</span>
          <span>${esc(p.label)}</span>
        </div>`;
    }).join("")}
  </div>`;
}

function botBody(message: ChatMessage): string {
  if (message.status === "thinking") {
    return `
    <div class="thinking">
      <span class="pulse"><i></i><i></i><i></i></span>
      <span>Reading the published price file&hellip;</span>
    </div>
    ${stepsList(message.steps)}`;
  }

  if (message.status === "error") {
    return `
    <div class="notice warn">
      ${alert}
      <span><b>Something went wrong running that.</b> ${esc(message.error ?? "Unknown error.")}
      Try again, or rephrase the question.</span>
    </div>`;
  }

  const grounding = message.grounding ?? { segments: [], sources: [], unverifiedValues: [] };
  const body = grounding.segments.length ? renderGroundedAnswer(grounding) : mdToHtml(message.text || "");

  // Sources, the comparison and the raw table live in the pane beside the chat,
  // not under the answer — so the citation markers here have somewhere to point.
  const records = grounding.sources.length;
  const jump =
    records > 0
      ? `<button class="pane-jump" type="button" data-focus="${esc(message.id)}"
                 >${plural(records, "published record")} behind this answer
           <span aria-hidden="true">→</span></button>`
      : "";

  return `
  <div class="prose">${body}</div>
  ${noPricesNotice(message)}
  ${unsourcedNotice(grounding)}
  ${jump}
  ${FOOTER_NOTE}`;
}

export function renderMessage(message: ChatMessage): string {
  if (message.role === "user") {
    return `
  <div class="msg user" data-msg="${esc(message.id)}">
    <div class="bubble">${esc(message.text)}</div>
  </div>`;
  }

  const secs =
    message.status === "done" && message.seconds
      ? `<span class="faint num">${message.seconds < 60 ? `${message.seconds.toFixed(0)}s` : `${Math.floor(message.seconds / 60)}m ${(message.seconds % 60).toFixed(0)}s`}</span>`
      : "";

  return `
  <div class="msg bot" data-msg="${esc(message.id)}">
    <div class="who">
      <span class="avatar">$</span>
      <span>Price assistant</span>
      ${secs}
    </div>
    <div class="bubble">${botBody(message)}</div>
  </div>`;
}

export function renderThread(messages: ChatMessage[]): string {
  if (!messages.length) return "";
  return `<div class="thread">${messages.map(renderMessage).join("")}</div>`;
}

/* ── Shell ────────────────────────────────────────────────────────────── */

export function renderComposer(busy: boolean): string {
  return `
<div class="composer-wrap">
  <div class="composer">
    <form class="composer-inner" id="ask-form" autocomplete="off">
      <label class="sr-only" for="ask">Your question</label>
      <textarea id="ask" rows="1" placeholder="Ask about a procedure or a hospital&hellip;"
                ${busy ? 'aria-disabled="true"' : ""}></textarea>
      <button class="send" type="submit" ${busy ? "disabled" : ""} aria-label="Send question"
              title="Send">${busy ? "" : send}</button>
    </form>
    <div class="hint">
      Answers come from prices hospitals are required to publish. Not medical or billing advice.
    </div>
  </div>
</div>`;
}

export function renderFooter(pipelineHref: string): string {
  return `
<footer class="colophon">
  Built on the CMU&ndash;NIST AIMSEC price-transparency pipeline: a language model turns your
  question into billing codes, the codes are checked against a retrieval index, and the validated
  codes are priced against the hospital's CMS machine-readable file.
  <a href="${esc(pipelineHref)}">See the step-by-step pipeline</a>.
  Demo runs on UPMC's published file; nothing you type is stored.
</footer>`;
}

/* ── About ────────────────────────────────────────────────────────────── */

/**
 * "About" as an in-page panel rather than another page: a patient who opens it
 * mid-question should not lose the conversation, and it needs to be readable on
 * a phone.
 */
export function renderAbout(open: boolean, pipelineHref: string): string {
  const logos = `
    <img class="logo-on-light" src="${ASSETS.cmuLight}" alt="Carnegie Mellon University">
    <img class="logo-on-dark" src="${ASSETS.cmuDark}" alt="Carnegie Mellon University">
    <span class="sep" aria-hidden="true"></span>
    <img class="logo-on-light" src="${ASSETS.nistLight}" alt="National Institute of Standards and Technology">
    <img class="logo-on-dark" src="${ASSETS.nistDark}" alt="National Institute of Standards and Technology">`;

  return `
<div class="about-overlay${open ? " open" : ""}" id="about-overlay"
     ${open ? "" : 'aria-hidden="true"'}>
  <div class="about-card" role="dialog" aria-modal="true" aria-label="About this tool">
    <button class="about-close" type="button" data-about-close aria-label="Close">&#10005;</button>

    <div class="logos about-logos">${logos}</div>

    <h2>What this is</h2>
    <p>
      An experimental assistant that answers one question: <em>what will this procedure cost
      me?</em> It reads the prices hospitals are required to publish and shows you the record
      behind every number it gives you.
    </p>

    <h2>Who made it</h2>
    <p>
      It was built at the <strong>CMU&ndash;NIST AI Measurement Science &amp; Engineering
      Cooperative Research Center (AIMSEC)</strong> &mdash; a joint center between Carnegie
      Mellon University and the National Institute of Standards and Technology, focused on
      measuring and evaluating how AI systems actually behave. This tool is a working example
      of one of the center's concerns: whether a model's output can be traced back to
      something real.
    </p>

    <h2>Where the prices come from</h2>
    <p>
      Under a federal price-transparency rule, every US hospital must publish a
      <em>machine-readable file</em> listing its charges and the rates it has negotiated with
      each insurer. This demo reads <strong>UPMC's</strong> published file. The numbers shown
      are those published figures &mdash; not estimates, and not a quote.
    </p>

    <h2>How it answers</h2>
    <ol>
      <li>Your question is broken into medical terms, a hospital and an insurer.</li>
      <li>Those terms are matched against a catalogue of billing codes to find candidates.</li>
      <li>A model picks from those candidates only &mdash; it cannot invent a code.</li>
      <li>The chosen codes are priced against the hospital's published file.</li>
      <li>Every figure in the answer is checked back against the records. If a number does
          not match one, it is labelled
          <span class="about-flag">Not in the published file</span> instead of being passed
          off as fact.</li>
    </ol>
    <p>
      <a href="${esc(pipelineHref)}">See the step-by-step pipeline</a> for the detail behind
      any answer.
    </p>

    <h2>What it cannot do</h2>
    <ul>
      <li><strong>It is not a bill, a quote, or coverage advice.</strong> What you pay also
          depends on your deductible, whether the hospital is in network, and what is actually
          done on the day.</li>
      <li><strong>One health system.</strong> Only UPMC's published file is loaded, so it
          cannot price care anywhere else yet.</li>
      <li><strong>It does not know your plan.</strong> It shows rates by payer and plan name;
          it cannot tell you which of them you are on, or your cost-sharing.</li>
      <li><strong>Rates are only comparable when the method matches.</strong> Hospitals report
          per-visit, per-unit and case rates, which can differ by several hundred percent for
          the same code. That is why every price is shown with how it is charged.</li>
      <li><strong>Not medical advice.</strong> Nothing here tells you what care you need.</li>
    </ul>

    <h2>Your privacy</h2>
    <p>
      This is a research demo. Questions you type are processed in the notebook session and
      are not stored, logged to an account, or used to train anything.
    </p>

    <p class="about-foot">
      <a href="https://www.cmu.edu/aimsec" target="_blank" rel="noreferrer">cmu.edu/aimsec</a>
      &middot; CMU&ndash;NIST AI Measurement Science &amp; Engineering
    </p>
  </div>
</div>`;
}

export function renderShell(
  theme: "light" | "dark",
  messages: ChatMessage[],
  pane: PaneState,
  busy: boolean,
  aboutOpen: boolean,
  pipelineHref: string,
): string {
  return `
${renderTopbar(theme, pipelineHref)}
<div class="wrap">
  <div class="chat-col">
    <div id="intro-slot">${messages.length ? "" : renderIntro() + renderSuggestions()}</div>
    <div id="thread-slot">${renderThread(messages)}</div>
    ${messages.length ? "" : renderFooter(pipelineHref)}
    ${renderComposer(busy)}
  </div>
  <div id="pane-slot">${renderSourcePane(messages, pane)}</div>
</div>
<div id="about-slot">${renderAbout(aboutOpen, pipelineHref)}</div>`;
}
