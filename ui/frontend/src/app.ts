/**
 * Application shell: state, rendering, event wiring.
 *
 * Rendering strategy: build the whole document once, then re-render only the
 * sub-trees that change. Streaming a 3-minute run must not touch the query
 * input, so the hero is never re-rendered mid-run — only the button is patched.
 */

import { health, embeddedPayload, runQuery } from "./api";
import { DEMO_SCENARIOS } from "./demo";
import { esc } from "./format";
import { play, spinner } from "./icons";
import {
  blankStages,
  initialState,
  stagesOf,
  STEP_META,
  type AppState,
} from "./state";
import { applyTheme, currentTheme, toggleTheme, watchSystemTheme } from "./theme";
import type { PipelineResult, Stage } from "./types";
import { answerCard } from "./ui/answer";
import { PRICE_COLS, csvOf, sortRows, priceCard } from "./ui/price";
import { EXAMPLES, renderFooter, renderHeader, renderHero, renderLog, renderStatus } from "./ui/shell";
import { cptCard, decisionCard, entitiesCard, hcpcsCard, validationCard } from "./ui/steps";

const root = document.getElementById("app");
if (!root) throw new Error("#app mount point missing");

const state: AppState = initialState();
state.theme = currentTheme();
state.query = EXAMPLES[0];

let cancelRun: (() => void) | null = null;
let search = "";

/* ── Slots ────────────────────────────────────────────────────────────── */

function slot(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function put(id: string, html: string): void {
  const el = slot(id);
  if (el) el.innerHTML = html;
}

/* ── Rail ─────────────────────────────────────────────────────────────── */

function railHtml(stages: Stage[]): string {
  return `
  <div class="rail-title">Pipeline</div>
  ${STEP_META.map((m) => {
    const st = stages.find((s) => s.key === m.key);
    const cls = st ? st.state : "pending";
    const secs = st?.seconds != null ? `${st.seconds.toFixed(2)}s` : "";
    return `<button class="rail-item ${cls}" type="button" data-goto="${m.key}">
        <span class="n">${m.n}</span>
        <span>${esc(m.title)}</span>
        <span class="secs">${secs}</span>
      </button>`;
  }).join("")}`;
}

/* ── Cards ────────────────────────────────────────────────────────────── */

function stackHtml(): string {
  const stages = stagesOf(state.result, state.stages);
  return [
    entitiesCard(state, stages),
    cptCard(state, stages),
    hcpcsCard(state, stages),
    decisionCard(state, stages, state.showReasoning),
    validationCard(state, stages),
    priceCard(state, stages),
    answerCard(state, stages),
  ].join("");
}

function patchProgress(): void {
  const stages = stagesOf(state.result, state.stages);
  put("status-slot", renderStatus(state));
  put("rail-slot", railHtml(stages));
  put("log-slot", renderLog(state));
  const log = slot("log");
  if (log) log.scrollTop = log.scrollHeight;
}

function patchStack(): void {
  put("stack", stackHtml());
}

function patchHeader(): void {
  put("header-slot", renderHeader(state));
}

function setBusy(busy: boolean): void {
  const btn = document.getElementById("run") as HTMLButtonElement | null;
  if (!btn) return;
  btn.disabled = busy;
  btn.innerHTML = `${busy ? spinner.replace("<svg", '<svg class="spin"') : play}<span>${busy ? "Running" : "Run pipeline"}</span>`;
}

/* ── Full render ──────────────────────────────────────────────────────── */

function renderAll(): void {
  // Use the theme already in state — re-reading storage here would discard an
  // explicit choice where localStorage is unavailable (private mode, file://).
  document.documentElement.dataset.theme = state.theme;
  const stages = stagesOf(state.result, state.stages);

  root!.innerHTML = `
    <div id="header-slot">${renderHeader(state)}</div>
    ${renderHero(state)}
    <div id="status-slot">${renderStatus(state)}</div>
    <div id="log-slot">${renderLog(state)}</div>
    <main class="layout wrap">
      <aside class="rail" id="rail-slot">${railHtml(stages)}</aside>
      <div class="stack" id="stack">${stackHtml()}</div>
    </main>
    ${renderFooter(state)}`;

  const input = document.getElementById("q") as HTMLInputElement | null;
  if (input) input.value = state.query;

  // Deep-link on load, e.g. pt_frontend.html?q=colonoscopy
  if (search) {
    const p = new URLSearchParams(search).get("q");
    if (p) {
      state.query = p;
      if (input) input.value = p;
      search = "";
    }
  }

  setBusy(state.status === "running");
}

/* ── Running a query ──────────────────────────────────────────────────── */

function loadDemo(index: number): void {
  const d = DEMO_SCENARIOS[index] ?? DEMO_SCENARIOS[0];
  state.demoIndex = index;
  state.result = d;
  state.stages = d.stages.map((s) => ({ ...s, state: "done" as const }));
  state.query = d.query;
  state.status = "done";
  state.error = null;
  state.log = [];
  const input = document.getElementById("q") as HTMLInputElement | null;
  if (input) input.value = d.query;
  renderAll();
}

function run(query: string): void {
  const q = query.trim();
  if (!q) return;

  cancelRun?.();
  state.query = q;
  state.status = "running";
  state.error = null;
  state.log = [];
  state.stages = blankStages();
  state.result = null;
  setBusy(true);
  patchStack();
  patchProgress();

  cancelRun = runQuery(q, {
    onStage(stage) {
      const i = state.stages.findIndex((s) => s.key === stage.key);
      if (i >= 0) state.stages[i] = { ...state.stages[i], ...stage };
      else state.stages.push(stage);
      state.status = stage.state === "error" ? "error" : "running";
      patchProgress();
    },
    onLog(line) {
      state.log.push(line);
      if (state.log.length > 500) state.log.splice(0, state.log.length - 500);
      put("log-slot", renderLog(state));
      const log = slot("log");
      if (log) log.scrollTop = log.scrollHeight;
    },
    onResult(result: PipelineResult) {
      state.result = result;
      state.status = "done";
      state.stages = result.stages?.length ? result.stages : state.stages;
      if (result.stages?.length) {
        // Keep the rail in step with the server's own timeline.
        state.stages = result.stages;
      }
      setBusy(false);
      renderAll();
      slot("status-slot")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    },
    onError(message: string) {
      state.status = "error";
      state.error = message;
      setBusy(false);
      patchProgress();
    },
  });
}

/* ── Events ───────────────────────────────────────────────────────────── */

/** Example chip index -> which bundled scenario it demonstrates offline. */
const EXAMPLE_TO_DEMO = [0, 2, 1];

root.addEventListener("submit", (ev) => {
  const form = ev.target as HTMLElement;
  if (form.id !== "query-form") return;
  ev.preventDefault();
  const input = document.getElementById("q") as HTMLInputElement | null;
  const q = input?.value ?? state.query;
  if (state.live) run(q);
  else {
    // Offline: match the typed text to a bundled scenario when we can, so the
    // chip text and the rendered result never contradict each other.
    const idx = EXAMPLES.findIndex((e) => e.trim() === q.trim());
    loadDemo(idx >= 0 ? (EXAMPLE_TO_DEMO[idx] ?? 0) : state.demoIndex);
  }
});

root.addEventListener("input", (ev) => {
  const t = ev.target as HTMLElement;
  if (t.id === "q") state.query = (t as HTMLInputElement).value;
});

root.addEventListener("click", (ev) => {
  const target = ev.target as HTMLElement;

  const themeBtn = target.closest("#theme-toggle");
  if (themeBtn) {
    state.theme = toggleTheme();
    patchHeader();
    return;
  }

  const chip = target.closest("[data-example]") as HTMLElement | null;
  if (chip) {
    const i = Number(chip.dataset.example);
    const q = EXAMPLES[i] ?? "";
    const input = document.getElementById("q") as HTMLInputElement | null;
    if (input) input.value = q;
    state.query = q;
    if (state.live) run(q);
    else loadDemo(EXAMPLE_TO_DEMO[i] ?? 0);
    return;
  }

  const th = target.closest("th[data-sort]") as HTMLElement | null;
  if (th) {
    const col = th.dataset.sort ?? PRICE_COLS[7].key;
    state.sort =
      state.sort.col === col
        ? { col, dir: state.sort.dir === "asc" ? "desc" : "asc" }
        : { col, dir: "asc" };
    patchStack();
    return;
  }

  const csv = target.closest("#export-csv");
  if (csv) {
    const rows = state.result?.code_plausible?.price_summary ?? [];
    const col = PRICE_COLS.find((c) => c.key === state.sort.col) ?? PRICE_COLS[7];
    const text = csvOf(sortRows(rows, col, state.sort.dir));
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `mrf_prices_${state.result?.query.slice(0, 30).replace(/\W+/g, "_") ?? "export"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast(`Exported ${rows.length} rows`);
    return;
  }

  const rail = target.closest("[data-goto]") as HTMLElement | null;
  if (rail) {
    document.getElementById(`step-${rail.dataset.goto}`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
    return;
  }

  // Native <details> toggle — remember the preference across re-renders.
  const sum = target.closest("summary");
  if (sum?.parentElement instanceof HTMLDetailsElement) {
    const det = sum.parentElement;
    state.showReasoning = det.open;
  }
});

function toast(msg: string): void {
  let el = document.querySelector(".toast") as HTMLElement | null;
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add("show");
  window.setTimeout(() => el?.classList.remove("show"), 1900);
}

/* ── Boot ─────────────────────────────────────────────────────────────── */

async function boot(): Promise<void> {
  applyTheme(currentTheme());
  watchSystemTheme();
  search = window.location.search;

  // Something is on screen immediately, even with no backend at all.
  const embedded = embeddedPayload();
  if (embedded) {
    state.result = embedded;
    state.stages = embedded.stages ?? blankStages();
    state.query = embedded.query || state.query;
    state.status = "done";
  } else {
    state.result = DEMO_SCENARIOS[0] ?? null;
    state.stages = state.result?.stages ? state.result.stages.slice() : blankStages();
    state.status = state.result ? "done" : "idle";
  }

  renderAll();

  const h = await health();
  if (h) {
    state.live = true;
    state.health = h;
    // Live kernel: don't show yesterday's sample as if it were this kernel's
    // output. Clear to an empty pipeline and leave the example prefilled.
    if (!embedded) {
      state.result = null;
      state.stages = blankStages();
      state.status = "idle";
      state.query = EXAMPLES[0];
      renderAll();
    }
    patchProgress();
  }
  state.booted = true;
}

void boot();
