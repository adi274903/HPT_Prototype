/**
 * Patient-facing price assistant.
 *
 * A conversation surface over the same payload the dashboard consumes. Nothing
 * about the pipeline is re-implemented here: the answer text is the model's,
 * the numbers are the matched MRF rows, and citations are produced by matching
 * one against the other (see cite.ts). Anything the model says that no
 * published row corroborates is flagged rather than shown as fact.
 *
 * Layout is chat + evidence: the conversation reads on the left, and the
 * published records behind the focused answer fill a rail on the right.
 */

import { embeddedPayload, health, runQuery } from "../api";
import { DEMO_SCENARIOS } from "../demo";
import { send as sendIcon } from "../icons";
import { applyTheme, currentTheme, toggleTheme } from "../theme";
import type { PipelineResult, Stage } from "../types";
import { buildGrounding, hospitalOffers, type Grounding, type HospitalOffer } from "./cite";
import {
  emptyPane,
  renderAbout,
  renderMessage,
  renderShell,
  renderSourcePane,
  renderThread,
  SUGGESTIONS,
  type ChatMessage,
  type PaneState,
  type PaneTab,
} from "./render";

const PIPELINE_HREF = "./pt_frontend.html";
const WIDE = "(min-width: 1101px)";

const root = document.getElementById("app");
if (!root) throw new Error("#app mount point missing");

interface PatientState {
  theme: "light" | "dark";
  messages: ChatMessage[];
  pane: PaneState;
  busy: boolean;
  live: boolean;
  aboutOpen: boolean;
}

const state: PatientState = {
  theme: currentTheme(),
  messages: [],
  pane: emptyPane(),
  busy: false,
  live: false,
  aboutOpen: false,
};

let nextId = 1;
const newId = () => `m${nextId++}`;

const isWide = (): boolean => {
  try {
    return window.matchMedia(WIDE).matches;
  } catch {
    return true;
  }
};

/* ── Deriving the patient view from a payload ─────────────────────────── */

function viewOf(result: PipelineResult): {
  grounding: Grounding;
  offers: HospitalOffer[];
  rows: PipelineResult["code_plausible"]["price_summary"];
  seconds: number;
  matchCount: number;
} {
  const rows = result.code_plausible?.price_summary ?? [];
  return {
    grounding: buildGrounding(result.answer ?? "", rows),
    offers: hospitalOffers(rows),
    rows,
    seconds: result.total_seconds ?? 0,
    matchCount: result.code_plausible?.match_count ?? rows.length,
  };
}

function answerMessage(result: PipelineResult): ChatMessage {
  const v = viewOf(result);
  return {
    id: newId(),
    role: "bot",
    text: result.answer ?? "",
    grounding: v.grounding,
    offers: v.offers,
    rows: v.rows,
    seconds: v.seconds,
    matchCount: v.matchCount,
    steps: result.stages,
    status: "done",
  };
}

/** Open on whichever tab actually has something in it. */
function bestTab(m: ChatMessage): PaneTab {
  if (m.grounding?.sources.length) return "sources";
  if ((m.offers?.length ?? 0) >= 2) return "compare";
  return "prices";
}

/* ── Rendering ────────────────────────────────────────────────────────── */

function patchPane(): void {
  const slot = document.getElementById("pane-slot");
  if (slot) slot.innerHTML = renderSourcePane(state.messages, state.pane);
}

function patchAbout(): void {
  const slot = document.getElementById("about-slot");
  if (slot) slot.innerHTML = renderAbout(state.aboutOpen, PIPELINE_HREF);
  // Keep the page behind the panel from scrolling away under it.
  document.body.classList.toggle("modal-open", state.aboutOpen);
}

function renderAll(): void {
  // Use the theme already in state. Re-reading it from storage here would
  // discard an explicit choice wherever localStorage is unavailable (private
  // mode, file://, some webviews) and make the toggle appear to revert.
  document.documentElement.dataset.theme = state.theme;
  root!.innerHTML = renderShell(
    state.theme,
    state.messages,
    state.pane,
    state.busy,
    state.aboutOpen,
    PIPELINE_HREF,
  );
  document.body.classList.toggle("modal-open", state.aboutOpen);
}

/** Re-render one message in place; used on every streaming stage update. */
function patchMessage(id: string): void {
  const msg = state.messages.find((m) => m.id === id);
  const el = root!.querySelector(`[data-msg="${id}"]`);
  if (!msg) return;
  if (el) el.outerHTML = renderMessage(msg);
  else {
    const thread = document.getElementById("thread-slot");
    if (thread) thread.innerHTML = renderThread(state.messages);
  }
  scrollToBottom();
}

function scrollToBottom(force = false): void {
  const nearBottom =
    window.innerHeight + window.scrollY > document.body.scrollHeight - 240;
  if (force || nearBottom) {
    try {
      window.scrollTo({ top: document.body.scrollHeight, behavior: force ? "smooth" : "auto" });
    } catch {
      /* scrolling is a nicety; never let it break the answer */
    }
  }
}

function setBusy(busy: boolean): void {
  state.busy = busy;
  const btn = document.querySelector(".send") as HTMLButtonElement | null;
  if (btn) {
    btn.disabled = busy;
    btn.innerHTML = busy ? "" : sendIcon;
  }
}

function focusComposer(): void {
  const ta = document.getElementById("ask") as HTMLTextAreaElement | null;
  if (ta && !ta.disabled && !state.busy) ta.focus({ preventScroll: true });
}

/**
 * Scroll the evidence pane to a record and flash it.
 *
 * Scrolls the pane's own box rather than the page, so following a citation does
 * not throw the reader out of the conversation they were reading.
 */
function revealRecord(n: number): void {
  const pane = document.getElementById("srcpane");
  if (!pane) return;
  const src = pane.querySelector(`.src[data-src="${n}"]`) as HTMLElement | null;
  if (!src) return;

  if (!isWide()) {
    try {
      pane.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch {
      /* not implemented everywhere */
    }
  }

  const body = pane.querySelector(".pane-body") as HTMLElement | null;
  if (body && body.scrollHeight > body.clientHeight) {
    try {
      const delta = src.getBoundingClientRect().top - body.getBoundingClientRect().top;
      body.scrollTo({ top: body.scrollTop + delta - 8, behavior: "smooth" });
    } catch {
      /* not implemented everywhere; the highlight below still applies */
    }
  }

  src.classList.add("flash");
  window.setTimeout(() => src.classList.remove("flash"), 1600);
}

/* ── Asking ───────────────────────────────────────────────────────────── */

function ask(question: string): void {
  const q = question.trim();
  if (!q || state.busy) return;

  state.messages.push({ id: newId(), role: "user", text: q, status: "done" });

  const thinking: ChatMessage = {
    id: newId(),
    role: "bot",
    text: "",
    status: "thinking",
    steps: [],
  };
  state.messages.push(thinking);

  state.busy = true;
  renderAll();
  scrollToBottom(true);

  if (!state.live) {
    // Offline: answer from the bundled sample so the page is still usable.
    window.setTimeout(() => finishWith(pickDemo(q), thinking.id), 900);
    return;
  }

  runQuery(q, {
    onStage(stage: Stage) {
      const steps = thinking.steps ?? [];
      const i = steps.findIndex((s) => s.key === stage.key);
      if (i >= 0) steps[i] = { ...steps[i], ...stage };
      else steps.push(stage);
      thinking.steps = [...steps];
      patchMessage(thinking.id);
    },
    onResult(result: PipelineResult) {
      thinking.status = "done";
      finishWith(result, thinking.id);
    },
    onError(message: string) {
      thinking.status = "error";
      thinking.error = message;
      state.busy = false;
      patchMessage(thinking.id);
      setBusy(false);
    },
  });
}

function finishWith(result: PipelineResult, id: string): void {
  const idx = state.messages.findIndex((m) => m.id === id);
  if (idx < 0) return;
  const done = answerMessage(result);
  done.id = id; // keep the DOM node stable
  state.messages[idx] = done;

  // Point the pane at the answer that just landed.
  state.pane.focusedId = id;
  state.pane.tab = bestTab(done);

  state.busy = false;
  patchMessage(id);
  patchPane();
  setBusy(false);
  scrollToBottom(true);
}

function pickDemo(query: string): PipelineResult {
  const q = query.toLowerCase();
  return (
    DEMO_SCENARIOS.find((s) => s.query.toLowerCase() === q) ??
    DEMO_SCENARIOS.find((s) =>
      s.query
        .toLowerCase()
        .split(/\s+/)
        .filter((w) => w.length > 6)
        .some((w) => q.includes(w)),
    ) ??
    DEMO_SCENARIOS[0]
  );
}

/* ── Composer behaviour ───────────────────────────────────────────────── */

function autoGrow(el: HTMLTextAreaElement): void {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
}

/* ── Events ───────────────────────────────────────────────────────────── */

root.addEventListener("submit", (ev) => {
  const form = ev.target as HTMLElement;
  if (form.id !== "ask-form") return;
  ev.preventDefault();
  const ta = document.getElementById("ask") as HTMLTextAreaElement | null;
  const value = ta?.value ?? "";
  if (ta) {
    ta.value = "";
    autoGrow(ta);
  }
  ask(value);
});

root.addEventListener("keydown", (ev) => {
  const ta = ev.target as HTMLElement;
  if (ta.id !== "ask") return;
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    (document.getElementById("ask-form") as HTMLFormElement | null)?.requestSubmit();
  }
});

root.addEventListener("input", (ev) => {
  const ta = ev.target as HTMLElement;
  if (ta.id === "ask") autoGrow(ta as HTMLTextAreaElement);
});

root.addEventListener("click", (ev) => {
  const target = ev.target as HTMLElement;

  // ── the About panel ───────────────────────────────────────────────────
  if (target.closest("[data-about]")) {
    state.aboutOpen = true;
    patchAbout();
    return;
  }
  if (target.closest("[data-about-close]") || target.id === "about-overlay") {
    state.aboutOpen = false;
    patchAbout();
    return;
  }

  if (target.closest("#theme-toggle")) {
    state.theme = toggleTheme();
    renderAll();
    return;
  }

  const suggestion = target.closest("[data-ask]") as HTMLElement | null;
  if (suggestion) {
    ask(SUGGESTIONS[Number(suggestion.dataset.ask)] ?? "");
    return;
  }

  // ── evidence pane controls ────────────────────────────────────────────
  const tabBtn = target.closest("[data-tab]") as HTMLElement | null;
  if (tabBtn && !tabBtn.hasAttribute("data-focus")) {
    state.pane.tab = tabBtn.dataset.tab as PaneTab;
    patchPane();
    return;
  }

  if (target.closest("[data-pane-toggle]")) {
    state.pane.open = !state.pane.open;
    patchPane();
    return;
  }

  if (target.closest("[data-sort-neg]")) {
    state.pane.sortDir = state.pane.sortDir === "asc" ? "desc" : "asc";
    patchPane();
    return;
  }

  // "N published records behind this answer" in a bubble.
  const jump = target.closest("[data-focus]") as HTMLElement | null;
  if (jump) {
    state.pane.focusedId = jump.dataset.focus ?? null;
    // The button promises records, so show the records — leaving the reader on
    // whichever tab they were browsing would contradict its own label.
    state.pane.tab = "sources";
    state.pane.open = true;
    patchPane();
    try {
      document.getElementById("srcpane")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch {
      /* not implemented everywhere */
    }
    return;
  }

  // A citation marker in the prose -> show its record in the pane.
  const cite = target.closest("[data-cite]") as HTMLElement | null;
  if (cite) {
    const first = Number((cite.dataset.cite ?? "").split(",")[0]);
    const msgId = cite.closest(".msg")?.getAttribute("data-msg") ?? null;
    state.pane.focusedId = msgId;
    state.pane.tab = "sources";
    state.pane.open = true;
    patchPane();
    if (Number.isFinite(first) && first > 0) revealRecord(first);
    return;
  }
});

// Escape closes the About panel. Bound on the document, not #app, because focus
// can sit on the body when the panel opened via a mouse click.
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && state.aboutOpen) {
    state.aboutOpen = false;
    patchAbout();
  }
});

window.addEventListener("resize", () => {
  const wide = isWide();
  if (wide !== state.pane.open) {
    state.pane.open = wide;
    patchPane();
  }
});

/* ── Boot ─────────────────────────────────────────────────────────────── */

async function boot(): Promise<void> {
  applyTheme(currentTheme());
  state.theme = currentTheme();
  state.pane.open = isWide();

  // A pre-rendered snapshot (static embed) opens with the conversation already
  // answered, so the page is never empty.
  const embedded = embeddedPayload();
  if (embedded) {
    state.messages.push({ id: newId(), role: "user", text: embedded.query, status: "done" });
    const answer = answerMessage(embedded);
    state.messages.push(answer);
    state.pane.focusedId = answer.id;
    state.pane.tab = bestTab(answer);
  }

  renderAll();

  const h = await health();
  if (h) {
    state.live = true;
    if (!embedded) renderAll();
  }
  focusComposer();
}

void boot();
