#!/usr/bin/env node
/**
 * Verification for the patient-facing chat page (dist/pt_patient.html).
 *
 * The interesting claims are about grounding and about the evidence pane, so
 * the tests are too:
 *   - every dollar figure in an answer resolves to a real published row
 *   - a figure that resolves to nothing is flagged, not shown as fact
 *   - citation numbers, the pane's tab counts and the rendered records agree
 *   - a citation click lands on the right record in the pane
 *   - the full price table and the hospital comparison are actually reachable
 *
 *   node tools/verify_patient.mjs
 */

import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PATIENT_HTML = join(ROOT, "dist", "pt_patient.html");
const PORT = 8994;
const BASE = `http://127.0.0.1:${PORT}/`;

let pass = 0;
const failures = [];

function check(name, cond, detail = "") {
  if (cond) {
    pass++;
    console.log(`  ok   ${name}${detail ? `  (${detail})` : ""}`);
  } else {
    failures.push(name);
    console.log(`  FAIL ${name}${detail ? `  (${detail})` : ""}`);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ── render helpers ───────────────────────────────────────────────────── */

let streamsOpened = 0;

async function render({ payload = null, serveApi = false, wide = true } = {}) {
  const html = (await readFile(PATIENT_HTML, "utf8")).replace(
    '<script id="pt-boot"></script>',
    `<script id="pt-boot">window.__PT_BOOT__ = ${JSON.stringify({ apiBase: ".", payload })};</script>`,
  );

  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => {
    // jsdom cannot navigate or scroll; those are environment gaps, not page bugs.
    if (/navigation|Not implemented/i.test(e.message)) return;
    errors.push(e.message);
  });

  const fetchAbs = async (u, o = {}) => {
    const { signal, ...rest } = o;
    void signal;
    return fetch(new URL(u, BASE), rest);
  };

  const dom = new JSDOM(html, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: serveApi ? BASE : `file://${PATIENT_HTML}`,
    virtualConsole: vc,
    beforeParse(window) {
      // jsdom reports a 1024px viewport and a matchMedia that always answers
      // "no". Pin both, so the layout mode under test is the one the page
      // actually sees rather than an accident of the harness.
      Object.defineProperty(window, "innerWidth", { value: wide ? 1440 : 820, configurable: true });
      Object.defineProperty(window, "innerHeight", { value: 900, configurable: true });
      window.matchMedia = (q) => ({
        matches: /min-width:\s*1101px/.test(q) ? wide : !wide,
        media: q,
        onchange: null,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {},
        dispatchEvent: () => false,
      });

      if (serveApi) {
        window.fetch = fetchAbs;
        window.EventSource = class {
          constructor(url) {
            streamsOpened++;
            this.url = String(url);
            this.readyState = 0;
            this.onmessage = null;
            this.onerror = null;
            this._closed = false;
            void this._pump();
          }
          async _pump() {
            try {
              const res = await fetchAbs(this.url, { headers: { accept: "text/event-stream" } });
              if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
              this.readyState = 1;
              const reader = res.body.getReader();
              const dec = new TextDecoder();
              let buf = "";
              for (;;) {
                const { value, done } = await reader.read();
                if (done || this._closed) break;
                buf += dec.decode(value, { stream: true });
                let i;
                while ((i = buf.indexOf("\n\n")) >= 0) {
                  const chunk = buf.slice(0, i);
                  buf = buf.slice(i + 2);
                  for (const line of chunk.split("\n")) {
                    if (line.startsWith("data:")) this.onmessage?.({ data: line.slice(5).trim() });
                  }
                }
              }
              this.readyState = 2;
            } catch (e) {
              this.readyState = 2;
              if (!this._closed) this.onerror?.({ message: String(e) });
            }
          }
          close() {
            this._closed = true;
            this.readyState = 2;
          }
        };
      }
    },
  });

  await sleep(150);
  const { window } = dom;
  return { window, doc: window.document, errors };
}

const txt = (node) => {
  const root = node?.body ?? (node?.cloneNode ? node.cloneNode(true) : node);
  if (root?.querySelectorAll) root.querySelectorAll("script,style").forEach((el) => el.remove());
  return (root?.textContent ?? "").replace(/\s+/g, " ");
};

const pane = (doc) => doc.getElementById("srcpane");
const click = (win, el) => el.dispatchEvent(new win.MouseEvent("click", { bubbles: true }));
const tab = (doc, name) => doc.querySelector(`#srcpane .ptab[data-tab="${name}"]`);

/** Citation numbers used in the prose, in order. */
function citedNumbers(doc) {
  return [...doc.querySelectorAll("sup.cite")].flatMap((s) =>
    (s.dataset.cite ?? "").split(",").filter(Boolean).map(Number),
  );
}

async function main() {
  const scenarios = JSON.parse(await readFile(join(ROOT, "src", "demo.json"), "utf8"));

  /* ═══ A. a grounded answer, desktop layout ═══════════════════════════ */
  console.log("\n── patient page: a fully grounded answer ──");
  const { doc, window, errors } = await render({ payload: scenarios[2] });

  check("no script errors", errors.length === 0, errors.slice(0, 1).join(" | "));
  // Scoped to the topbar: the About panel also carries .logos (so the light/dark
  // swap rules stay in one place), so a document-wide query counts its marks too.
  check("topbar shows one CMU + one NIST mark",
    [...doc.querySelectorAll("header.topbar .logos img")].filter(
      (i) => window.getComputedStyle(i).display !== "none",
    ).length === 2);
  check("links through to the pipeline view",
    /pt_frontend\.html/.test(doc.querySelector('a[href*="pt_frontend"]')?.getAttribute("href") ?? ""));
  check("user question rendered as a bubble", /colonoscopy/.test(txt(doc.querySelector(".msg.user"))));
  check("assistant answer rendered", /45378/.test(txt(doc.querySelector(".msg.bot .prose"))));
  check("no developer jargon in the conversation",
    !/use_codes|cpt_candidates|cosine|STEP \d\/7/.test(txt(doc.querySelector(".thread"))));

  /* — the pane is a rail, not a disclosure under the answer — */
  check("evidence pane exists", !!pane(doc));
  check("records live in the pane, not inside the bubbles",
    doc.querySelectorAll("#srcpane .src").length > 0 &&
      doc.querySelectorAll(".msg .src").length === 0,
    `${doc.querySelectorAll(".msg .src").length} inside bubbles`);
  check("pane is titled and counts its records",
    /Evidence/.test(txt(pane(doc))) && /62 records/.test(txt(pane(doc))),
    (txt(pane(doc)).match(/\d+ records?/) ?? ["?"])[0]);
  check("pane names the question it belongs to",
    /colonoscopy/i.test(txt(doc.querySelector(".pane-q"))));
  check("pane is open on a wide layout", pane(doc).classList.contains("open"));

  /* — figures — */
  const amounts = doc.querySelectorAll(".amount.grounded");
  check("dollar figures render as grounded amounts", amounts.length === 9, `${amounts.length}`);
  check("nothing is flagged unsourced in this answer",
    doc.querySelectorAll(".amount.unsourced").length === 0);
  check("every grounded amount carries a citation marker",
    [...amounts].every((a) => a.dataset.cite));

  /* — tabs — */
  const tabs = [...doc.querySelectorAll("#srcpane .ptab")];
  check("pane offers Sources / Compare / All prices", tabs.length === 3,
    tabs.map((t) => t.dataset.tab).join(","));
  check("tab counts agree with the data",
    /62/.test(txt(tab(doc, "sources"))) &&
      /26/.test(txt(tab(doc, "compare"))) &&
      /157/.test(txt(tab(doc, "prices"))),
    tabs.map((t) => txt(t).replace(/\s+/g, "")).join(" "));
  check("Sources is the active tab", tab(doc, "sources").classList.contains("on"));

  /* — sources tab content — */
  const srcCards = [...doc.querySelectorAll("#srcpane .src")];
  const cited = citedNumbers(doc);
  check("source records rendered", srcCards.length === 62, `${srcCards.length}`);
  check("citation numbers are all in range",
    Math.max(...cited) <= srcCards.length && Math.min(...cited) >= 1,
    `1..${Math.max(...cited)}`);
  check("every record is actually cited",
    new Set(cited).size === srcCards.length,
    `${new Set(cited).size} distinct cited vs ${srcCards.length}`);

  const firstSrc = srcCards[0];
  check("a record names the hospital, payer and plan",
    /Highmark|UPMC|Muncy|Altoona/i.test(txt(firstSrc)) && txt(firstSrc).length > 40);
  check("a record states its charging methodology", /charged as/.test(txt(firstSrc)));
  check("a record exposes the published fields",
    /record id/.test(txt(firstSrc.querySelector("details.raw-row"))));
  check("a record names the price field in plain language",
    /Negotiated rate|Cash price|List price|Typical allowed amount|Low end|High end|Lowest published|Highest published/.test(txt(firstSrc)),
    (txt(firstSrc).match(/Negotiated rate|Cash price|List price|Typical allowed amount|Low end|High end|Lowest published|Highest published/) ?? ["none"])[0]);
  check("the 'not a bill' warning stays in the answer",
    /not your bill/i.test(txt(doc.querySelector(".msg.bot"))));

  /* — a citation click drives the pane — */
  const marker = doc.querySelector("sup.cite");
  const wanted = Number(marker.dataset.cite.split(",")[0]);
  click(window, marker);
  check("clicking a citation shows the Sources tab", tab(doc, "sources").classList.contains("on"));
  check("clicking a citation highlights that record",
    doc.querySelector(`#srcpane .src[data-src="${wanted}"]`)?.classList.contains("flash") === true,
    `record ${wanted}`);

  /* — compare tab — */
  click(window, tab(doc, "compare"));
  check("Compare tab activates", tab(doc, "compare").classList.contains("on"));
  const offers = [...doc.querySelectorAll("#srcpane .offer")];
  check("hospital comparison rendered", offers.length === 26, `${offers.length} hospitals`);
  check("cheapest hospital is first and marked", offers[0]?.classList.contains("cheapest") === true);
  const prices = offers.map((o) =>
    Number((txt(o.querySelector(".price .v")) || "").replace(/[^0-9.]/g, "")));
  check("comparison is sorted cheapest first",
    prices.every((p, i) => i === 0 || p >= prices[i - 1]),
    prices.slice(0, 4).map((p) => `$${p}`).join(" ≤ "));
  check("the compare tab renders no raw price table",
    doc.querySelectorAll("#srcpane .ptable").length === 0);

  /* — all-prices tab — */
  click(window, tab(doc, "prices"));
  const table = doc.querySelector("#srcpane .ptable");
  check("All prices tab renders a table", !!table);
  check("the table carries every matched row",
    table.querySelectorAll("tbody tr").length === 157,
    `${table.querySelectorAll("tbody tr").length} rows`);
  check("the table keeps the methodology column",
    /Method/i.test(txt(table)) && /fee schedule|percent of total/i.test(txt(table)));
  check("the table foot states the count",
    /157 rows shown/.test(txt(doc.querySelector(".pane-foot"))),
    txt(doc.querySelector(".pane-foot")).slice(0, 46));

  const firstNeg = () =>
    doc.querySelector("#srcpane .ptable tbody tr td:nth-child(4)")?.textContent?.trim() ?? "";
  const sortHead = () => doc.querySelector("#srcpane [data-sort-neg]");
  const ascending = firstNeg();
  click(window, sortHead());
  const descending = firstNeg();
  check("clicking Negotiated re-sorts the table", ascending !== descending,
    `${ascending} → ${descending}`);
  check("the sort marker flips", /↓/.test(txt(sortHead())), txt(sortHead()).trim());

  /* — the bubble's jump button focuses the pane — */
  const jump = doc.querySelector(".pane-jump");
  check("the answer links to its records", !!jump && /62 published records/.test(txt(jump)));
  click(window, jump);
  check("the jump button returns to the Sources tab", tab(doc, "sources").classList.contains("on"));

  /* — theme — */
  click(window, doc.getElementById("theme-toggle"));
  check("theme toggles to dark", doc.documentElement.dataset.theme === "dark");
  check("the answer and its records survive a theme toggle",
    doc.querySelectorAll("#srcpane .src").length === 62);

  /* ═══ B. a figure nothing can corroborate ════════════════════════════ */
  console.log("\n── patient page: a figure that cannot be sourced ──");
  const tampered = {
    ...scenarios[0],
    answer: `${scenarios[0].answer}\n\nIt might also cost around $12,345.67.`,
  };
  const b = await render({ payload: tampered });
  const bBot = b.doc.querySelector(".msg.bot");

  check("the fabricated figure is rendered", /\$12,345\.67/.test(txt(bBot.querySelector(".prose"))));
  check("it is marked unsourced, not presented as fact",
    b.doc.querySelectorAll(".amount.unsourced").length === 1);
  check("it carries an explicit 'not in the published file' label",
    /not in the published file/i.test(txt(bBot)));
  check("a warning explains the untraced figure",
    /could not be traced/i.test(txt(bBot.querySelector(".notice.warn"))));
  check("the genuine figures are still grounded",
    b.doc.querySelectorAll(".amount.grounded").length === 3);
  check("no fabricated value appears among the records",
    !/\$12,345\.67/.test(txt(pane(b.doc))));

  /* ═══ C. the methodology warning, on mixed data ══════════════════════ */
  const mixed = {
    ...scenarios[0],
    code_plausible: {
      ...scenarios[0].code_plausible,
      match_count: 2,
      price_summary: [
        { ...scenarios[0].code_plausible.price_summary[0], hospital_name: "Alpha Hospital",
          "standard_charge|negotiated_dollar": 100, "standard_charge|methodology": "fee schedule" },
        { ...scenarios[0].code_plausible.price_summary[0], hospital_name: "Beta Hospital",
          "standard_charge|negotiated_dollar": 250, "standard_charge|methodology": "percent of total billed charges" },
      ],
    },
  };
  const m = await render({ payload: mixed });
  click(m.window, tab(m.doc, "compare"));
  check("a methodology mismatch is warned about when it exists",
    /report charges differently/i.test(txt(pane(m.doc))));
  check("the warning names the methods in plain words",
    /fee schedule|percentage of billed/i.test(txt(m.doc.querySelector(".method-note"))));
  check("both hospitals appear in the comparison",
    m.doc.querySelectorAll("#srcpane .offer").length === 2);
  check("the cheaper hospital ranks first",
    /Alpha Hospital/.test(txt(m.doc.querySelector(".offer.cheapest"))));

  /* ═══ D. narrow layout ═══════════════════════════════════════════════ */
  console.log("\n── patient page: narrow layout ──");
  const n = await render({ payload: scenarios[2], wide: false });
  check("the pane starts collapsed on a narrow layout", !pane(n.doc).classList.contains("open"));
  check("the pane header becomes a toggle",
    !!pane(n.doc).querySelector("[data-pane-toggle]"));
  check("the records are rendered even while collapsed",
    n.doc.querySelectorAll("#srcpane .src").length === 62);
  click(n.window, n.doc.querySelector("[data-pane-toggle]"));
  check("tapping the header opens the pane", pane(n.doc).classList.contains("open"));
  click(n.window, n.doc.querySelector("[data-pane-toggle]"));
  check("tapping again closes it", !pane(n.doc).classList.contains("open"));

  /* ═══ E. the live path ═══════════════════════════════════════════════ */
  console.log("\n── patient page: asking a question against the kernel ──");
  const server = await startServer();
  try {
    const c = await render({ serveApi: true });
    check("composer is present and the thread is empty",
      !!c.doc.getElementById("ask") && c.doc.querySelectorAll(".msg").length === 0);
    check("the pane explains itself before the first question",
      /will appear here/i.test(txt(pane(c.doc))));
    check("suggested questions offered", c.doc.querySelectorAll(".suggestion").length === 3);

    const ta = c.doc.getElementById("ask");
    ta.value = "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?";
    c.doc.getElementById("ask-form").dispatchEvent(
      new c.window.Event("submit", { bubbles: true, cancelable: true }),
    );

    check("the question appears immediately as a user bubble",
      /colonoscopy/.test(txt(c.doc.querySelector(".msg.user"))));
    check("a waiting state is shown while the pipeline runs", !!c.doc.querySelector(".thinking"));

    let answered = false;
    let sawSteps = false;
    for (let i = 0; i < 400; i++) {
      if (c.doc.querySelector(".steps-list")) sawSteps = true;
      if (c.doc.querySelector(".msg.bot .prose")) {
        answered = true;
        break;
      }
      await sleep(40);
    }

    check("progress is described in patient language while running",
      sawSteps || answered,
      [...c.doc.querySelectorAll(".step-line")].map((s) => txt(s)).slice(0, 2).join(" / "));
    check("the answer renders from the live kernel", answered);
    check("live answer carries citations", c.doc.querySelectorAll("sup.cite").length > 0,
      `${c.doc.querySelectorAll("sup.cite").length} markers`);
    check("the pane filled with records", c.doc.querySelectorAll("#srcpane .src").length > 0,
      `${c.doc.querySelectorAll("#srcpane .src").length} records`);
    check("the pane switched to the new answer",
      /records/.test(txt(c.doc.querySelector("#srcpane .pane-count"))),
      txt(c.doc.querySelector("#srcpane .pane-count")));

    click(c.window, tab(c.doc, "compare"));
    check("live answer provides the hospital comparison",
      c.doc.querySelectorAll("#srcpane .offer").length >= 5,
      `${c.doc.querySelectorAll("#srcpane .offer").length} hospitals`);
    click(c.window, tab(c.doc, "prices"));
    check("live answer exposes the full price table",
      c.doc.querySelectorAll("#srcpane .ptable tbody tr").length === 157,
      `${c.doc.querySelectorAll("#srcpane .ptable tbody tr").length} rows`);
    check("step labels never leak pipeline internals",
      !/cpt|candidates|MedGemma|Qdrant|MRF/i.test(
        [...c.doc.querySelectorAll(".step-line")].map((s) => txt(s)).join(" "),
      ));
    check("no script errors during the live run",
      c.errors.length === 0, c.errors.slice(0, 1).join(" | "));
  } finally {
    server.kill("SIGTERM");
  }

  /* ═══ F. no backend at all ═══════════════════════════════════════════ */
  console.log("\n── patient page: no backend at all ──");
  const before = streamsOpened;
  const d = await render();
  check("intro is shown before the first question",
    /What will this procedure cost you/.test(txt(d.doc)));
  check("suggestions shown on the empty state", d.doc.querySelectorAll(".suggestion").length === 3);
  click(d.window, d.doc.querySelectorAll(".suggestion")[1]);
  check("clicking a suggestion asks it", /colonoscopy/i.test(txt(d.doc.querySelector(".msg.user"))));
  check("nothing tries to open a stream with no backend", streamsOpened === before);

  let offlineDone = false;
  for (let i = 0; i < 80; i++) {
    if (d.doc.querySelector("#srcpane .src")) {
      offlineDone = true;
      break;
    }
    await sleep(50);
  }
  check("an answer still arrives with no backend", !!d.doc.querySelector(".msg.bot .prose"));
  check("the offline answer is still grounded in published rows", offlineDone,
    `${d.doc.querySelectorAll("#srcpane .src").length} records`);
  click(d.window, tab(d.doc, "prices"));
  check("the offline answer still offers the full table",
    d.doc.querySelectorAll("#srcpane .ptable tbody tr").length >= 100,
    `${d.doc.querySelectorAll("#srcpane .ptable tbody tr").length} rows`);
  check("no script errors with no backend", d.errors.length === 0, d.errors.slice(0, 1).join(" | "));

  /* ═══ G. the About panel ═════════════════════════════════════════════ */
  console.log("\n── patient page: about ──");
  const a = await render();
  const aboutBtn = a.doc.querySelector("#about-btn");
  check("an About control sits in the top bar", !!aboutBtn);
  check("the panel is closed until asked for",
    !a.doc.querySelector("#about-overlay.open"));

  click(a.window, aboutBtn);
  const overlay = a.doc.querySelector("#about-overlay.open");
  check("clicking About opens the panel", !!overlay);

  if (overlay) {
    const body = txt(overlay);
    check("it says where the prices come from",
      /machine-readable file/i.test(body) && /UPMC/i.test(body));
    check("it says who built it",
      /AIMSEC/i.test(body) && /Carnegie Mellon/i.test(body) && /Standards and Technology/i.test(body));
    check("it draws its limits explicitly",
      /what it cannot do/i.test(body) &&
        /not a bill/i.test(body) &&
        /not medical advice/i.test(body));
    check("it names the single-health-system limit",
      /cannot price care anywhere else/i.test(body));
    check("it repeats the methodology caveat",
      /comparable when the method matches/i.test(body));
    check("it states the privacy position",
      /not stored/i.test(body));
    check("it links on to the pipeline view",
      [...overlay.querySelectorAll("a")].some((el) =>
        /pipeline|how it works/i.test(txt(el)) ||
        /pt_frontend|pipeline/i.test(el.getAttribute("href") || "")));
    check("it carries all four logo variants",
      overlay.querySelectorAll(".about-logos img").length === 4);
    // All four exist so the theme can swap them, but only one per brand may be
    // visible. A separate container class hid neither, so both rendered at once.
    const visLogos = [...overlay.querySelectorAll(".about-logos img")]
      .filter((im) => a.window.getComputedStyle(im).display !== "none");
    check("only the current theme's logo variants are shown",
      visLogos.length === 2,
      `${visLogos.length} visible: ${visLogos.map((im) => im.className).join(", ")}`);
    check("the About logo strip reuses the shared theme-swap container",
      !!overlay.querySelector(".logos.about-logos"));
    check("the logo strip is not shown twice",
      overlay.querySelectorAll(".about-logos").length === 1);
    // The panel explains the flagging treatment; it must not demonstrate it with a
    // decoy. A decorative .amount element would be indistinguishable from a real
    // figure to anything scanning the DOM for unsourced values.
    check("the panel invents no amounts of its own",
      overlay.querySelectorAll(".amount").length === 0,
      `${overlay.querySelectorAll(".amount").length} amount nodes`);
  }

  click(a.window, a.doc.querySelector("[data-about-close]"));
  check("the close button closes it", !a.doc.querySelector("#about-overlay.open"));

  click(a.window, a.doc.querySelector("#about-btn"));
  a.doc.dispatchEvent(new a.window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  check("Escape closes it", !a.doc.querySelector("#about-overlay.open"));

  const threadBefore = a.doc.querySelector("#thread-slot");
  const composerBefore = a.doc.querySelector(".composer-inner");
  click(a.window, a.doc.querySelector("#about-btn"));
  click(a.window, a.doc.querySelector("[data-about-close]"));
  check("About never re-renders the conversation",
    a.doc.querySelector("#thread-slot") === threadBefore);
  check("About never disturbs the composer",
    a.doc.querySelector(".composer-inner") === composerBefore);
  check("no script errors around the About panel",
    a.errors.length === 0, a.errors.slice(0, 1).join(" | "));

  console.log(`\n${pass} passed, ${failures.length} failed`);
  if (failures.length) {
    console.log(`failed: ${failures.join("; ")}`);
    process.exit(1);
  }
  console.log("VERIFIED\n");
}

/* ── server ───────────────────────────────────────────────────────────── */

function startServer() {
  const child = spawn("python3", [join(ROOT, "tools", "serve_stub.py"), String(PORT)], {
    cwd: ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });
  return new Promise((resolve, reject) => {
    let out = "";
    const timer = setTimeout(() => reject(new Error(`server did not start:\n${out}`)), 20000);
    child.stdout.on("data", (dd) => {
      out += dd;
      if (out.includes("READY")) {
        clearTimeout(timer);
        resolve(child);
      }
    });
    child.stderr.on("data", (dd) => {
      out += dd;
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`server exited early (${code}):\n${out}`));
    });
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
