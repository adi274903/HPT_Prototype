#!/usr/bin/env node
/**
 * Browser-level end-to-end test.
 *
 * Boots the real Python server (colab/pt_serve.py) against a notebook-shaped
 * stub, then drives the *built* front end inside jsdom with a fetch bridge to
 * Node and a minimal EventSource shim — so the SSE path the Colab notebook
 * actually uses is exercised, not simulated.
 *
 *   node tools/verify_e2e.mjs
 */

import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 8993;
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

/* ── start the real server ───────────────────────────────────────────── */

function startServer() {
  const child = spawn("python3", [join(ROOT, "tools", "serve_stub.py"), String(PORT)], {
    cwd: ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });

  return new Promise((resolve, reject) => {
    let out = "";
    const timer = setTimeout(() => reject(new Error(`server did not start:\n${out}`)), 20000);
    child.stdout.on("data", (d) => {
      out += d;
      if (out.includes("READY")) {
        clearTimeout(timer);
        resolve(child);
      }
    });
    child.stderr.on("data", (d) => {
      out += d;
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`server exited early (${code}):\n${out}`));
    });
  });
}

/* ── browser shims ───────────────────────────────────────────────────── */

/** Nodes fetch rejects cross-realm signals, so drop it in the bridge. */
const calls = [];
const fetchAbs = async (u, o = {}) => {
  const { signal, ...rest } = o;
  void signal;
  const href = new URL(u, BASE).href;
  try {
    const res = await fetch(href, rest);
    calls.push(`${o.method ?? "GET"} ${href} -> ${res.status}`);
    return res;
  } catch (e) {
    calls.push(`${o.method ?? "GET"} ${href} !! ${e.message}`);
    throw e;
  }
};

function makeEventSource() {
  return class EventSourceShim {
    constructor(url) {
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
              if (line.startsWith("data:")) {
                this.onmessage?.({ data: line.slice(5).trim() });
              }
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

/* ── main ────────────────────────────────────────────────────────────── */

async function main() {
  console.log("\n── browser end-to-end against the real server ──");
  const server = await startServer();
  console.log(`  (server on ${BASE})`);

  try {
    const html = await readFile(join(ROOT, "dist", "pt_frontend.html"), "utf8");
    const errors = [];
    const vc = new VirtualConsole();
    vc.on("jsdomError", (e) => {
      // jsdom cannot navigate; the CSV download click triggers that, so filter it.
      if (!/navigation/i.test(e.message)) errors.push(e.message);
    });

    const exported = { blob: null, name: null };

    const dom = new JSDOM(html, {
      runScripts: "dangerously",
      pretendToBeVisual: true,
      url: BASE,
      virtualConsole: vc,
      beforeParse(window) {
        window.fetch = fetchAbs;
        window.EventSource = makeEventSource();
        window.URL.createObjectURL = (blob) => {
          exported.blob = blob;
          return "blob:stub";
        };
        window.URL.revokeObjectURL = () => {};
        const realCreate = window.document.createElement.bind(window.document);
        window.document.createElement = (tag) => {
          const el = realCreate(tag);
          if (tag === "a") {
            el.click = () => {
              exported.name = el.download;
            };
          }
          return el;
        };
      },
    });

    const { window } = dom;
    const doc = window.document;
    const txt = (sel) => (doc.querySelector(sel)?.textContent ?? "").replace(/\s+/g, " ");

    // ── live detection over real HTTP ───────────────────────────────
    // Wait for the probe to *settle*: the badge renders "Offline sample" on the
    // first paint and only flips once /api/health answers, so polling for either
    // string would race past the flip.
    const waitFor = async (fn, ms = 8000, step = 40) => {
      for (let t = 0; t < ms; t += step) {
        if (fn()) return true;
        await sleep(step);
      }
      return false;
    };

    const wentLive = await waitFor(() => /Live kernel/.test(txt(".badge")));
    check("probed /api/health over real HTTP and found the kernel", wentLive,
      `${txt(".badge").trim()} | calls: ${calls.join(" ; ") || "none"}`);
    check(
      "real MRF/index counts shown in the status strip",
      /MRF rows/.test(txt(".status")) && /CPT idx/.test(txt(".status")),
      txt(".status").replace(/.*elapsed/, "elapsed").slice(0, 90),
    );
    check("pipeline starts empty (no stale sample) in live mode", !/Lowest negotiated rate/.test(txt("#stack")));
    check("no uncaught script errors before run", errors.length === 0, errors.slice(0, 1).join(" | "));

    // ── run a query ─────────────────────────────────────────────────
    const input = doc.getElementById("q");
    input.value = "Best and cheapest hospital for me to get my colonoscopy done with Highmark BCBS?";
    doc.getElementById("query-form").dispatchEvent(
      new window.Event("submit", { bubbles: true, cancelable: true }),
    );

    check("run button disabled while running", doc.getElementById("run").disabled === true);

    // Poll the rail while the run streams.
    const railSnapshots = [];
    let done = false;
    for (let i = 0; i < 500; i++) {
      railSnapshots.push({
        done: doc.querySelectorAll(".rail-item.done").length,
        active: doc.querySelectorAll(".rail-item.active").length,
      });
      if (/Answer generation completed/.test(doc.getElementById("log")?.textContent ?? "") &&
          doc.getElementById("run").disabled === false) {
        done = true;
        break;
      }
      await sleep(40);
    }

    const maxActive = Math.max(...railSnapshots.map((s) => s.active));
    const monotonic = railSnapshots.every(
      (s, i) => i === 0 || s.done >= railSnapshots[i - 1].done,
    );
    check("run completes and re-enables the button", done, `${railSnapshots.length * 40}ms`);
    check("rail showed a step actively running during the stream", maxActive >= 1, `max ${maxActive}`);
    check("rail progress is monotonic", monotonic);
    check("all seven rail steps finish", doc.querySelectorAll(".rail-item.done").length === 7);
    check(
      "per-step timings streamed into the rail",
      /77\.97s/.test(txt(".rail")),
      (txt(".rail").match(/\d+\.\d\ds/g) ?? []).slice(0, 3).join(" "),
    );

    // ── streamed log ────────────────────────────────────────────────
    const log = txt("#log");
    check("kernel output streamed live into the UI", /STEP 4\/7/.test(log), `${log.split("STEP").length - 1} STEP lines`);
    check("log reached pipeline completion", /Answer generation completed/.test(log));

    // ── result rendering ────────────────────────────────────────────
    check("entities rendered from the live payload", /colonoscopy/.test(txt("#step-entities")));
    check("CPT candidates rendered", /45378/.test(txt("#step-cpt")));
    check("HCPCS candidates rendered", /G9937/.test(txt("#step-hcpcs")));
    check("decision rendered", /45378/.test(txt("#step-decision")));
    check(
      "rejected codes streamed from the pipeline log and shown struck through",
      doc.querySelectorAll("#step-validation td.strike").length >= 3,
      `${doc.querySelectorAll("#step-validation td.strike").length} rejected`,
    );
    check("price table populated from the live payload", doc.querySelectorAll("#step-mrf tbody tr").length === 157,
      `${doc.querySelectorAll("#step-mrf tbody tr").length} rows`);
    check("lowest-rate callout rendered", /Lowest negotiated rate/.test(txt("#step-mrf")));
    check("methodology surfaced alongside price", /fee schedule|percent of total/i.test(txt("#step-mrf")));
    check("answer rendered", /45378/.test(txt("#step-answer")));
    check("status strip reports complete", /complete/.test(txt(".status")),
      txt(".status").slice(0, 130));
    check("total elapsed time shown", /\d/.test(txt(".status")));

    // ── CSV export ──────────────────────────────────────────────────
    doc.getElementById("export-csv").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    await sleep(60);
    const csv = exported.blob ? await exported.blob.text() : "";
    const lines = csv.split("\n");
    check("CSV export produced a blob", !!exported.blob);
    check("CSV has a header row", /Hospital,Payer,Plan/.test(lines[0] ?? ""), (lines[0] ?? "").slice(0, 46));
    check("CSV row count matches the table", lines.length === 158, `${lines.length} lines`);
    check("CSV quotes fields containing commas",
      lines.some((l) => l.includes('"')) || !lines.some((l) => l.split(",").length > 11),
      "");
    check("CSV filename derived from the query", /mrf_prices_Best_and_cheapest/.test(exported.name ?? ""), exported.name ?? "");

    // ── theme toggle after a real run ───────────────────────────────
    doc.getElementById("theme-toggle").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    check("theme toggles after a live run without losing the result",
      doc.documentElement.dataset.theme === "dark" &&
      doc.querySelectorAll("#step-mrf tbody tr").length === 157);

    check("no uncaught script errors during the whole run", errors.length === 0, errors.slice(0, 2).join(" | "));

    window.close();
  } finally {
    server.kill("SIGTERM");
  }

  console.log(`\n${pass} passed, ${failures.length} failed`);
  if (failures.length) {
    console.log(`failed: ${failures.join("; ")}`);
    process.exit(1);
  }
  console.log("VERIFIED\n");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
