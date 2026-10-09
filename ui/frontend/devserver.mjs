#!/usr/bin/env node
/**
 * Local dev server.
 *
 *   node devserver.mjs           static only  -> UI runs in offline-sample mode
 *   node devserver.mjs --mock    also serves /api/*, replaying a bundled demo
 *                                run with real delays -> exercises the live path
 *
 * The mock endpoints speak exactly the same envelope as the Colab Flask server
 * in colab/pt_serve.py, so verifying against this verifies the UI contract.
 */

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const DIST = join(ROOT, "dist");
const PORT = Number(process.env.PORT ?? 8770);
const MOCK = process.argv.includes("--mock");

let scenarios = [];
if (MOCK) {
  scenarios = JSON.parse(await readFile(join(ROOT, "src", "demo.json"), "utf8"));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function pick(query) {
  const q = (query ?? "").toLowerCase();
  return (
    scenarios.find((s) => s.query.toLowerCase().includes(q.slice(0, 24))) ??
    scenarios.find((s) => q && s.query.toLowerCase().split(" ").some((w) => w.length > 6 && q.includes(w))) ??
    scenarios[0]
  );
}

async function streamRun(query, res) {
  const result = pick(query);
  if (!result) {
    res.writeHead(500).end("no demo scenarios");
    return;
  }

  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });

  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
  const health = { ok: true, mode: "live", mrf_rows: 10_307_535, collections: { cpt_medte: 15927, hcpcs_medte: 9154 }, top_k: 10, models: ["google/medgemma-1.5-4b-it", "MohammadKhodadad/MedTE-cl15-step8000"] };

  send({ type: "log", line: "=".repeat(72) });
  send({ type: "log", line: `[mock] ${query}` });

  const stages = [];
  for (const st of result.stages) {
    stages.push({ ...st, state: "active", seconds: null, detail: null });
    send({ type: "stage", stage: { ...st, state: "active", seconds: null, detail: null } });
    send({ type: "log", line: `STEP ${st.n}/7 — ${st.title}` });
    await sleep(Math.max(180, Math.min(900, (st.seconds ?? 1) * 8)));
    const done = { ...st, state: "done" };
    stages[stages.length - 1] = done;
    send({ type: "stage", stage: done });
    send({ type: "log", line: `${st.title} completed in ${(st.seconds ?? 0).toFixed(2)}s` });
    if (st.key === "mrf") {
      send({ type: "log", line: `MRF rows matched: ${result.code_plausible.match_count}` });
    }
  }

  send({ type: "log", line: "PIPELINE COMPLETE" });
  send({
    type: "result",
    result: { ...result, mode: "live", query, stages, total_seconds: result.total_seconds, created_at: new Date().toISOString() },
  });
  void health;
  res.end();
}

const PAGES = {
  "/": "pt_patient.html",
  "/index.html": "pt_patient.html",
  "/patient": "pt_patient.html",
  "/pt_patient.html": "pt_patient.html",
  "/pipeline": "pt_frontend.html",
  "/dashboard": "pt_frontend.html",
  "/pt_frontend.html": "pt_frontend.html",
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const path = url.pathname.length > 1 ? url.pathname.replace(/\/+$/, "") || "/" : url.pathname;

  if (MOCK && url.pathname === "/api/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true, mode: "live", mrf_rows: 10307535, collections: { cpt_medte: 15927, hcpcs_medte: 9154 }, top_k: 10, pages: ["dashboard", "patient"], models: ["google/medgemma-1.5-4b-it", "MohammadKhodadad/MedTE-cl15-step8000"] }));
    return;
  }

  if (MOCK && url.pathname === "/api/stream") {
    await streamRun(url.searchParams.get("q") ?? "", res);
    return;
  }

  if (MOCK && url.pathname === "/api/query" && req.method === "POST") {
    let body = "";
    for await (const c of req) body += c;
    let q = "";
    try { q = JSON.parse(body || "{}").query ?? ""; } catch { /* ignore */ }
    await sleep(1200);
    const result = pick(q);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ...result, mode: "live", query: q }));
    return;
  }

  const file = PAGES[path];
  if (!file) {
    res.writeHead(404, { "content-type": "text/plain" }).end("not found");
    return;
  }
  try {
    const html = await readFile(join(DIST, file));
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(html);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" }).end(`${file} not built`);
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`serving on http://127.0.0.1:${PORT}/  (patient)  ${MOCK ? "[mock api]" : "[static only]"}`);
  console.log(`          http://127.0.0.1:${PORT}/pipeline  (dashboard)`);
});
