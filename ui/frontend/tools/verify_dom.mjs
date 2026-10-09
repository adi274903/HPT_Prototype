#!/usr/bin/env node
/**
 * Headless verification of dist/pt_frontend.html.
 *
 * Renders the real built artifact in jsdom, then asserts on the DOM it
 * produces — layout, both themes, the price table, sorting, CSV export,
 * example chips, and the live-kernel detection path.
 *
 *   node tools/verify_dom.mjs
 */

import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const HTML = join(ROOT, "dist", "pt_frontend.html");

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

/** Render the artifact with optional fetch/EventSource stubs. */
async function render({ serveApi = false, payload = null } = {}) {
  const html = (await readFile(HTML, "utf8")).replace(
    '<script id="pt-boot"></script>',
    `<script id="pt-boot">window.__PT_BOOT__ = ${JSON.stringify({
      apiBase: ".",
      payload,
    })};</script>`,
  );

  const vc = new VirtualConsole();
  const errors = [];
  vc.on("jsdomError", (e) => errors.push(e.message));

  // fetch must exist BEFORE the bundle runs — boot() calls /api/health
  // synchronously on parse, so a stub installed afterwards is too late.
  const stubs = serveApi
    ? (window) => {
        window.fetch = async (u) => {
          if (String(u).includes("/api/health")) {
            return {
              ok: true,
              status: 200,
              json: async () => ({
                ok: true,
                mode: "live",
                mrf_rows: 10307535,
                collections: { cpt_medte: 15927, hcpcs_medte: 9154 },
                top_k: 10,
                models: ["google/medgemma-1.5-4b-it"],
              }),
            };
          }
          return { ok: false, status: 404, text: async () => "nope" };
        };
      }
    : undefined;

  const dom = new JSDOM(html, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: "http://localhost/pt_frontend.html",
    virtualConsole: vc,
    beforeParse: stubs,
  });

  const { window } = dom;
  if (!window.fetch && stubs) stubs(window);

  await new Promise((r) => setTimeout(r, 90));
  return { dom, window, doc: window.document, errors };
}

/** Visible text only — the inline <script> and <style> are part of body.textContent. */
const text = (node) => {
  const root = node?.body ?? (node?.cloneNode ? node.cloneNode(true) : node);
  if (root?.querySelectorAll) {
    root.querySelectorAll("script,style").forEach((el) => el.remove());
  }
  return (root?.textContent ?? "").replace(/\s+/g, " ");
};

async function main() {
  console.log("\n── rendering dist/pt_frontend.html in jsdom ──");
  const { doc, window, errors } = await render();

  check("no script errors", errors.length === 0, errors.slice(0, 2).join(" | "));
  check("app mounted", !!doc.getElementById("app")?.children.length);

  /* — header / branding — */
  const logos = [...doc.querySelectorAll(".logos img")];
  check("4 logo <img> elements (light+dark × CMU,NIST)", logos.length === 4, `${logos.length}`);
  check(
    "every logo is an inlined data URI (no external fetch)",
    logos.every((i) => i.getAttribute("src")?.startsWith("data:")),
  );
  check(
    "CMU + NIST alt text present",
    logos.some((i) => /Carnegie Mellon/.test(i.alt)) && logos.some((i) => /NIST|Standards/.test(i.alt)),
  );
  check("AIMSEC wordmark in header", /AIMSEC/.test(text(doc)));

  /* — dark / light — */
  check("starts in light mode", doc.documentElement.dataset.theme === "light");

  /* — logo variant swapping — */
  const visibleLogos = (w) =>
    [...doc.querySelectorAll(".logos img")].filter(
      (i) => w.getComputedStyle(i).display !== "none",
    );
  const lightLogos = visibleLogos(window);
  check(
    "light mode shows exactly one CMU + one NIST mark (not all four)",
    lightLogos.length === 2,
    `${lightLogos.length} visible: ${lightLogos.map((i) => i.alt.slice(0, 18)).join(" / ")}`,
  );

  /* — query bar — */
  check("query input present", !!doc.getElementById("q"));
  check("run button present", !!doc.getElementById("run"));
  check("example chips present", doc.querySelectorAll("[data-example]").length === 3);

  /* — pipeline rail — */
  const rail = [...doc.querySelectorAll(".rail-item")];
  check("7 rail items", rail.length === 7, `${rail.length}`);
  check(
    "rail shows all seven step titles",
    ["Entity extraction", "CPT retrieval", "HCPCS retrieval", "Code decision", "Code validation", "MRF price filter", "Final answer"]
      .every((t) => rail.some((r) => r.textContent.includes(t))),
  );

  /* — all seven cards — */
  const ids = ["step-entities", "step-cpt", "step-hcpcs", "step-decision", "step-validation", "step-mrf", "step-answer"];
  check("all 7 step cards rendered", ids.every((id) => !!doc.getElementById(id)),
    ids.filter((id) => !doc.getElementById(id)).join(",") || "7/7");

  /* — entities — */
  check("medical entity tagged", [...doc.querySelectorAll(".tag")].some((t) => /diagnostic mammogram/i.test(t.textContent)));
  check("insurer entity tagged", [...doc.querySelectorAll(".tag")].some((t) => /UPMC Health Plan/i.test(t.textContent)));

  /* — candidate tables — */
  const candTables = [doc.getElementById("step-cpt"), doc.getElementById("step-hcpcs")];
  check(
    "CPT + HCPCS tables have 10 candidate rows each",
    candTables.every((s) => s && s.querySelectorAll("tbody tr").length === 10),
    candTables.map((s) => s?.querySelectorAll("tbody tr").length).join("/"),
  );
  check("code 77065 retrieved", /77065/.test(text(doc.getElementById("step-cpt"))));
  check("G0204 retrieved", /G0204/.test(text(doc.getElementById("step-hcpcs"))));

  /* — decision + validation — */
  check("selected CPT codes listed", /77066/.test(text(doc.getElementById("step-decision"))));
  check("use_codes surfaced", /use_codes/.test(text(doc.getElementById("step-decision"))));
  check(
    "validation shows 5 kept / 0 rejected",
    /5 kept/.test(text(doc.getElementById("step-validation"))),
    (text(doc.getElementById("step-validation")).match(/\d+ kept . \d+ rejected/) ?? ["?"])[0],
  );
  check(
    "validation explains that every code passed",
    /passed validation/i.test(text(doc.getElementById("step-validation"))),
  );

  /* — price table — */
  const mrf = doc.getElementById("step-mrf");
  const rows = mrf.querySelectorAll("table tbody tr");
  check("price table rows present", rows.length === 1, `${rows.length}`);
  check(
    "single-row scenario omits the 'lowest rate' callout (no lowest-of-one noise)",
    !/Lowest negotiated rate/i.test(text(mrf)),
  );
  check("negotiated amount shown", /\$228\.75/.test(text(mrf)));
  check("gross charge shown", /\$905\.00/.test(text(mrf)));
  check("cash charge shown", /\$543\.00/.test(text(mrf)));
  check("methodology column populated", /fee schedule/i.test(text(mrf)));
  check("row marked as best", mrf.querySelectorAll("tr.best").length >= 1);
  check("sortable headers", mrf.querySelectorAll("th.sortable").length >= 10);
  check("filters-used strip", /filters used/i.test(text(mrf)));

  /* — answer — */
  const ans = doc.getElementById("step-answer");
  check("final answer rendered", /\$228\.75|\$543\.00|diagnostic mammogram/i.test(text(ans)));
  check("no <unused94> control tokens leaked", !/unused9\d/.test(text(ans)));
  check("grounding table present", /grounded on/i.test(text(ans)));

  /* — footer / disclaimer — */
  check("methodology disclaimer in footer", /per-visit, per-unit and case rates/i.test(text(doc)));

  /* ── interaction: theme toggle ── */
  doc.getElementById("theme-toggle").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  check("theme toggles to dark", doc.documentElement.dataset.theme === "dark");
  const darkLogos = visibleLogos(window);
  check(
    "dark mode shows exactly one CMU + one NIST mark",
    darkLogos.length === 2,
    `${darkLogos.length} visible`,
  );
  check(
    "the visible pair actually swaps between themes",
    darkLogos.length === 2 &&
      lightLogos.every((l) => !darkLogos.includes(l)),
  );

  /* ── interaction: sort the price table ── */
  const negTh = [...mrf.querySelectorAll("th[data-sort]")].find((t) => t.dataset.sort === "standard_charge|negotiated_dollar");
  check("negotiated column is sortable", !!negTh);
  negTh.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  const sortedHead = doc.querySelector("#step-mrf th.sorted");
  check("sorted header highlighted", !!sortedHead);
  check("sort direction marker flips", /↑|↓/.test(sortedHead?.textContent ?? ""));

  /* ── interaction: example chip loads the colonoscopy scenario ── */
  const chips = [...doc.querySelectorAll("[data-example]")];
  chips[1].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  const after = doc.getElementById("step-mrf");
  const afterText = text(after);
  check("chip 2 loads the colonoscopy scenario", /colonoscopy/i.test(text(doc.getElementById("q"))) || /45378/.test(text(doc)), "query/result swapped");
  check("colonoscopy price table is large", after.querySelectorAll("tbody tr").length > 100,
    `${after.querySelectorAll("tbody tr").length} rows`);
  check("colonoscopy codes present", /45378|G9937/.test(afterText));
  check(
    "highmark payer filter applied",
    /highmark/i.test(text(after)),
  );
  check("lowest negotiated callout on multi-row scenario", /Lowest negotiated rate/i.test(afterText));
  check(
    "callout names a cheapest hospital and a dollar figure",
    /\$[\d,]+\.\d\d/.test(after.querySelector(".callout .big")?.textContent ?? "") &&
      (after.querySelector(".callout")?.textContent ?? "").length > 30,
    (after.querySelector(".callout .big")?.textContent ?? "").trim(),
  );
  check("methodology spread visible across payers",
    new Set([...after.querySelectorAll("tbody tr")].map((tr) => tr.lastElementChild?.textContent?.trim())).size > 1);
  check(
    "rejected codes struck through in validation",
    doc.querySelectorAll("#step-validation td.strike").length >= 1,
    `${doc.querySelectorAll("#step-validation td.strike").length} rejected`,
  );
  check("sortable table re-sorts on click and flips direction",
    (() => {
      const firstCell = () => doc.querySelector("#step-mrf tbody tr td:nth-child(8)")?.textContent?.trim() ?? "";
      const arrow = () => {
        const t = [...doc.querySelectorAll("#step-mrf th[data-sort]")]
          .find((x) => x.dataset.sort === "standard_charge|negotiated_dollar");
        return /↓/.test(t.textContent) ? "desc" : /↑/.test(t.textContent) ? "asc" : "none";
      };
      const click = () => [...doc.querySelectorAll("#step-mrf th[data-sort]")]
        .find((x) => x.dataset.sort === "standard_charge|negotiated_dollar")
        .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

      // The <th> is replaced on every re-render, so re-query instead of holding
      // a stale node. Direction-agnostic: the earlier test already flipped it.
      const startArrow = arrow();
      click();
      const a = { cell: firstCell(), arrow: arrow() };
      click();
      const b = { cell: firstCell(), arrow: arrow() };

      return a.arrow !== startArrow && b.arrow === startArrow &&
        a.cell !== b.cell && /\$/.test(a.cell) && /\$/.test(b.cell);
    })(),
    (() => {
      const c = doc.querySelector("#step-mrf tbody tr td:nth-child(8)")?.textContent?.trim();
      return `asc/desc first negotiated cell: ${c}`;
    })());
  check("nulls are pinned last in ascending negotiated sort",
    (() => {
      const cells = [...doc.querySelectorAll("#step-mrf tbody tr td:nth-child(8)")].map((t) => t.textContent.trim());
      const firstDash = cells.findIndex((c) => c === "—");
      return firstDash === -1 || cells.slice(firstDash).every((c) => c === "—");
    })());

  /* ── live-kernel detection ── */
  const live = await render({ serveApi: true });
  check("live mode detected via /api/health", /Live kernel/.test(text(live.doc)), 
    (text(live.doc).match(/Live kernel|Offline sample/) ?? ["?"])[0]);
  check("health meta rendered in status strip", /10,307,535|10307535/.test(text(live.doc)));
  check(
    "live mode clears the sample result (does not pass it off as this kernel's)",
    !/Lowest negotiated rate/i.test(text(live.doc.getElementById("stack") ?? live.doc)),
  );
  check("live mode leaves an example prefilled", (live.doc.getElementById("q")?.value ?? "").length > 20);

  /* ── theme-agnostic selectors that out-rank a [data-theme] rule ── */
  // Regression guard. The dashboard header used to carry both
  //   [data-theme="dark"] header.top { background: rgba(19,19,21,.86) }
  //   :root              header.top { background: rgba(255,255,255,.86) }
  // Equal specificity (0,2,1) — so source order decided, the light value won,
  // and the header stayed white in dark mode. jsdom cannot catch this for us:
  // it does not evaluate color-mix(), so getComputedStyle reports transparent
  // in both themes (confirmed). Hence a check on the source itself.
  for (const sheet of ["src/styles.css", "src/patient/styles.css"]) {
    const css = stripCssComments(await readFile(join(ROOT, sheet), "utf8"));
    const offenders = [...css.matchAll(/(?:^|\})([^{}]*?)\{/g)]
      .map((m) => m[1].trim())
      .filter((sel) => /:root\s+\S/.test(sel));
    check(`no ":root <descendant>" selectors in ${sheet}`,
      offenders.length === 0, offenders.slice(0, 3).join(" | "));
  }

  const dashCss = stripCssComments(await readFile(join(ROOT, "src/styles.css"), "utf8"));
  check("the sticky header takes its background from a theme token",
    /header\.top\s*\{[^}]*background:[^;}]*--surface/s.test(dashCss));
  check("no hard-coded light header background survives",
    !/header\.top[^{]*\{[^}]*rgba\(\s*255\s*,\s*255\s*,\s*255/s.test(dashCss));

  console.log(`\n${pass} passed, ${failures.length} failed`);
  if (failures.length) {
    console.log(`failed: ${failures.join("; ")}`);
    process.exit(1);
  }
  console.log("VERIFIED\n");
}

function stripCssComments(css) {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
