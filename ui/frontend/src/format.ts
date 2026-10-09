/**
 * Formatting helpers. Everything a user reads as a number goes through here so
 * "is that dollars or a raw score" is never ambiguous on screen.
 */

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const NUM = new Intl.NumberFormat("en-US");

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** Currency, or an explicit em-dash for missing/no-charge cells. */
export function money(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? USD.format(n) : "—";
}

export function int(v: unknown): string {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? NUM.format(n) : "—";
}

export function pct(v: unknown, digits = 1): string {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "—";
}

export function score(v: unknown): string {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

export function seconds(v: unknown): string {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return "—";
  if (n < 60) return `${n.toFixed(2)}s`;
  const m = Math.floor(n / 60);
  return `${m}m ${(n - m * 60).toFixed(1)}s`;
}

/** "1 row" / "3 rows" — counts are read by humans, so they agree in number. */
export function plural(n: number, one: string, many = `${one}s`): string {
  return `${int(n)} ${n === 1 ? one : many}`;
}

export function titleCase(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** MedGemma control tokens (`<unused94>`, `<unused95>`) are transport noise. */
export function stripTokens(s: string): string {
  return String(s ?? "").replace(/<unused\d+>/g, "").trim();
}

/** Escape for safe interpolation into innerHTML. */
export function esc(s: unknown): string {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Compact label for a MRF column key: `standard_charge|gross` -> `Gross`. */
export function columnLabel(col: string): string {
  const tail = col.includes("|") ? col.slice(col.lastIndexOf("|") + 1) : col;
  const cleaned = tail.replace(/_/g, " ");
  return titleCase(cleaned);
}

/**
 * MedGemma emits a reasoning block wrapped in control tokens before the real
 * answer. Split it out so the UI can show the answer clean and the reasoning
 * behind a disclosure instead of dumping `<unused94>thought...` at the user.
 */
export function splitAnswer(raw: string): { answer: string; reasoning: string } {
  if (!raw) return { answer: "", reasoning: "" };
  const text = stripTokens(raw);
  const marker = /\bthought\b\s*/i;
  const m = marker.exec(text);

  if (!m) return { answer: text, reasoning: "" };

  // Heuristic: the reasoning block runs until the first blank-line-separated
  // paragraph that looks like prose output rather than a numbered work list.
  const after = text.slice(m.index + m[0].length);
  const paras = after.split(/\n\s*\n/);
  let cut = -1;
  for (let i = 0; i < paras.length; i++) {
    const p = paras[i].trim();
    if (!p) continue;
    const looksLikeWorking = /^(\d+[.)]|[-*•])\s/.test(p) || /^\d+\.\s+\*\*/.test(p);
    if (!looksLikeWorking && p.length > 120) {
      cut = i;
      break;
    }
  }

  if (cut <= 0) return { answer: text, reasoning: "" };
  return {
    reasoning: paras.slice(0, cut).join("\n\n").trim(),
    answer: paras.slice(cut).join("\n\n").trim(),
  };
}

/** Minimal, dependency-free markdown for the answer pane. */
export function mdToHtml(md: string): string {
  const lines = esc(md).split("\n");
  const out: string[] = [];
  let inList = false;

  const inline = (s: string) =>
    s
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(
        /(https?:\/\/[^\s<)]+)/g,
        '<a href="$1" target="_blank" rel="noreferrer">$1</a>',
      );

  for (const line of lines) {
    const t = line.trim();
    if (!t) {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
      continue;
    }
    const bullet = /^[-*•]\s+(.*)$/.exec(t);
    const num = /^(\d+)[.)]\s+(.*)$/.exec(t);
    if (bullet) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    if (num) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li><span class="ord">${num[1]}.</span> ${inline(num[2])}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(t);
    if (h) {
      out.push(`<h4>${inline(h[2])}</h4>`);
      continue;
    }
    out.push(`<p>${inline(t)}</p>`);
  }
  if (inList) out.push("</ul>");
  return out.join("\n");
}
