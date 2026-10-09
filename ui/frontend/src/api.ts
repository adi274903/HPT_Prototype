/**
 * Transport.
 *
 * Three ways to get a result, tried in this order:
 *
 *   run()    -> POST {base}/api/run  then GET {base}/api/poll?run=..&since=..
 *   stream() -> GET  {base}/api/stream?q=...   (SSE, live step progress)
 *   query()  -> POST {base}/api/query          (blocking, one shot)
 *
 * Polling comes first because it is made of ordinary request/response pairs. SSE
 * rides one long-lived response whose end is defined by connection close —
 * `_sse_open` sends no `Transfer-Encoding: chunked` — and a proxy that buffers
 * such a response strands every frame until the run finishes, leaving the page
 * blank while the backend is working perfectly. A Cloudflare quick tunnel does
 * exactly that, which is why this ordering exists.
 *
 * `stream()` is kept for servers that predate `/api/run` (the front-end repo's
 * own mock, and the original `colab/pt_serve.py`), and `query()` is the last
 * resort when the page is served from a plain static host.
 *
 * `base` is injected by the Colab server as window.__PT_BOOT__.apiBase and
 * defaults to "." so the same build also works from a plain static server or
 * a file:// open (where it falls back to the bundled demo scenarios).
 */

import type { HealthInfo, PipelineResult, Stage, StreamEvent } from "./types";

declare global {
  interface Window {
    __PT_BOOT__?: {
      apiBase?: string;
      payload?: PipelineResult | null;
      served?: string;
    };
  }
}

export function boot(): NonNullable<Window["__PT_BOOT__"]> {
  return window.__PT_BOOT__ ?? {};
}

export function apiBase(): string {
  const explicit = boot().apiBase;
  if (explicit && explicit !== "auto") return explicit.replace(/\/+$/, "") || ".";

  // Colab serves notebook ports behind a path prefix such as
  // https://<hash>.colab.googleusercontent.com/proxy/8000/
  // Relative fetches would escape that prefix and 404, so resolve it from the
  // iframe's own location instead.
  const m = /^(.*\/proxy\/\d+)(?:\/|$)/.exec(window.location.pathname);
  if (m) return m[1];

  return ".";
}

export function embeddedPayload(): PipelineResult | null {
  return boot().payload ?? null;
}

function url(path: string): string {
  return `${apiBase()}${path}`;
}

export async function health(timeoutMs = 2500): Promise<HealthInfo | null> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url("/api/health"), {
      signal: ctl.signal,
      headers: { accept: "application/json" },
    });
    if (!res.ok) return null;
    const data = (await res.json()) as HealthInfo;
    return data?.ok ? data : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export interface RunHandlers {
  onStage?(stage: Stage): void;
  onLog?(line: string): void;
  /**
   * A partially-built result, delivered as each stage produces its content.
   *
   * Unlike `onResult` this fires many times per run, and the object is
   * incomplete — callers merge it into what they already have rather than
   * replacing it. Optional, so a server that predates the partial frame is still
   * usable.
   */
  onPartial?(patch: Partial<PipelineResult>): void;
  onResult(result: PipelineResult): void;
  onError(message: string): void;
}

/** How often to ask for progress, and the ceiling on a single run. */
const POLL_INTERVAL_MS = 700;
const POLL_TIMEOUT_MS = 30 * 60 * 1000;

/** Statuses that mean "this server predates the polling transport". */
const NOT_IMPLEMENTED = new Set([404, 405, 501]);

class NoPollingTransport extends Error {
  constructor(message = "server has no /api/run") {
    super(message);
    this.name = "NoPollingTransport";
  }
}

function abortable(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/**
 * Run a query, preferring the polling transport and falling back to the stream.
 *
 * Returns a cancel function, as before.
 */
export function runQuery(
  query: string,
  handlers: RunHandlers,
  signal?: AbortSignal,
): () => void {
  let closed = false;
  let finished = false;

  // Once a result (or an error) is delivered the run is over. The poll loop and
  // the stream fallback share these, so neither can report twice.
  const finish = (result: PipelineResult) => {
    if (closed || finished) return;
    finished = true;
    closed = true;
    handlers.onResult(result);
  };
  const fail = (message: string) => {
    if (closed || finished) return;
    finished = true;
    handlers.onError(message);
  };

  const poll = async (runId: string) => {
    let since = 0;
    const deadline = Date.now() + POLL_TIMEOUT_MS;

    while (!closed && !finished) {
      if (Date.now() > deadline) throw new Error("run timed out");

      const res = await fetch(
        url(`/api/poll?run=${encodeURIComponent(runId)}&since=${since}`),
        { signal, headers: { accept: "application/json" } },
      );

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${text.slice(0, 200)}`.trim());
      }

      const data = await res.json();

      // The server sends all seven stage rows every time; the caller merges by
      // key, so repeating them is idempotent.
      for (const stage of data.stages ?? []) handlers.onStage?.(stage);
      for (const line of data.lines ?? []) handlers.onLog?.(line);

      // Content built so far. Absent on a server that only sends the final
      // result, which is why the caller's handler is optional.
      if (data.partial) handlers.onPartial?.(data.partial as Partial<PipelineResult>);

      if (typeof data.next === "number") since = data.next;

      if (data.state === "done") {
        finish(data.result as PipelineResult);
        return;
      }
      if (data.state === "error") {
        fail(data.error ?? "the run failed");
        return;
      }

      await abortable(POLL_INTERVAL_MS, signal);
    }
  };

  (async () => {
    // Phase 1 — start the run. A failure here is the only place we fall back:
    // once a run has begun, restarting it elsewhere would double the work.
    let runId: string;
    try {
      const res = await fetch(url("/api/run"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query }),
        signal,
      });

      if (NOT_IMPLEMENTED.has(res.status)) throw new NoPollingTransport();

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${text.slice(0, 200)}`.trim());
      }

      const started = await res.json();
      if (!started?.run) throw new NoPollingTransport("server returned no run id");
      runId = started.run as string;
    } catch (e) {
      if (closed || finished) return;
      if (e instanceof DOMException && e.name === "AbortError") {
        closed = true;
        return;
      }
      streamRun(query, handlers, signal, finish, fail);
      return;
    }

    // Phase 2 — follow it. A failure in here is a real failure, not a fallback.
    try {
      await poll(runId);
    } catch (e) {
      if (closed || finished) return;
      if (e instanceof DOMException && e.name === "AbortError") {
        closed = true;
        return;
      }
      fail(e instanceof Error ? e.message : String(e));
    }
  })();

  return () => {
    closed = true;
  };
}

/**
 * The streaming transport: SSE for progress, with a blocking POST as the last
 * resort. Used when /api/run is unavailable.
 */
function streamRun(
  query: string,
  handlers: RunHandlers,
  signal: AbortSignal | undefined,
  finish: (result: PipelineResult) => void,
  fail: (message: string) => void,
): void {
  let cancelled = false;

  // EventSource fires a trailing `error` whenever the server closes the stream —
  // including a clean close right after the result frame — and treating that as
  // a failure would mark every successful run as "stream interrupted".
  let finished = false;
  const settleResult = (r: PipelineResult) => {
    if (finished || cancelled) return;
    finished = true;
    finish(r);
  };
  const settleError = (m: string) => {
    if (finished || cancelled) return;
    finished = true;
    fail(m);
  };

  const blocking = async () => {
    if (cancelled || finished) return;
    try {
      const res = await fetch(url("/api/query"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query }),
        signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${text.slice(0, 300)}`);
      }
      settleResult((await res.json()) as PipelineResult);
    } catch (e) {
      settleError(e instanceof Error ? e.message : String(e));
    }
  };

  if (typeof EventSource === "undefined") {
    void blocking();
    return;
  }

  let es: EventSource | null = null;
  try {
    es = new EventSource(url(`/api/stream?q=${encodeURIComponent(query)}`));
  } catch {
    void blocking();
    return;
  }

  let gotAnything = false;

  es.onmessage = (ev) => {
    if (cancelled || finished) return;
    let frame: StreamEvent;
    try {
      frame = JSON.parse(ev.data) as StreamEvent;
    } catch {
      return;
    }
    gotAnything = true;
    switch (frame.type) {
      case "stage":
        handlers.onStage?.(frame.stage);
        break;
      case "log":
        handlers.onLog?.(frame.line);
        break;
      case "partial":
        handlers.onPartial?.(frame.result);
        break;
      case "result":
        settleResult(frame.result);
        es?.close();
        break;
      case "error":
        settleError(frame.message);
        es?.close();
        break;
    }
  };

  es.onerror = () => {
    es?.close();
    es = null;
    // A finished run ignores the trailing close-event; see `finished` above.
    if (finished || cancelled) return;
    // Never streamed at all -> the endpoint is missing; try the blocking route.
    if (!gotAnything) void blocking();
    else settleError("stream interrupted");
  };

  if (signal) {
    signal.addEventListener("abort", () => {
      cancelled = true;
      es?.close();
      es = null;
    });
  }
}
