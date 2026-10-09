/**
 * One run, one result — shared by both pages.
 *
 * The pipeline page renders seven transparency cards from this. The patient page
 * renders a patient-facing answer and, when the patient asks, reveals the same
 * seven cards. Both read the same incrementally-updated `result`, so neither has
 * to wait for the run to finish before it can show anything.
 *
 * Before this existed the two pages had separate content pipelines and the
 * patient page never shared the card renderers at all — which is why "make the
 * blocks live" could not be fixed in one place.
 */

import { runQuery } from "./api";
import type { PipelineResult, Stage } from "./types";

/**
 * Keys whose values are themselves objects that stages fill in one field at a
 * time. Merging these by replacement would drop fields an earlier stage set.
 */
const NESTED_KEYS = ["categorized", "code_plausible", "raw", "rejected"] as const;

/**
 * Fold a partial result into the one being built up.
 *
 * Arrays (`cpt_candidates`, `stages`, `price_summary`) replace wholesale — a
 * later stage sends the complete list, not a delta.
 */
export function mergeResult(
  base: PipelineResult | null,
  patch: Partial<PipelineResult>,
): PipelineResult {
  const next = { ...(base ?? {}) } as Record<string, unknown>;

  for (const [key, value] of Object.entries(patch)) {
    if (value === undefined || value === null) continue;

    if ((NESTED_KEYS as readonly string[]).includes(key)) {
      next[key] = { ...((next[key] as object) ?? {}), ...(value as object) };
      continue;
    }

    next[key] = value;
  }

  return next as unknown as PipelineResult;
}

export interface RunListeners {
  onStage?(stage: Stage): void;
  onLog?(line: string): void;
  /** Fires many times per run, with a growing but still incomplete result. */
  onPartial?(result: PipelineResult): void;
  onResult?(result: PipelineResult): void;
  onError?(message: string): void;
}

export type RunStatus = "idle" | "running" | "done" | "error";

/**
 * The state of a single run, driven by the transport.
 *
 * `result` is populated as partials arrive, so a listener that re-renders on
 * every `onPartial` shows each block the moment its stage produces data.
 */
export class Run {
  result: PipelineResult | null = null;
  stages: Stage[] = [];
  log: string[] = [];
  status: RunStatus = "idle";
  error: string | null = null;

  private cancel: (() => void) | null = null;

  /** Keep the raw log bounded; a 3-minute run produces a lot of lines. */
  static readonly LOG_LIMIT = 500;

  start(query: string, listeners: RunListeners): void {
    this.stop();

    this.result = null;
    this.log = [];
    this.error = null;
    this.status = "running";

    this.cancel = runQuery(query, {
      onStage: (stage) => {
        const i = this.stages.findIndex((s) => s.key === stage.key);
        if (i >= 0) this.stages[i] = { ...this.stages[i], ...stage };
        else this.stages.push(stage);
        listeners.onStage?.(stage);
      },
      onLog: (line) => {
        this.log.push(line);
        if (this.log.length > Run.LOG_LIMIT) {
          this.log.splice(0, this.log.length - Run.LOG_LIMIT);
        }
        listeners.onLog?.(line);
      },
      onPartial: (patch) => {
        // Merge, never replace: this is a fraction of the run's output.
        this.result = mergeResult(this.result, patch);
        listeners.onPartial?.(this.result);
      },
      onResult: (result) => {
        this.result = mergeResult(this.result, result);
        this.status = "done";
        if (result.stages?.length) this.stages = result.stages;
        listeners.onResult?.(this.result);
      },
      onError: (message) => {
        this.status = "error";
        this.error = message;
        listeners.onError?.(message);
      },
    });
  }

  stop(): void {
    this.cancel?.();
    this.cancel = null;
  }

  /** Reset to a freshly-loaded payload, e.g. an embedded one or a demo. */
  load(result: PipelineResult | null): void {
    this.stop();
    this.result = result;
    this.stages = result?.stages ? result.stages.slice() : [];
    this.log = [];
    this.error = null;
    this.status = result ? "done" : "idle";
  }
}
