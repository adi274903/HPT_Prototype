/** Shared UI state and step metadata. */

import type { HealthInfo, PipelineResult, Stage } from "./types";

export type RunStatus = "idle" | "running" | "done" | "error";

export interface AppState {
  theme: "light" | "dark";
  status: RunStatus;
  query: string;
  live: boolean;
  health: HealthInfo | null;
  result: PipelineResult | null;
  /** Live stage states while a streamed run is in flight. */
  stages: Stage[];
  log: string[];
  error: string | null;
  /** Price-table sort. */
  sort: { col: string; dir: "asc" | "desc" };
  showReasoning: boolean;
  demoIndex: number;
  booted: boolean;
}

export const STEP_META: ReadonlyArray<{ key: string; n: number; title: string; blurb: string }> = [
  { key: "entities", n: 1, title: "Entity extraction", blurb: "MedGemma 1.5-4B splits the question into medical, hospital, insurer and medication spans." },
  { key: "cpt", n: 2, title: "CPT retrieval", blurb: "MedTE embeddings over the CPT catalogue in Qdrant — candidate codes, not answers." },
  { key: "hcpcs", n: 3, title: "HCPCS retrieval", blurb: "The same semantic search against the HCPCS catalogue." },
  { key: "decision", n: 4, title: "Code decision", blurb: "The model picks from the retrieved candidates only; it cannot invent a code." },
  { key: "validation", n: 5, title: "Code validation", blurb: "Selected codes are intersected with the retrieved set before they touch the data." },
  { key: "mrf", n: 6, title: "MRF price filter", blurb: "Exact-key lookup over the CMS machine-readable file — where the dollars come from." },
  { key: "answer", n: 7, title: "Final answer", blurb: "Grounded on the matched rows, with methodology and caveats attached." },
];

export function blankStages(): Stage[] {
  return STEP_META.map((s) => ({
    n: s.n,
    key: s.key,
    title: s.title,
    state: "pending" as const,
    seconds: null,
    detail: null,
  }));
}

export function initialState(): AppState {
  return {
    theme: "light",
    status: "idle",
    query: "",
    live: false,
    health: null,
    result: null,
    stages: blankStages(),
    log: [],
    error: null,
    sort: { col: "standard_charge|negotiated_dollar", dir: "asc" },
    showReasoning: false,
    demoIndex: 0,
    booted: false,
  };
}

export function stagesOf(result: PipelineResult | null, fallback: Stage[]): Stage[] {
  if (result?.stages?.length) return result.stages;
  return fallback;
}

export function overallState(stages: Stage[]): { done: number; total: number; active: Stage | null } {
  const total = stages.length || 7;
  const done = stages.filter((s) => s.state === "done").length;
  const active = stages.find((s) => s.state === "active") ?? null;
  return { done, total, active };
}
