/**
 * Wire contract between the Colab (Python) backend and this front end.
 *
 * `PipelineResult` is a 1:1 JSON projection of the dict returned by
 * `orchestration(user_query)` in Copy_of_Try_PT-3.ipynb, plus the stage
 * timeline that the server reconstructs from the pipeline's own stdout.
 */

export type UseCodes = "cpt" | "hcpcs" | "both" | "none" | (string & {});

export interface Candidate {
  rank: number;
  score: number;
  code: string;
  text: string;
}

export interface Entities {
  medical: string[];
  hospital: string[];
  insurer: string[];
  medication: string[];
}

export interface Decision {
  use_codes: UseCodes;
  cpt_list: string[];
  hcpcs_list: string[];
}

export interface FiltersUsed {
  cpt_codes: string[];
  hcpcs_codes: string[];
  insurers: string[];
  hospitals: string[];
}

/** A MRF price row. Values are scalars because the frame is heterogeneous. */
export type PriceRow = Record<string, string | number | null>;

export interface CodePlausible {
  match_count: number;
  filters_used: FiltersUsed;
  price_summary: PriceRow[];
  /** Rows present in `matches` before the server capped the payload. */
  matched_rows_total?: number;
}

export type StageState = "pending" | "active" | "done" | "error";

export interface Stage {
  n: number;
  key: string;
  title: string;
  state: StageState;
  seconds?: number | null;
  detail?: string | null;
}

export interface PipelineResult {
  query: string;
  mode: "live" | "demo";
  categorized: Entities;
  cpt_candidates: Candidate[];
  hcpcs_candidates: Candidate[];
  decision: Decision;
  /** Codes the decision model proposed that were not in the retrieved set. */
  rejected: { cpt: string[]; hcpcs: string[] };
  code_plausible: CodePlausible;
  answer: string;
  /** Truncated raw model text, for the transparency panels. */
  raw: { categorization: string; decision: string };
  stages: Stage[];
  total_seconds: number;
  created_at: string;
}

export interface HealthInfo {
  ok: boolean;
  mode: "live";
  mrf_rows: number | null;
  collections: Record<string, number>;
  top_k: number;
  models: string[];
}

/** SSE frames sent while a query runs. */
export type StreamEvent =
  | { type: "stage"; stage: Stage }
  | { type: "log"; line: string }
  /**
   * A partially-built result, sent as each stage produces its content.
   *
   * This is the frame that lets a block render the moment it has data. The
   * `result` frame below carries everything at once, which is why every panel
   * used to stay empty until the run finished — even though the categorizer's
   * output was on the wire at roughly second 12 of a 200-second run.
   *
   * Fields arrive progressively and are merged by key, so every field here is
   * optional and a partial must never be treated as complete.
   */
  | { type: "partial"; result: Partial<PipelineResult> }
  | { type: "result"; result: PipelineResult }
  | { type: "error"; message: string };
