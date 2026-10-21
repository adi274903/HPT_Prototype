"""A stand-in for ``vllm serve`` that answers from the answer key, on purpose.

This is a **test fixture, not a model**. It exists so the harness can be verified
end to end — HTTP, prompt plumbing, JSON parsing, metric arithmetic, report — on a
machine with no GPU, and so the metrics can be checked against known error rates:
the stub is built to be wrong on a fixed fraction of cases, so a correct harness
must land on those fractions.

    python eval_code_gen/stub_vllm.py --port 8123 --dataset eval_dataset

Then point the runner at it::

    python -m pt_healthcare.eval_runner --base-url http://127.0.0.1:8123/v1 \\
        --model stub-gazetteer --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

QUERY_RX = re.compile(r"Query: ([^\n]+)", re.DOTALL)  # prompts end with the real query
ERROR_EVERY = {"insurer_drop": 5, "code_noise": 4, "intent_wrong": 6,
               "rate_wrong": 7, "false_price": 4}


def extract_array(prompt: str, label: str) -> List[Dict[str, Any]]:
    """Pull the JSON list that follows ``label:`` out of a prompt, depth-aware."""
    start = prompt.find(f"{label}:")
    if start < 0:
        return []
    start = prompt.find("[", start)
    if start < 0:
        return []
    depth = 0
    for index in range(start, len(prompt)):
        if prompt[index] == "[":
            depth += 1
        elif prompt[index] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(prompt[start:index + 1])
                except json.JSONDecodeError:
                    return []
    return []


def load(dataset: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    cases = [json.loads(line) for line in
             (dataset / "pt_eval_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = {record["eval_id"]: record for record in
               (json.loads(line) for line in
                (dataset / "pt_eval_v2_targets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip())}
    return cases, targets


class Stub:
    def __init__(self, cases: List[Dict[str, Any]], targets: Dict[str, Dict[str, Any]],
                 rates: Dict[str, List[float]]) -> None:
        self.by_query = {case["eval_query"]: case for case in cases}
        self.targets = targets
        self.rates = rates

    def identify(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Find which case this prompt is about, whatever label the prompt uses.

        Matching on a literal "Query:" line breaks the moment a prompt phrases it
        differently, so the stub looks for a known question instead.
        """
        for query in self.by_query:
            if len(query) > 20 and query in prompt:
                return self.by_query[query]
        matches = QUERY_RX.findall(prompt)
        return self.by_query.get(matches[-1].strip()) if matches else None

    @staticmethod
    def ordinal(eval_id: str) -> int:
        return int(str(eval_id).split("_")[1])

    def reply(self, prompt: str) -> str:
        case = self.identify(prompt)
        if case is None:
            return json.dumps({"error": "unknown query"})
        eval_id = str(case["eval_id"])
        target = self.targets[eval_id]
        index = self.ordinal(eval_id)

        if "intent classifier" in prompt:
            label = str(target["intent"])
            if index % ERROR_EVERY["intent_wrong"] == 0:
                others = [value for value in ("price_lookup", "hospital_comparison", "cash_vs_insurance")
                          if value != label]
                label = others[index % len(others)]
            return json.dumps({"intent": label})

        if "entity extractor" in prompt:
            entities = {field: [item["text"] for item in target["entities"][field]]
                        for field in ("medical", "hospital", "insurer", "medication")}
            if index % ERROR_EVERY["insurer_drop"] == 0:
                entities["insurer"] = []
            return json.dumps(entities)

        if prompt.startswith("Select billing-code candidates"):
            return self.decide(prompt, case, index)

        if "healthcare navigation assistant" in prompt:
            return self.answer(case, target, index)

        return "{}"

    def decide(self, prompt: str, case: Dict[str, Any], index: int) -> str:
        cpt = [str(item.get("code", "")).upper() for item in extract_array(prompt, "CPT candidates")]
        hcpcs = [str(item.get("code", "")).upper() for item in extract_array(prompt, "HCPCS candidates")]

        truth = [str(code) for code in case.get("ground_truth_code_MRF") or []]
        chosen_cpt = [code for code in truth if code in cpt][:3]
        chosen_hcpcs = [code for code in truth if code in hcpcs][:3]

        if index % ERROR_EVERY["code_noise"] == 0:
            decoys = [code for code in cpt if code not in truth][:1]
            chosen_cpt += decoys
            chosen_cpt.append("99999")  # never in any catalog: exercises the validator

        use_codes = "both" if chosen_cpt and chosen_hcpcs else (
            "cpt" if chosen_cpt else "hcpcs" if chosen_hcpcs else "none")
        return json.dumps({"use_codes": use_codes, "cpt_list": chosen_cpt,
                           "hcpcs_list": chosen_hcpcs})

    def answer(self, case: Dict[str, Any], target: Dict[str, Any], index: int) -> str:
        eval_id = str(case["eval_id"])
        keyed = self.rates.get(eval_id, [])
        if not keyed:
            if index % ERROR_EVERY["false_price"] == 0:
                return "Based on the records, this service costs $999.99."
            return "There is no negotiated rate on file for this service at that facility."
        if index % ERROR_EVERY["rate_wrong"] == 0:
            return "Based on the records, the negotiated amount is $999.99."
        amount = keyed[0]
        return (f"Based on the file, the negotiated amount is ${amount:,.2f}. "
                f"This is an estimate from the published price file and not a bill.")


def build_server(stub: Stub, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            prompt = ""
            for message in payload.get("messages") or []:
                if message.get("role") == "user":
                    prompt = str(message.get("content") or "")
            try:
                text = stub.reply(prompt)
            except Exception as exc:  # noqa: BLE001 - surface it as 500, not a dropped socket
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps({
                "id": "stub", "object": "chat.completion", "model": payload.get("model", "stub"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            body = json.dumps({"object": "list", "data": [{"id": "stub-gazetteer"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dataset", default="eval_dataset")
    parser.add_argument("--db", default=".cache/mrf_eval.duckdb")
    args = parser.parse_args()

    cases, targets = load(Path(args.dataset))

    import duckdb

    con = duckdb.connect(str(args.db), read_only=True)
    rates: Dict[str, List[float]] = {}
    for case in cases:
        row_ids = [str(value) for value in case.get("ground_truth_row_ids") or []]
        if not row_ids:
            continue
        placeholders = ",".join("?" * len(row_ids))
        values = [row[0] for row in con.execute(
            f"SELECT rate FROM rows_ext WHERE row_id IN ({placeholders}) AND rate IS NOT NULL",
            row_ids).fetchall()]
        rates[str(case["eval_id"])] = sorted(float(value) for value in values)
    con.close()

    server = build_server(Stub(cases, targets, rates), args.host, args.port)
    print(f"stub vLLM listening on http://{args.host}:{args.port}/v1 "
          f"({len(cases)} cases, {sum(1 for v in rates.values() if v)} priced)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
