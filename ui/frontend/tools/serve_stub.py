#!/usr/bin/env python3
"""
Serve the front end against the stub pipeline, for browser-level tests.

    python3 tools/serve_stub.py [port]

Prints `READY <url>` on stdout once listening, then blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "colab"))

import _stub_pipeline as stub  # noqa: E402
import pt_serve  # noqa: E402


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8990
    handle = pt_serve.serve(
        orchestration=stub.orchestration,
        mrf_data=stub.mrf_data,
        client=stub.client,
        top_k=stub.TOP_K,
        port=port,
    )
    print(f"READY {handle.url.rstrip('/')}", flush=True)
    try:
        handle.thread.join()
    except KeyboardInterrupt:
        handle.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
