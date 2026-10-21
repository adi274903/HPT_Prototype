"""Start a faster inference backend next to the notebook, and point the app at it.

The pipeline's model calls go through ``pt_healthcare.llm.MedGemmaEngine``, which
only needs a callable that returns transformers' chat shape. Both backends below
speak ``POST /v1/chat/completions``, so ``load_models()`` hands back
``OpenAICompatPipe`` instead of loading 8 GB of bfloat16 weights.

    from pt_healthcare.backend import start_vllm, use
    use(start_vllm("google/medgemma-1.5-4b-it"))     # Colab: pip install vllm first
    # or, with a GGUF and a llama-server binary:
    use(start_llama_server("/content/models/medgemma-1.5-4b-it-Q4_K_M.gguf",
                           llama_server="/content/llama.cpp/build/bin/llama-server"))

Nothing in the pipeline imports this module; it is notebook scaffolding. Pass
``n_gpu_layers=0`` to llama.cpp when the card is too small for the quant.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

LOG_DIR = os.environ.get("PT_LOG_DIR", "logs")

_PROCS: List[subprocess.Popen] = []


def _log_path(name: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"{name}.log")


def _tail(path: str, lines: int = 15) -> str:
    try:
        with open(path, "r", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return ""


def wait_ready(base_url: str, timeout: float = 900, every: float = 2.0) -> str:
    """Block until the server answers ``/v1/models``; return the base URL.

    Starting is slow on purpose: vLLM compiles CUDA graphs and captures memory
    before it binds, and llama.cpp maps the GGUF. Polling beats guessing.
    """
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/models", timeout=3) as resp:
                if resp.status == 200:
                    return base
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(every)
    raise RuntimeError(f"{base} did not come up in {timeout:.0f}s ({last})")


def start_vllm(
    model: str = "google/medgemma-1.5-4b-it",
    port: int = 8000,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "bfloat16",
    extra: Optional[List[str]] = None,
    timeout: float = 900,
) -> str:
    """Serve ``model`` with vLLM and return its base URL.

    vLLM is the Colab-friendly option: one ``pip install``, paged KV cache and
    CUDA graphs, no compile step. A 4B model fits a 16 GB T4 in bfloat16; on a
    card that cannot hold it, quantise (``--quantization awq``) or use llama.cpp.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--served-model-name", model,
    ] + list(extra or [])

    log = _log_path("vllm")
    print("starting vLLM:", " ".join(cmd[2:]), "->", log)
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    _PROCS.append(proc)

    try:
        return wait_ready(f"http://127.0.0.1:{port}", timeout=timeout)
    except RuntimeError:
        tail = _tail(log)
        proc.terminate()
        raise RuntimeError(f"vLLM did not start; last lines:\n{tail}") from None


def start_llama_server(
    gguf: str,
    port: int = 8080,
    llama_server: Optional[str] = None,
    n_gpu_layers: int = 99,
    ctx: int = 8192,
    threads: Optional[int] = None,
    timeout: float = 300,
) -> str:
    """Serve a GGUF with ``llama-server`` and return its base URL.

    Verified against MedGemma 1.5 4B Q8_0. ``n_gpu_layers=0`` runs on the CPU,
    which is the fallback when the GPU is occupied or too small for the quant.
    """
    binary = llama_server or os.environ.get("PT_LLAMA_SERVER") or shutil.which("llama-server")
    if not binary or not os.path.exists(binary):
        raise FileNotFoundError(
            "llama-server not found: pass llama_server=…, set PT_LLAMA_SERVER, or "
            "build it (cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server)"
        )
    if not os.path.exists(gguf):
        raise FileNotFoundError(f"no GGUF at {gguf}")

    cmd = [
        binary, "-m", gguf,
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "--jinja",  # honour the model's own chat template (Gemma turn markers)
    ]
    if threads:
        cmd += ["-t", str(threads)]

    log = _log_path("llama-server")
    print("starting llama-server:", " ".join(cmd[1:]), "->", log)
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    _PROCS.append(proc)

    try:
        return wait_ready(f"http://127.0.0.1:{port}", timeout=timeout)
    except RuntimeError:
        tail = _tail(log)
        proc.terminate()
        raise RuntimeError(f"llama-server did not start; last lines:\n{tail}") from None


def use(base_url: str) -> str:
    """Point the pipeline at ``base_url`` for the rest of this kernel."""
    os.environ["PT_LLM_BASE_URL"] = base_url
    print(f"PT_LLM_BASE_URL={base_url} — the pipeline will call this backend, "
          f"not the in-process transformers pipeline")
    return base_url


def stop() -> None:
    """Stop every server this module started."""
    while _PROCS:
        proc = _PROCS.pop()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
