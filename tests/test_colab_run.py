"""Tests for the Colab/UI wiring in ``colab_run`` (no models, no server, no tunnel).

Everything here is pure plumbing: asset discovery, tunnel selection, URL
composition. Starting a real server or a real tunnel is out of scope — those
need the models and the network.
"""

import io
import json
import sys
import time
import urllib.error
import urllib.request

import pytest

import colab_run
from colab_run import Tunnel, UIServer, check_ui_assets, import_pt_serve, start_tunnel
from pt_healthcare import config


def test_ui_assets_are_vendored():
    assets = check_ui_assets()

    assert assets == {
        "pt_serve.py": True,
        "pt_patient.html": True,
        "pt_frontend.html": True,
    }


def test_import_pt_serve_exposes_the_server_entry_points():
    pt_serve = import_pt_serve()

    for name in ("serve", "launch", "to_payload", "render"):
        assert hasattr(pt_serve, name), f"pt_serve is missing {name}"


def test_vendored_server_accepts_a_single_positional_orchestration_callable():
    """The UI calls ``orchestration(query)`` — the pipeline's run() must match."""
    from pt_healthcare.pipeline import HealthcarePricingPipeline

    import inspect

    signature = inspect.signature(HealthcarePricingPipeline.run)
    parameters = list(signature.parameters)

    assert parameters[0] == "self"
    assert parameters[1] == "user_query"


def test_start_tunnel_none_returns_nothing():
    assert start_tunnel(8000, provider="none") is None
    assert start_tunnel(8000, provider="off") is None
    assert start_tunnel(8000, provider="") is None


def test_start_tunnel_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown tunnel provider"):
        start_tunnel(8000, provider="hamachi")


def test_tunnel_stop_is_idempotent():
    calls = []

    tunnel = Tunnel(url="https://x.trycloudflare.com", provider="cloudflare", _stop=lambda: calls.append(1))

    tunnel.stop()
    tunnel.stop()

    assert calls == [1]
    assert tunnel._stop is None


def test_ui_server_urls_without_a_tunnel():
    ui = UIServer(server=type("S", (), {"port": 8000})(), local_url="http://127.0.0.1:8000/")

    assert ui.url == "http://127.0.0.1:8000/"
    assert ui.page_url == "http://127.0.0.1:8000/"
    assert ui.pages["dashboard"] == "http://127.0.0.1:8000/pipeline"


def test_ui_server_urls_prefer_the_public_tunnel():
    tunnel = Tunnel(url="https://abc123.trycloudflare.com", provider="cloudflare")
    ui = UIServer(
        server=type("S", (), {"port": 8000})(),
        local_url="http://127.0.0.1:8000/",
        page="dashboard",
        tunnel=tunnel,
    )

    assert ui.url == "https://abc123.trycloudflare.com"
    assert ui.page_url == "https://abc123.trycloudflare.com/pipeline"
    assert ui.pages["patient"] == "https://abc123.trycloudflare.com/"


def test_serve_ui_refuses_when_assets_are_missing(monkeypatch):
    monkeypatch.setattr(colab_run, "PATIENT_HTML", "/nonexistent/pt_patient.html")

    with pytest.raises(FileNotFoundError, match="Front-end assets missing"):
        colab_run.serve_ui(pipeline=object())


def test_default_paths_match_the_notebook():
    paths = colab_run.default_paths()

    assert paths["db_archive"].endswith("New_PT_DB_Backups/New_PT_DB.tar.gz")
    assert paths["qdrant_path"] == "/content/New_PT_DB"
    assert paths["mrf_csv"].endswith("everyUPMCmrf_clean (1).csv")


def _clear_token_env(monkeypatch):
    for name in config.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_hf_token_reads_huggingface_api_key(monkeypatch):
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_primary")

    assert colab_run.hf_token() == "hf_primary"


def test_hf_token_accepts_the_hf_token_alias(monkeypatch):
    """A .env using HF_TOKEN (huggingface_hub's own name) must still work."""
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_alias")

    assert colab_run.hf_token() == "hf_alias"


def test_hf_token_accepts_the_hub_token_alias(monkeypatch):
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_hub")

    assert colab_run.hf_token() == "hf_hub"


def test_hf_token_prefers_huggingface_api_key(monkeypatch):
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "alias")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "primary")

    assert colab_run.hf_token() == "primary"


def test_config_exposes_the_same_resolution(monkeypatch):
    _clear_token_env(monkeypatch)
    assert config.hf_token_from_env() is None

    monkeypatch.setenv("HF_TOKEN", "alias")
    assert config.hf_token_from_env() == "alias"


def test_resolve_mrf_csv_precedence(monkeypatch):
    monkeypatch.delenv("PT_MRF_CSV", raising=False)
    assert colab_run.resolve_mrf_csv() == colab_run.DRIVE_MRF_CSV
    assert colab_run.resolve_mrf_csv("/explicit.csv") == "/explicit.csv"

    monkeypatch.setenv("PT_MRF_CSV", "/from/env.csv")
    assert colab_run.resolve_mrf_csv() == "/from/env.csv"
    assert colab_run.resolve_mrf_csv("/explicit.csv") == "/explicit.csv"


def test_resolve_qdrant_path_precedence(monkeypatch):
    monkeypatch.delenv("PT_QDRANT_PATH", raising=False)
    assert colab_run.resolve_qdrant_path() == colab_run.COLAB_DB_DIR
    assert colab_run.resolve_qdrant_path("/explicit") == "/explicit"

    monkeypatch.setenv("PT_QDRANT_PATH", "/tmp/db")
    assert colab_run.resolve_qdrant_path() == "/tmp/db"


def test_load_env_reads_a_file_outside_this_repo(tmp_path, monkeypatch):
    """The Colab case: the .env lives next to the notebook, not in the repo."""
    monkeypatch.delenv("PT_MRF_CSV", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("PT_MRF_CSV=/from/file.csv\n", encoding="utf-8")

    loaded = colab_run.load_env(str(env_file))

    assert loaded == [str(env_file)]
    assert colab_run.resolve_mrf_csv() == "/from/file.csv"

    monkeypatch.delenv("PT_MRF_CSV", raising=False)
    config.refresh()  # leave the module snapshot as we found it


def test_missing_dependencies_reports_absent_modules(monkeypatch):
    monkeypatch.setattr(
        colab_run,
        "REQUIRED_PACKAGES",
        (("pandas", "pandas"), ("definitely_not_installed_xyz", "definitely-not-installed-xyz")),
    )

    assert colab_run.missing_dependencies() == ["definitely-not-installed-xyz"]


def test_ensure_dependencies_is_a_noop_when_everything_is_present(monkeypatch):
    monkeypatch.setattr(colab_run, "REQUIRED_PACKAGES", (("pandas", "pandas"),))

    assert colab_run.ensure_dependencies(auto_install=False) == []


def test_ensure_dependencies_raises_with_the_fix_in_the_message(monkeypatch):
    """Not a ModuleNotFoundError three frames down — a RuntimeError naming the fix."""
    monkeypatch.setattr(colab_run, "missing_dependencies", lambda: ["qdrant-client"])

    with pytest.raises(RuntimeError, match="pip install -q -r requirements.txt"):
        colab_run.ensure_dependencies(auto_install=False)


def test_ensure_dependencies_installs_when_allowed(monkeypatch):
    state = {"missing": ["qdrant-client"]}
    calls = []

    monkeypatch.setattr(colab_run, "missing_dependencies", lambda: state["missing"])

    def fake_install(packages=None, quiet=True):
        calls.append(packages)
        state["missing"] = []
        return 0

    monkeypatch.setattr(colab_run, "install_requirements", fake_install)

    assert colab_run.ensure_dependencies(auto_install=True) == ["qdrant-client"]
    assert calls == [None]  # None -> "pip install -r requirements.txt"


def test_ensure_dependencies_asks_for_a_restart_if_still_missing(monkeypatch):
    monkeypatch.setattr(colab_run, "missing_dependencies", lambda: ["torch"])
    monkeypatch.setattr(colab_run, "install_requirements", lambda packages=None, quiet=True: 0)

    with pytest.raises(RuntimeError, match="Restart session"):
        colab_run.ensure_dependencies(auto_install=True)


# ---------------------------------------------------------------------------
# Backend log pane
# ---------------------------------------------------------------------------
def test_log_buffer_indexes_absolutely_across_eviction():
    buffer = colab_run.LogBuffer(maxlen=3)

    for i in range(5):
        buffer.append(f"line {i}")

    lines, nxt = buffer.snapshot(0)
    assert lines == ["line 2", "line 3", "line 4"]
    assert nxt == 5

    lines, nxt = buffer.snapshot(4)
    assert lines == ["line 4"]
    assert nxt == 5

    lines, nxt = buffer.snapshot(5)
    assert lines == []
    assert nxt == 5

    assert len(buffer) == 3


def test_log_printer_buffers_always_and_only_echoes_under_the_tee(monkeypatch):
    class FakeTee:
        def write(self, _s):
            return len(_s)

        def flush(self):
            pass

    buffer = colab_run.LogBuffer()
    echo = io.StringIO()
    printer = colab_run.log_printer(buffer, echo, tee_type=FakeTee)

    # Not under the tee: print() already reached the notebook, so echoing would
    # duplicate every line.
    printer("one")
    assert buffer.snapshot(0)[0] == ["one"]
    assert echo.getvalue() == ""

    monkeypatch.setattr(sys, "stdout", FakeTee())
    printer("two")

    assert echo.getvalue() == "two\n"
    assert buffer.snapshot(0)[0] == ["one", "two"]


def test_log_printer_detects_the_vendored_tee():
    """A redirection that is not pt_serve's tee must not trigger the echo."""
    pt_serve = import_pt_serve()

    class NotTheTee:
        def write(self, _s):
            return len(_s)

        def flush(self):
            pass

    echo = io.StringIO()
    printer = colab_run.log_printer(colab_run.LogBuffer(), echo, tee_type=pt_serve._Tee)

    import sys as _sys

    original = _sys.stdout
    try:
        _sys.stdout = NotTheTee()
        printer("not-a-tee")
        assert echo.getvalue() == ""

        _sys.stdout = pt_serve._Tee(lambda _line: None)
        printer("under-the-tee")
        assert echo.getvalue() == "under-the-tee\n"
    finally:
        _sys.stdout = original


def test_logging_handler_subclasses_the_vendored_one():
    pt_serve = import_pt_serve()

    handler = colab_run.make_logging_handler(pt_serve._Handler, colab_run.LogBuffer())

    assert issubclass(handler, pt_serve._Handler)
    assert handler is not pt_serve._Handler


def test_install_log_routes_swaps_the_handler_pt_serve_uses():
    pt_serve = import_pt_serve()
    before = pt_serve._Handler

    try:
        handler = colab_run.install_log_routes(pt_serve, colab_run.LogBuffer())
        assert pt_serve._Handler is handler
    finally:
        pt_serve._Handler = before


class _FakePipeline:
    """Just enough pipeline for the server: run(), printer, mrf_data, retriever."""

    def __init__(self, fail: bool = False):
        self.printer = print
        self.mrf_data = None
        self.top_k = 10
        self.retriever = type("R", (), {"client": None})()
        self.fail = fail

    def run(self, query, verbose=True):
        self.printer("=" * 40)
        self.printer("STEP 1/7 - Extracting medical, hospital, insurer, and medication entities")
        self.printer(f"User query: {query}")
        self.printer("Categorization completed in 0.01s")
        self.printer("STEP 7/7 - Generating final user-facing answer")
        self.printer("Answer generation completed in 0.02s")

        if self.fail:
            raise RuntimeError("boom")

        return {
            "categorized": {"medical": ["colonoscopy"], "hospital": [], "insurer": [], "medication": []},
            "cpt_candidates": [],
            "hcpcs_candidates": [],
            "output": {"use_codes": "cpt", "cpt_list": ["45378"], "hcpcs_list": []},
            "code_plausible": {"match_count": 0, "price_summary": None, "filters_used": {}},
            "answer": "test answer",
            "total_seconds": 0.02,
        }


def _start_server(pipeline=None):
    return colab_run.serve_ui(
        pipeline=pipeline or _FakePipeline(),
        tunnel="none",
        display=False,
        verbose=False,
    )


def test_serve_ui_exposes_the_log_pane_and_the_log_endpoint():
    fake = _FakePipeline()

    ui = _start_server(fake)

    try:
        base = ui.local_url.rstrip("/")

        with urllib.request.urlopen(base + "/logs", timeout=5) as response:
            assert response.status == 200
            body = response.read().decode("utf-8")

        assert "Backend log" in body
        assert "/api/log" in body

        with urllib.request.urlopen(base + "/api/log?since=0", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["lines"] == [] and payload["next"] == 0

        # serve_ui replaced the pipeline's default printer with the buffering one
        # that feeds this endpoint.
        assert fake.printer is not print
    finally:
        ui.stop()


def _post_json(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_to_completion(base, query, timeout=15):
    started = _post_json(base + "/api/run", {"query": query})
    run_id = started["run"]

    deadline = time.time() + timeout
    payload = None

    while time.time() < deadline:
        payload = _get_json(f"{base}/api/poll?run={run_id}&since=0")
        if payload["state"] != "running":
            return payload
        time.sleep(0.05)

    raise AssertionError(f"run never finished: {payload}")


def test_polling_transport_carries_stages_and_the_payload():
    ui = _start_server()

    try:
        base = ui.local_url.rstrip("/")
        payload = _run_to_completion(base, "Cost of colonoscopy in pittsburgh?")

        assert payload["state"] == "done"

        states = {stage["key"]: stage["state"] for stage in payload["stages"]}
        assert states["entities"] == "done"
        assert states["answer"] == "done"

        # The client consumes the same payload contract as /api/stream.
        assert payload["result"]["mode"] == "live"
        assert payload["result"]["query"] == "Cost of colonoscopy in pittsburgh?"
        assert payload["result"]["decision"]["cpt_list"] == ["45378"]

        assert payload["seconds"] is not None
        assert payload["next"] == len(payload["lines"])
    finally:
        ui.stop()


def test_polling_transport_reports_new_lines_incrementally():
    ui = _start_server()

    try:
        base = ui.local_url.rstrip("/")
        run_id = _post_json(base + "/api/run", {"query": "colonoscopy"})["run"]

        # Wait for the run to finish first: while it is still going, more lines
        # legitimately appear between two polls and the counts race.
        deadline = time.time() + 10
        final = None
        while time.time() < deadline:
            final = _get_json(f"{base}/api/poll?run={run_id}&since=0")
            if final["state"] != "running":
                break
            time.sleep(0.05)

        assert final is not None and final["state"] == "done"
        assert final["next"] == len(final["lines"])

        # `since` is an absolute high-water mark: past it, nothing comes back.
        beyond = _get_json(f"{base}/api/poll?run={run_id}&since={final['next']}")
        assert beyond["lines"] == []
        assert beyond["next"] == final["next"]

        # And it really does slice, rather than always returning everything.
        partial = _get_json(f"{base}/api/poll?run={run_id}&since=1")
        assert partial["lines"] == final["lines"][1:]
        assert partial["next"] == final["next"]
    finally:
        ui.stop()


def test_polling_transport_surfaces_a_failure():
    ui = _start_server(_FakePipeline(fail=True))

    try:
        base = ui.local_url.rstrip("/")
        payload = _run_to_completion(base, "colonoscopy")

        assert payload["state"] == "error"
        assert "boom" in payload["error"]
        assert payload["result"] is None
    finally:
        ui.stop()


def test_polling_rejects_unknown_runs():
    ui = _start_server()

    try:
        base = ui.local_url.rstrip("/")

        with pytest.raises(urllib.error.HTTPError) as error:
            _get_json(base + "/api/poll?run=nope&since=0")

        assert error.value.code == 404
    finally:
        ui.stop()


def test_polling_rejects_an_empty_query():
    ui = _start_server()

    try:
        base = ui.local_url.rstrip("/")

        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(base + "/api/run", {"query": "   "})

        assert error.value.code == 400
    finally:
        ui.stop()


def test_stage_rows_agree_with_pt_serve_on_a_real_log():
    """The polling transport must derive the same stages the SSE path does."""
    pt_serve = import_pt_serve()

    log = "\n".join(
        [
            "STEP 1/7 - Extracting medical, hospital, insurer, and medication entities",
            "Categorization completed in 13.85s",
            "STEP 2/7 - Retrieving CPT candidates from Qdrant",
            "CPT retrieval completed in 0.25s",
            "MRF rows matched: 95",
        ]
    )

    mine = {stage["key"]: stage["seconds"] for stage in colab_run.stage_rows(log.splitlines(), done=True)}
    theirs = {stage["key"]: stage["seconds"] for stage in pt_serve._stages_from_log(log)}

    assert mine == theirs
    assert mine["entities"] == 13.85
    assert mine["cpt"] == 0.25


def test_stage_rows_keeps_a_running_step_active():
    rows = colab_run.stage_rows(
        [
            "STEP 1/7 - Extracting",
            "Categorization completed in 1.00s",
            "STEP 2/7 - Retrieving CPT candidates from Qdrant",
        ]
    )
    states = {stage["key"]: stage["state"] for stage in rows}

    assert states["entities"] == "done"
    assert states["cpt"] == "active"      # not ticked early
    assert states["hcpcs"] == "pending"

    finished = colab_run.stage_rows(
        ["STEP 1/7 - Extracting", "Categorization completed in 1.00s"], done=True
    )
    assert all(stage["state"] == "done" for stage in finished)
