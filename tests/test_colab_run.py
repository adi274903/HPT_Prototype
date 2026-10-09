"""Tests for the Colab/UI wiring in ``colab_run`` (no models, no server, no tunnel).

Everything here is pure plumbing: asset discovery, tunnel selection, URL
composition. Starting a real server or a real tunnel is out of scope — those
need the models and the network.
"""

import io
import json
import sys
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


def test_log_printer_buffers_always_and_only_echoes_under_a_tee(monkeypatch):
    buffer = colab_run.LogBuffer()
    echo = io.StringIO()
    printer = colab_run.log_printer(buffer, echo)

    # No tee: sys.stdout is the real stdout, so print() already reached the
    # notebook and echoing would duplicate it.
    printer("one")
    assert buffer.snapshot(0)[0] == ["one"]
    assert echo.getvalue() == ""

    class FakeTee:
        def write(self, _s):
            return len(_s)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", FakeTee())
    printer("two")

    assert echo.getvalue() == "two\n"
    assert buffer.snapshot(0)[0] == ["one", "two"]


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

    def __init__(self):
        self.printer = print
        self.mrf_data = None
        self.top_k = 10
        self.retriever = type("R", (), {"client": None})()

    def run(self, query, verbose=True):
        self.printer("STEP 1/7 - test")
        self.printer("Categorization completed in 0.01s")
        return {}


def test_serve_ui_exposes_the_log_pane_and_the_log_endpoint():
    fake = _FakePipeline()

    ui = colab_run.serve_ui(
        pipeline=fake,
        tunnel="none",
        display=False,
        verbose=False,
    )

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
