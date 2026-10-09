"""Tests for the Colab/UI wiring in ``colab_run`` (no models, no server, no tunnel).

Everything here is pure plumbing: asset discovery, tunnel selection, URL
composition. Starting a real server or a real tunnel is out of scope — those
need the models and the network.
"""

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
