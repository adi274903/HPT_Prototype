"""Tests for environment/.env resolution in ``pt_healthcare.config``.

The bug these guard against: reading the environment at import time freezes the
defaults, so a ``.env`` loaded later — from a Colab cell in a different directory
— was silently ignored for everything except the token.
"""

from pt_healthcare import config

PT_KEYS = (
    "PT_MRF_CSV",
    "PT_QDRANT_PATH",
    "PT_TOP_K",
    "PT_CPT_COLLECTION",
    "PT_HCPCS_COLLECTION",
    "PT_MEDGEMMA_MODEL",
    "PT_EMBED_MODEL",
    "PT_MAX_CANDIDATES",
    "PT_CANDIDATE_TEXT_LIMIT",
    "PT_MAX_NEW_TOKENS",
    "PT_ENV_FILE",
)


def _clear(monkeypatch):
    for key in PT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_accessors_fall_back_to_defaults(monkeypatch):
    _clear(monkeypatch)

    assert config.mrf_csv_path() == config.DEFAULT_MRF_CSV
    assert config.qdrant_dir() == config.DEFAULT_QDRANT_PATH
    assert config.top_k() == config.DEFAULT_TOP_K


def test_accessors_read_the_environment_after_import(monkeypatch):
    """The whole point: setting the env *after* import must still take effect."""
    _clear(monkeypatch)
    monkeypatch.setenv("PT_MRF_CSV", "/tmp/other.csv")
    monkeypatch.setenv("PT_QDRANT_PATH", "/tmp/other_db")
    monkeypatch.setenv("PT_TOP_K", "3")

    assert config.mrf_csv_path() == "/tmp/other.csv"
    assert config.qdrant_dir() == "/tmp/other_db"
    assert config.top_k() == 3


def test_blank_values_fall_back(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PT_MRF_CSV", "")

    assert config.mrf_csv_path() == config.DEFAULT_MRF_CSV


def test_non_numeric_int_falls_back(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PT_TOP_K", "not-a-number")

    assert config.top_k() == config.DEFAULT_TOP_K


def test_refresh_updates_the_module_constants(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PT_TOP_K", "4")

    assert config.TOP_K != 4  # snapshot taken at import

    config.refresh()

    assert config.TOP_K == 4

    monkeypatch.delenv("PT_TOP_K", raising=False)
    config.refresh()


def test_load_dotenv_files_reads_a_file(tmp_path, monkeypatch):
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("PT_MRF_CSV=/from/file.csv\nPT_TOP_K=6\n", encoding="utf-8")

    loaded = config.load_dotenv_files([str(env_file)])

    assert loaded == [str(env_file)]
    assert config.mrf_csv_path() == "/from/file.csv"
    assert config.top_k() == 6


def test_real_environment_beats_the_dotenv(tmp_path, monkeypatch):
    """override=False: an export or Colab secret must win over the file."""
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("PT_MRF_CSV=/from/file.csv\n", encoding="utf-8")
    monkeypatch.setenv("PT_MRF_CSV", "/from/env.csv")

    config.load_dotenv_files([str(env_file)])

    assert config.mrf_csv_path() == "/from/env.csv"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert config.load_dotenv_files([str(tmp_path / "nope.env")]) == []


def test_env_file_candidates_cover_the_colab_cases(monkeypatch):
    _clear(monkeypatch)

    candidates = config.env_file_candidates()

    assert ".env" in candidates          # the working directory (a notebook's cwd)
    assert "/content/.env" in candidates  # Colab's default working directory
    assert any(path.endswith("PT_Healthcare/.env") for path in candidates)


def test_pt_env_file_overrides_the_search_list(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PT_ENV_FILE", "/custom/my.env")

    assert config.env_file_candidates()[0] == "/custom/my.env"
