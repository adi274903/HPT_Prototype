"""The two in-process backends, tested without a GPU or the real packages.

A fake ``llama_cpp`` / ``vllm`` module records what the wrapper sends, so the
contract that matters is pinned: the transformers chat shape in, the transformers
chat shape out, greedy decoding by default, no server anywhere.
"""

from __future__ import annotations

import sys
import types

import pytest

from pt_healthcare import config
from pt_healthcare.backends import LlamaCppPipe, VLLMPipe, to_openai_messages


class _FakeLlama:
    calls: list = []
    answer = "about $133"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).calls.append({"init": kwargs})

    def create_chat_completion(self, **kwargs):
        type(self).calls.append({"chat": kwargs})
        return {
            "choices": [{"message": {"role": "assistant", "content": type(self).answer}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        }

    def close(self):
        type(self).calls.append({"close": True})


@pytest.fixture()
def fake_llamacpp(monkeypatch, tmp_path):
    _FakeLlama.calls = []
    module = types.ModuleType("llama_cpp")
    module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    return str(gguf)


def test_llamacpp_returns_the_transformers_shape(fake_llamacpp):
    pipe = LlamaCppPipe(fake_llamacpp, n_ctx=2048, n_gpu_layers=0)
    out = pipe(text=[{"role": "user", "content": [{"type": "text", "text": "cost?"}]}], max_new_tokens=64)
    assert out == [{"generated_text": [{"role": "assistant", "content": "about $133"}]}]
    chat = [c for c in _FakeLlama.calls if "chat" in c][0]["chat"]
    assert chat["messages"] == [{"role": "user", "content": "cost?"}]
    assert chat["max_tokens"] == 64
    assert chat["temperature"] == 0.0, "do_sample=False must decode greedily"


def test_llamacpp_loads_the_weights_with_the_asked_placement(fake_llamacpp):
    LlamaCppPipe(fake_llamacpp, n_ctx=4096, n_gpu_layers=-1, n_threads=12)
    init = _FakeLlama.calls[0]["init"]
    assert init["model_path"] == fake_llamacpp
    assert init["n_ctx"] == 4096 and init["n_gpu_layers"] == -1 and init["n_threads"] == 12


def test_llamacpp_missing_gguf_says_which_path(fake_llamacpp):
    with pytest.raises(FileNotFoundError, match="/nope/missing.gguf"):
        LlamaCppPipe("/nope/missing.gguf")


def test_llamacpp_close_releases_the_model(fake_llamacpp):
    LlamaCppPipe(fake_llamacpp).close()
    assert {"close": True} in _FakeLlama.calls


class _FakeOutput:
    def __init__(self, text):
        self.text = text
        self.token_ids = [1, 2, 3]


class _FakeCompletion:
    prompt_token_ids = [9, 9]
    outputs = [_FakeOutput("$133 negotiated")]


class _FakeLLM:
    calls: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).calls.append({"init": kwargs})

    def chat(self, **kwargs):
        type(self).calls.append({"chat": kwargs})
        return [_FakeCompletion()]


@pytest.fixture()
def fake_vllm(monkeypatch):
    _FakeLLM.calls = []
    module = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.LLM = _FakeLLM
    module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def test_vllm_returns_the_transformers_shape(fake_vllm):
    pipe = VLLMPipe(model="google/medgemma-1.5-4b-it", max_model_len=8192)
    out = pipe([{"role": "user", "content": "cost?"}], max_new_tokens=32)
    assert out[0]["generated_text"][-1]["content"] == "$133 negotiated"
    assert pipe.last_usage == {"prompt_tokens": 2, "completion_tokens": 3}

    init = _FakeLLM.calls[0]["init"]
    assert init["model"] == "google/medgemma-1.5-4b-it"
    assert init["max_model_len"] == 8192 and init["dtype"] == "bfloat16"

    chat = [c for c in _FakeLLM.calls if "chat" in c][0]["chat"]
    params = chat["sampling_params"]
    assert params.max_tokens == 32 and params.temperature == 0.0


def test_both_backends_refuse_images_clearly(fake_llamacpp, fake_vllm):
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    for pipe in (LlamaCppPipe(fake_llamacpp), VLLMPipe()):
        with pytest.raises(ValueError, match="text-only"):
            pipe(text=image)


def test_the_switch_is_one_setting(monkeypatch):
    monkeypatch.setenv("PT_LLM_BACKEND", "LLAMACPP")
    assert config.llm_backend() == "llamacpp"
    monkeypatch.setenv("PT_LLM_BACKEND", "")
    assert config.llm_backend() == "transformers", "unset must keep the old path"
    monkeypatch.setenv("PT_LLM_BACKEND", "llamacpp")
    monkeypatch.setenv("PT_LLM_GGUF", "/content/medgemma-1.5-4b-it-Q4_K_M.gguf")
    assert config.llm_gguf_path().endswith("Q4_K_M.gguf")


def test_a_typo_does_not_silently_fall_back(monkeypatch):
    """load_models must not load 8 GB of weights because of a typo."""
    monkeypatch.setenv("PT_LLM_BACKEND", "lamacpp")
    from pt_healthcare.models import load_models

    with pytest.raises(ValueError, match="unknown PT_LLM_BACKEND='lamacpp'"):
        load_models()


def test_llamacpp_without_a_gguf_says_what_is_missing(monkeypatch):
    monkeypatch.setenv("PT_LLM_BACKEND", "llamacpp")
    monkeypatch.delenv("PT_LLM_GGUF", raising=False)
    from pt_healthcare.models import load_models

    with pytest.raises(ValueError, match="PT_LLM_GGUF"):
        load_models()


def test_a_prebuilt_llm_is_shared_instead_of_loaded_twice(monkeypatch):
    """The UI and the eval must hold one copy of the weights, not two."""
    import sys
    import types

    from pt_healthcare.backends import VLLMPipe

    built = []

    class LLM:
        def __init__(self, **kwargs):
            built.append(kwargs)

    module = types.ModuleType("vllm")
    module.LLM = LLM
    module.SamplingParams = lambda **kwargs: type("Params", (), kwargs)()
    monkeypatch.setitem(sys.modules, "vllm", module)

    shared = LLM(model="already-loaded")
    built.clear()  # the fixture above has built exactly one, and only one is wanted
    pipe = VLLMPipe(llm=shared)

    assert pipe._llm is shared
    assert built == [], "passing an instance must not construct a second one"
