"""The OpenAI-compatible backend, tested against a fake server.

No GPU, no model: what matters here is that the shim speaks the same shape as
the transformers pipeline (``[{"generated_text": [... assistant ...]}]``),
normalises the engine's content parts, and reports a dead server in words a
reader can act on.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pt_healthcare import config
from pt_healthcare.api_pipe import OpenAICompatPipe, to_openai_messages


class _Handler(BaseHTTPRequestHandler):
    seen = []
    reply = "It costs about $133."
    status = 200

    def do_POST(self):  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append({"path": self.path, "body": body, "auth": self.headers.get("authorization")})
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b'{"error": "model not found"}')
            return
        out = {
            "choices": [{"message": {"role": "assistant", "content": type(self).reply}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        raw = json.dumps({"data": [{"id": "medgemma"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def server():
    _Handler.seen = []
    _Handler.status = 200
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_parts_are_flattened_to_a_string():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "there"}]}]
    assert to_openai_messages(msgs) == [{"role": "user", "content": "hello there"}]


def test_a_bare_string_becomes_a_user_turn():
    assert to_openai_messages("hi") == [{"role": "user", "content": "hi"}]


def test_an_empty_assistant_seed_is_dropped():
    """The transformers chat template seeds it; servers add it themselves."""
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
    assert to_openai_messages(msgs) == [{"role": "user", "content": "q"}]


def test_images_are_refused_clearly():
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    with pytest.raises(ValueError, match="text-only"):
        to_openai_messages(msgs)


def test_call_returns_the_transformers_shape(server):
    pipe = OpenAICompatPipe(server)
    out = pipe(text=[{"role": "user", "content": [{"type": "text", "text": "cost?"}]}], max_new_tokens=64)
    assert out[0]["generated_text"][-1] == {"role": "assistant", "content": "It costs about $133."}
    body = _Handler.seen[-1]["body"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0, "do_sample=False must not sample"
    assert body["messages"] == [{"role": "user", "content": "cost?"}]
    assert pipe.last_usage["completion_tokens"] == 7


def test_positional_messages_work_like_the_decide_call(server):
    """MedGemmaEngine.decide passes messages positionally."""
    pipe = OpenAICompatPipe(server)
    out = pipe([{"role": "user", "content": "pick a code"}], max_new_tokens=32, do_sample=False)
    assert out[0]["generated_text"][-1]["role"] == "assistant"
    assert _Handler.seen[-1]["body"]["messages"][0]["content"] == "pick a code"


def test_sampling_is_forwarded_when_asked(server):
    pipe = OpenAICompatPipe(server)
    pipe(text="hi", do_sample=True, max_new_tokens=8)
    assert _Handler.seen[-1]["body"]["temperature"] > 0


def test_a_url_without_v1_gets_one(server):
    pipe = OpenAICompatPipe(server)
    assert pipe.base_url.endswith("/v1")
    pipe(text="hi")
    assert _Handler.seen[-1]["path"] == "/v1/chat/completions"


def test_server_errors_keep_the_server_words(server):
    _Handler.status = 404
    pipe = OpenAICompatPipe(server)
    with pytest.raises(RuntimeError, match="404 from .*/v1/chat/completions: .*model not found"):
        pipe(text="hi")


def test_a_dead_server_says_so(server):
    pipe = OpenAICompatPipe(server, timeout=1)
    pipe.base_url = "http://127.0.0.1:9/v1"  # nothing listening
    with pytest.raises(RuntimeError, match="cannot reach http://127.0.0.1:9/v1"):
        pipe(text="hi")


def test_health_reads_models(server):
    assert OpenAICompatPipe(server).health()["data"][0]["id"] == "medgemma"


def test_the_env_var_is_what_switches_backends(monkeypatch):
    assert config.llm_base_url() == ""
    monkeypatch.setenv("PT_LLM_BASE_URL", "http://127.0.0.1:8080/v1/")
    assert config.llm_base_url() == "http://127.0.0.1:8080/v1"
    monkeypatch.setenv("PT_LLM_MODEL", "medgemma-1.5-4b-it")
    assert config.llm_model_name() == "medgemma-1.5-4b-it"
