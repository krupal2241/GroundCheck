"""LLM layer: Ollama over a fake HTTP session, fake provider, cache, paid guard,
structured output, the local Hugging Face provider over stand-in modules, config."""

import json
import sys
import types

import pytest
import requests
from pydantic import BaseModel

from groundcheck.cli import main
from groundcheck.config import ConfigError, LLMConfig, load_llm_config
from groundcheck.llm import factory
from groundcheck.llm.base import LLMError, LLMRequest, PaidProviderRefused
from groundcheck.llm.cache import CachedProvider, DiskCache, cache_key
from groundcheck.llm.fake import FakeProvider
from groundcheck.llm.ollama import OllamaProvider
from groundcheck.llm.structured import (
    StructuredOutputError,
    extract_json_object,
    generate_structured,
)

DIGEST = "sha256:a80c4f17acd5"


class FakeHTTPResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeOllamaSession:
    """Answers like a local Ollama server; records every request."""

    def __init__(self, reply="Total net sales were $416,161 million.", models=None):
        self.reply = reply
        self.models = models if models is not None else [{"name": "llama3.2:3b", "digest": DIGEST}]
        self.requests = []

    def request(self, method, url, json=None, timeout=None):
        self.requests.append((method, url, json))
        if url.endswith("/api/version"):
            return FakeHTTPResponse(200, {"version": "0.12.3"})
        if url.endswith("/api/tags"):
            return FakeHTTPResponse(200, {"models": self.models})
        if url.endswith("/api/generate"):
            if json["model"] not in {m["name"] for m in self.models}:
                return FakeHTTPResponse(404, {"error": "model not found"})
            body = {
                "response": self.reply,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 40,
                "eval_count": 9,
                "total_duration": 1_500_000_000,
            }
            return FakeHTTPResponse(200, body)
        return FakeHTTPResponse(404, {})


# --- Ollama ---------------------------------------------------------------------------


def test_ollama_sends_deterministic_options_and_records_metadata():
    session = FakeOllamaSession()
    provider = OllamaProvider(session=session)
    response = provider.generate(
        LLMRequest(prompt="Q?", system="Be brief.", seed=42, max_tokens=100, json_mode=True)
    )
    assert response.text.startswith("Total net sales")
    method, url, payload = session.requests[-1]
    assert (method, url) == ("POST", "http://localhost:11434/api/generate")
    assert payload["options"] == {"temperature": 0.0, "seed": 42, "num_predict": 100}
    assert payload["stream"] is False and payload["format"] == "json"
    assert payload["system"] == "Be brief."
    meta = response.metadata
    assert (meta.provider, meta.model, meta.model_digest) == ("ollama", "llama3.2:3b", DIGEST)
    assert (meta.runtime, meta.runtime_version) == ("ollama", "0.12.3")
    assert (meta.temperature, meta.seed, meta.max_tokens) == (0.0, 42, 100)
    assert (meta.prompt_tokens, meta.completion_tokens, meta.duration_ms) == (40, 9, 1500)


def test_ollama_fetches_version_and_digest_once():
    session = FakeOllamaSession()
    provider = OllamaProvider(session=session)
    for _ in range(3):
        provider.generate(LLMRequest(prompt="Q?"))
    paths = [url.rsplit("/", 1)[1] for _, url, _ in session.requests]
    assert paths.count("version") == 1 and paths.count("tags") == 1
    assert paths.count("generate") == 3
    assert "format" not in session.requests[-1][2] and "system" not in session.requests[-1][2]
    assert "num_ctx" not in session.requests[-1][2]["options"]
    provider.generate(LLMRequest(prompt="Q?", context_window=8192))
    assert session.requests[-1][2]["options"]["num_ctx"] == 8192


def test_ollama_matches_latest_tag():
    session = FakeOllamaSession(models=[{"name": "mistral:latest", "digest": "sha256:1"}])
    assert OllamaProvider(model="mistral", session=session).identity().model_digest == "sha256:1"


def test_ollama_missing_model_says_how_to_pull():
    provider = OllamaProvider(model="qwen2.5:7b", session=FakeOllamaSession())
    with pytest.raises(LLMError, match="ollama pull qwen2.5:7b"):
        provider.generate(LLMRequest(prompt="Q?"))


def test_ollama_not_running_is_a_clean_error():
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaProvider(session=DownSession()).identity()


# --- Fake provider and cache ----------------------------------------------------------


def test_fake_provider_script_responder_and_calls():
    fake = FakeProvider(script=["one", "two"])
    assert [fake.generate(LLMRequest(prompt=p)).text for p in "ab"] == ["one", "two"]
    with pytest.raises(AssertionError):
        fake.generate(LLMRequest(prompt="c"))
    echo = FakeProvider(responder=lambda r: r.prompt.upper())
    assert echo.generate(LLMRequest(prompt="hi")).text == "HI"
    assert [c.prompt for c in fake.calls] == ["a", "b", "c"]
    assert echo.generate(LLMRequest(prompt="x")).metadata.model_digest == "fake-digest"


def test_cache_hits_on_identical_requests(tmp_path):
    inner = FakeProvider(responder=lambda r: f"answer to {r.prompt}")
    cached = CachedProvider(inner, DiskCache(tmp_path))
    first = cached.generate(LLMRequest(prompt="Q1"))
    second = cached.generate(LLMRequest(prompt="Q1"))
    assert first.text == second.text == "answer to Q1"
    assert not first.metadata.cached and second.metadata.cached
    assert len(inner.calls) == 1 and (cached.hits, cached.misses) == (1, 1)
    # A fresh process (new wrapper) reads the same files.
    again = CachedProvider(FakeProvider(script=[]), DiskCache(tmp_path))
    assert again.generate(LLMRequest(prompt="Q1")).text == "answer to Q1"


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "Q2"},
        {"system": "other"},
        {"temperature": 0.7},
        {"seed": 7},
        {"max_tokens": 10},
        {"json_mode": True},
        {"context_window": 8192},
    ],
)
def test_cache_key_covers_every_parameter(change):
    identity = FakeProvider().identity()
    base = LLMRequest(prompt="Q1")
    assert cache_key(identity, base) != cache_key(
        identity, LLMRequest(**{"prompt": "Q1", **change})
    )


def test_cache_key_covers_provider_model_and_digest():
    request = LLMRequest(prompt="Q")
    a = FakeProvider().identity()
    keys = {
        cache_key(a, request),
        cache_key(FakeProvider(model="other").identity(), request),
        cache_key(OllamaProvider(session=FakeOllamaSession()).identity(), request),
    }
    other_digest = FakeOllamaSession(models=[{"name": "llama3.2:3b", "digest": "sha256:new"}])
    keys.add(cache_key(OllamaProvider(session=other_digest).identity(), request))
    assert len(keys) == 4


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    inner = FakeProvider(script=["fresh"])
    key = cache_key(inner.identity(), LLMRequest(prompt="Q"))
    cache.path(key).parent.mkdir(parents=True)
    cache.path(key).write_text("{not json")
    assert CachedProvider(inner, cache).generate(LLMRequest(prompt="Q")).text == "fresh"
    assert json.loads(cache.path(key).read_text())["text"] == "fresh"


# --- Paid guard -------------------------------------------------------------------------


@pytest.fixture
def paid_provider(monkeypatch):
    monkeypatch.setitem(
        factory.PROVIDERS, "pricey", (lambda config, model, **kw: FakeProvider(paid=True), True)
    )


def test_paid_provider_is_refused_without_allow_paid(paid_provider):
    with pytest.raises(PaidProviderRefused, match="--allow-paid"):
        factory.make_provider(LLMConfig(), "pricey")


def test_paid_provider_runs_with_allow_paid(paid_provider, tmp_path):
    config = LLMConfig(cache_dir=str(tmp_path))
    provider = factory.make_provider(config, "pricey", allow_paid=True)
    assert provider.paid and provider.generate(LLMRequest(prompt="Q")).text == "fake response"


def test_guard_also_checks_the_instance(monkeypatch):
    # Registered as free by mistake, but the class says paid: still refused.
    monkeypatch.setitem(
        factory.PROVIDERS, "sneaky", (lambda config, model, **kw: FakeProvider(paid=True), False)
    )
    with pytest.raises(PaidProviderRefused):
        factory.make_provider(LLMConfig(), "sneaky")


def test_no_built_in_provider_is_paid():
    assert all(not paid for _, paid in factory.PROVIDERS.values())


def test_paid_refusal_is_a_clean_cli_error(paid_provider, capsys):
    assert main(["llm-info", "--provider", "pricey"]) == 2
    assert "--allow-paid" in capsys.readouterr().err


def test_unknown_provider_is_a_config_error():
    with pytest.raises(ConfigError, match="unknown LLM provider"):
        factory.make_provider(LLMConfig(), "nope")


# --- Structured output ------------------------------------------------------------------


class Claims(BaseModel):
    claims: list[str]


def test_extract_json_object_ignores_fences_and_prose():
    text = 'Sure! Here you go:\n```json\n{"claims": ["a {b}", "c\\"}"]}\n```\nAnything else?'
    assert json.loads(extract_json_object(text)) == {"claims": ["a {b}", 'c"}']}
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_structured_output_retries_with_the_error_then_succeeds():
    fake = FakeProvider(script=["not json", '{"claim": []}', '{"claims": ["x"]}'])
    value, response = generate_structured(fake, LLMRequest(prompt="Split it."), Claims)
    assert value.claims == ["x"] and response.metadata.attempts == 3
    assert all(c.json_mode for c in fake.calls)
    assert fake.calls[0].prompt == "Split it."
    assert "no JSON object found" in fake.calls[1].prompt
    assert "claims: Field required" in fake.calls[2].prompt


def test_structured_output_gives_up_after_max_attempts():
    fake = FakeProvider(script=["nope", "still nope"])
    with pytest.raises(StructuredOutputError) as info:
        generate_structured(fake, LLMRequest(prompt="Split it."), Claims, max_attempts=2)
    assert len(info.value.errors) == 2 and info.value.texts == ["nope", "still nope"]
    assert len(fake.calls) == 2


# --- Local Hugging Face provider (stand-in modules) -------------------------------------


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, i):
        return self.rows[i]


class FakeBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


@pytest.fixture
def fake_hf(monkeypatch):
    calls = {}

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name):
            calls["tokenizer"] = name
            return cls()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            calls["messages"] = messages
            return "|".join(m["content"] for m in messages)

        def __call__(self, text, return_tensors):
            return FakeBatch(input_ids=FakeTensor([[1, 2, 3]]))

        def decode(self, tokens, skip_special_tokens):
            return " generated " + "x" * len(tokens)

    class Model:
        config = types.SimpleNamespace(_commit_hash="abc123")

        @classmethod
        def from_pretrained(cls, name, torch_dtype):
            calls["model"] = (name, torch_dtype)
            return cls()

        def to(self, device):
            calls["device"] = device

        def eval(self):
            pass

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [[1, 2, 3, 9, 9]]

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.50.0"
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForCausalLM = Model
    torch = types.ModuleType("torch")
    torch.__version__ = "2.5.0"
    torch.float16, torch.float32 = "fp16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.no_grad = NoGrad
    torch.manual_seed = lambda seed: calls.setdefault("seeds", []).append(seed)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_hf_local_provider_greedy_with_metadata(fake_hf):
    from groundcheck.llm.hf_local import DEFAULT_HF_MODEL, HFLocalProvider

    provider = HFLocalProvider()
    response = provider.generate(LLMRequest(prompt="Q?", system="S", max_tokens=50))
    assert fake_hf["model"] == (DEFAULT_HF_MODEL, "fp16") and fake_hf["device"] == "cuda"
    assert fake_hf["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q?"},
    ]
    assert fake_hf["generate"]["do_sample"] is False
    assert fake_hf["generate"]["max_new_tokens"] == 50
    assert response.text == "generated xx"  # only the 2 new tokens are decoded
    meta = response.metadata
    assert (meta.provider, meta.model, meta.model_digest) == (
        "hf-local",
        DEFAULT_HF_MODEL,
        "abc123",
    )
    assert meta.runtime_version == "transformers 4.50.0, torch 2.5.0"
    assert meta.completion_tokens == 2 and meta.extra == {"device": "cuda"}


def test_hf_local_sampling_uses_the_seed(fake_hf):
    from groundcheck.llm.hf_local import HFLocalProvider

    HFLocalProvider(model="Qwen/Other").generate(LLMRequest(prompt="Q", temperature=0.5, seed=7))
    assert fake_hf["seeds"] == [7] and fake_hf["generate"]["do_sample"] is True


def test_hf_local_without_extra_is_a_clean_error(monkeypatch):
    from groundcheck.llm.hf_local import HFLocalProvider

    # A None entry in sys.modules makes `import x` raise ImportError without looking for
    # the real package, so this simulates the missing extra without ever importing torch
    # (a real torch import after faiss can abort the whole process on macOS: two OpenMP
    # runtimes in one process).
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ConfigError, match=r"\[hf\] extra"):
        HFLocalProvider()


# --- Config and CLI ---------------------------------------------------------------------


def test_llm_config_defaults_and_committed_file():
    default = LLMConfig()
    assert (default.provider, default.model, default.temperature) == ("ollama", "llama3.2:3b", 0)
    committed = load_llm_config()  # config/llm.yaml
    assert committed.model == "llama3.2:3b" and committed.seed == 42
    assert committed.cache_path.parts[-3:] == ("data", "cache", "llm")


def test_llm_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "llm.yaml"
    path.write_text("modle: llama3.2:3b\n")
    with pytest.raises(ConfigError, match="modle"):
        load_llm_config(path)
    with pytest.raises(ConfigError, match="not found"):
        load_llm_config(tmp_path / "missing.yaml")


def test_model_flag_overrides_config(tmp_path):
    config = LLMConfig(cache_dir=str(tmp_path))
    provider = factory.make_provider(config, "ollama", "qwen2.5:3b", session=FakeOllamaSession())
    assert provider.model == "qwen2.5:3b" and isinstance(provider, CachedProvider)
    assert factory.make_provider(config, "fake", use_cache=False).model == "fake-model"


def test_llm_info_command_with_fake_provider(capsys):
    assert main(["llm-info", "--provider", "fake"]) == 0
    out = capsys.readouterr().out
    assert "model digest:    fake-digest" in out and "reply:           fake response" in out


class DownSession:
    def request(self, *args, **kwargs):
        raise requests.ConnectionError("refused")


def test_llm_info_without_ollama_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.setattr("groundcheck.llm.ollama.requests.Session", DownSession)
    assert main(["llm-info", "--provider", "ollama"]) == 2
    assert "cannot reach Ollama" in capsys.readouterr().err
