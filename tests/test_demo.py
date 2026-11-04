"""The demo: pipeline over the tiny corpus with fakes, HTML highlighting, and the Gradio
app over a stand-in `gradio` module (the real one is never imported)."""

import sys
import types

import pytest

from groundcheck.cli import main
from groundcheck.config import ConfigError, LLMConfig
from groundcheck.demo.app import DemoPipeline, build_app, highlight, render_html
from groundcheck.generation.pipeline import RetrievalInfo
from groundcheck.generation.prompts import fake_responder
from groundcheck.llm.fake import FakeProvider
from groundcheck.retrieval.embedder import HashingEmbedder
from groundcheck.retrieval.index import CompanyIndex
from groundcheck.verify.fake import FakeVerifier


@pytest.fixture
def pipeline(tiny_corpus):
    embedder = HashingEmbedder()
    index = CompanyIndex.load("EXCO", tiny_corpus["chunks"], tiny_corpus["index"])
    built = []

    def build(name):
        built.append(name)
        return FakeVerifier()

    demo = DemoPipeline(
        provider=FakeProvider(responder=fake_responder),
        retrieve=lambda ticker, question, k: index.search(embedder, question, k),
        retrieval_info=lambda t: RetrievalInfo(
            embedder=embedder.name, chunker="c", top_k=3, chunks_sha256="x"
        ),
        llm_config=LLMConfig(top_k=3),
        build_verifier=build,
    )
    demo.built = built
    return demo


def test_pipeline_answers_splits_and_verifies(pipeline):
    result = pipeline.ask("What were total net sales in fiscal 2025?", "EXCO", "fake")
    assert result.candidate.answer.startswith("According to the filing, ")
    assert result.claims and len(result.candidate.retrieved) == 3
    first = result.claims[0].verification
    assert first.label == "supported" and first.evidence is not None
    pipeline.ask("Did the company repurchase shares?", "EXCO", "fake")
    assert pipeline.built == ["fake"]  # built once, then reused


def test_render_html_highlights_the_span_and_escapes(pipeline):
    result = pipeline.ask("What were total net sales in fiscal 2025?", "EXCO", "fake")
    page = render_html(result)
    span = result.claims[0].verification.evidence
    assert f"<mark>{span.text}</mark>" in page.replace("&#x27;", "'")
    assert "supported</span>" in page and "Retrieved passages" in page
    assert highlight("a <b> c", 2, 5) == "a <mark>&lt;b&gt;</mark> c"


def test_empty_question_is_refused(pipeline):
    with pytest.raises(ConfigError, match="type a question"):
        pipeline.ask("   ", "EXCO", "fake")


def fake_gradio(recorder):
    gr = types.ModuleType("gradio")

    class Component:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            recorder.setdefault("components", []).append((type(self).__name__, kwargs))

        def click(self, fn, inputs, outputs):
            recorder["click"] = fn

        def submit(self, fn, inputs, outputs):
            recorder["submit"] = fn

    class Context(Component):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def launch(self, **kwargs):
            recorder["launch"] = kwargs

    for name in ("Markdown", "Dropdown", "Textbox", "Button", "HTML"):
        setattr(gr, name, type(name, (Component,), {}))
    gr.Blocks = type("Blocks", (Context,), {})
    gr.Row = type("Row", (Context,), {})
    return gr


def test_build_app_wires_the_pipeline(pipeline, monkeypatch):
    recorder = {}
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio(recorder))
    build_app(pipeline, ["EXCO"], ["fake", "hhem"])
    dropdowns = [kw for name, kw in recorder["components"] if name == "Dropdown"]
    assert dropdowns[1]["choices"] == ["fake", "hhem"] and dropdowns[0]["value"] == "EXCO"
    page = recorder["click"]("What were total net sales?", "EXCO", "fake")
    assert "<h3>Answer</h3>" in page
    assert "Error:" in recorder["submit"]("", "EXCO", "fake")  # shown, not raised


def test_demo_without_gradio_is_a_clean_error(pipeline, monkeypatch):
    monkeypatch.setitem(sys.modules, "gradio", None)
    with pytest.raises(ConfigError, match=r"\[demo\] extra"):
        build_app(pipeline, ["EXCO"], ["fake"])


def test_demo_command_dry_run_launches_locally(tiny_corpus, monkeypatch, capsys):
    recorder = {}
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio(recorder))
    args = [
        "demo", "--dry-run", "--embedder", "hashing",
        "--config", str(tiny_corpus["config"]),
        "--chunks-dir", str(tiny_corpus["chunks"]),
        "--index-dir", str(tiny_corpus["index"]),
        "--llm-config", str(tiny_corpus["llm_config"]),
    ]  # fmt: skip
    assert main(args) == 0
    assert recorder["launch"] == {"server_name": "127.0.0.1", "server_port": 7860, "share": False}
    out = capsys.readouterr().out
    assert "companies EXCO" in out and "verifiers fake" in out
    page = recorder["click"]("What were total net sales in fiscal 2025?", "EXCO", "fake")
    assert "<mark>" in page


def test_demo_without_indexes_is_a_clean_error(tmp_path, capsys):
    config = tmp_path / "companies.yaml"
    config.write_text("companies:\n  - {ticker: EXCO, name: E, cik: 1, role: labeled}\n")
    args = ["demo", "--config", str(config), "--index-dir", str(tmp_path / "none")]
    assert main(args) == 2
    assert "run `groundcheck index`" in capsys.readouterr().err
