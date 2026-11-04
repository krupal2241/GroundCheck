"""A small local demo (`groundcheck demo`): ask, retrieve, answer, split, verify, highlight.

Flow per question, all on your machine:

1. retrieve the top-k chunks from the chosen company's FAISS index;
2. answer from only those chunks with the local model (Ollama by default) and split
   the answer into claims, with exactly the prompts and code of candidate generation
   (`generation.pipeline.Generator`);
3. verify each claim against the retrieved chunks with the verifier chosen in the UI;
4. show every claim with its label, P(supported) and latency, and the cited chunk with
   the evidence span highlighted.

The pipeline (`DemoPipeline.ask`) and the HTML rendering are plain Python, tested
without Gradio. Gradio is in the `[demo]` extra and imported only by `build_app`.
Verifiers are built the first time they are chosen and then kept, so switching back
is instant; on an 8 GB Mac, keep the list of choices short (`--verifiers`).
The server listens on 127.0.0.1 only and never creates a public share link.
"""

import html
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from groundcheck.config import ConfigError, LLMConfig
from groundcheck.generation.pipeline import CandidateRecord, Generator, RetrievalInfo
from groundcheck.generation.questions import GenerationQuestion
from groundcheck.llm.base import LLMProvider
from groundcheck.retrieval.evaluate import normalize_for_match
from groundcheck.verify.base import EvidenceChunk, Verification, Verifier

LABEL_COLORS = {
    "supported": "#1a7f37",
    "contradicted": "#cf222e",
    "unsupported": "#9a6700",
}


@dataclass
class ClaimResult:
    claim: str
    verification: Verification


@dataclass
class DemoResult:
    candidate: CandidateRecord
    verifier: str
    claims: list[ClaimResult]


@dataclass
class DemoPipeline:
    provider: LLMProvider
    retrieve: Callable  # (ticker, question, k) -> list[Hit]
    retrieval_info: Callable[[str], RetrievalInfo]
    llm_config: LLMConfig
    build_verifier: Callable[[str], Verifier]
    _verifiers: dict[str, Verifier] = field(default_factory=dict)

    def verifier(self, name: str) -> Verifier:
        if name not in self._verifiers:
            self._verifiers[name] = self.build_verifier(name)
        return self._verifiers[name]

    def ask(self, question: str, ticker: str, verifier_name: str) -> DemoResult:
        question = " ".join(question.split())
        if not question:
            raise ConfigError("type a question first")
        generator = Generator(self.provider, self.retrieve, self.retrieval_info, self.llm_config)
        candidate = generator.generate(
            GenerationQuestion(
                id="demo", ticker=ticker, qtype="demo", question=question,
                source="candidate_questions",
            )
        )  # fmt: skip
        chunks = [EvidenceChunk(c.chunk_id, c.text) for c in candidate.retrieved]
        verifier = self.verifier(verifier_name)
        claims = [ClaimResult(c.text, verifier.verify(c.text, chunks)) for c in candidate.claims]
        return DemoResult(candidate=candidate, verifier=verifier_name, claims=claims)


def highlight(chunk_text: str, start: int, end: int) -> str:
    """The normalized chunk text, HTML-escaped, with [start:end] wrapped in <mark>."""
    text = normalize_for_match(chunk_text)
    return (
        html.escape(text[:start])
        + "<mark>"
        + html.escape(text[start:end])
        + "</mark>"
        + html.escape(text[end:])
    )


def render_html(result: DemoResult) -> str:
    candidate = result.candidate
    chunks = {c.chunk_id: c for c in candidate.retrieved}
    parts = [
        f"<h3>Answer</h3><p>{html.escape(candidate.answer)}</p>",
        f"<p><small>Model: {html.escape(candidate.answer_model.model)}; verifier: "
        f"{html.escape(result.verifier)}</small></p>",
        f"<h3>Claims ({len(result.claims)})</h3>",
    ]
    if not result.claims:
        parts.append("<p>The answer makes no factual claim (nothing to verify).</p>")
    for number, item in enumerate(result.claims, start=1):
        v = item.verification
        color = LABEL_COLORS[v.label]
        parts.append(
            "<div style='margin:0.6em 0;padding:0.5em;border-left:4px solid "
            f"{color}'><b>{number}. {html.escape(item.claim)}</b><br>"
            f"<span style='color:{color};font-weight:bold'>{v.label}</span>"
            f" &middot; P(supported) {v.score:.2f} &middot; {v.latency_ms:.0f} ms"
        )
        if v.evidence is not None and v.evidence.chunk_id in chunks:
            chunk = chunks[v.evidence.chunk_id]
            body = highlight(chunk.text, v.evidence.start, v.evidence.end)
            parts.append(
                f"<details open><summary>Evidence: passage {chunk.rank} "
                f"({html.escape(chunk.chunk_id)})</summary>"
                f"<div style='font-size:0.9em'>{body}</div></details>"
            )
        else:
            parts.append("<br><i>No evidence span cited.</i>")
        parts.append("</div>")
    parts.append("<h3>Retrieved passages</h3>")
    for chunk in candidate.retrieved:
        parts.append(
            f"<details><summary>{chunk.rank}. {html.escape(chunk.chunk_id)} "
            f"(Item {html.escape(chunk.item)}, {chunk.kind}, score {chunk.score:.3f})</summary>"
            f"<pre style='white-space:pre-wrap'>{html.escape(chunk.text)}</pre></details>"
        )
    return "\n".join(parts)


def build_app(pipeline: DemoPipeline, tickers: list[str], verifiers: list[str]) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ConfigError('the demo needs the [demo] extra: pip install -e ".[demo]"') from exc

    def answer(question: str, ticker: str, verifier: str) -> str:
        try:
            return render_html(pipeline.ask(question, ticker, verifier))
        except Exception as exc:  # shown in the page instead of a stack trace
            return f"<p style='color:#cf222e'><b>Error:</b> {html.escape(str(exc))}</p>"

    with gr.Blocks(title="GroundCheck demo") as app:
        gr.Markdown(
            "# GroundCheck\nAsk about a company's 10-K. The local model answers from the "
            "retrieved passages; each claim is then checked by the chosen verifier."
        )
        with gr.Row():
            ticker = gr.Dropdown(choices=tickers, value=tickers[0], label="Company")
            verifier = gr.Dropdown(choices=verifiers, value=verifiers[0], label="Verifier")
        question = gr.Textbox(label="Question", placeholder="What were total net sales?")
        button = gr.Button("Answer and verify", variant="primary")
        output = gr.HTML()
        button.click(answer, inputs=[question, ticker, verifier], outputs=output)
        question.submit(answer, inputs=[question, ticker, verifier], outputs=output)
    return app
