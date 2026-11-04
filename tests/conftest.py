"""Test-wide safety nets.

Every test runs with:
- no network: any attempt to resolve a hostname or open a socket connection fails
  immediately, so no test can ever reach sec.gov (or anything else) by accident;
- no SEC_USER_AGENT inherited from the developer's shell, and the current directory
  set to an empty temp folder, so neither a real env var nor a real `.env` leaks in.
  Tests that need a user agent set one explicitly with monkeypatch.setenv.
"""

import socket

import pytest


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to use the real network."""


def _blocked(*args, **kwargs):
    raise NetworkBlockedError(f"tests must not use the network (attempted: {args!r})")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path_factory):
    # setenv first so monkeypatch records the original state; then a value that a test
    # loads from a .env file (load_dotenv writes straight to os.environ) is undone too.
    monkeypatch.setenv("SEC_USER_AGENT", "placeholder")
    monkeypatch.delenv("SEC_USER_AGENT")
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


# --- A tiny indexed corpus for generation and labeling tests ---------------------------

TINY_CHUNKS = {
    "EXCO-7-p0000": (
        "7",
        "prose",
        "Total net sales increased 6% to $12,345 million in fiscal 2025 compared with "
        "fiscal 2024. The increase was partially offset by lower services revenue.",
    ),
    "EXCO-7-t0000": (
        "7",
        "table",
        "[TABLE]\n | 2025 | 2024\nTotal net sales | $12,345 | $11,646\n"
        "Operating income | $3,210 | $2,987\n[/TABLE]",
    ),
    "EXCO-1A-p0000": (
        "1A",
        "prose",
        "The Company faces intense competition and may not be able to maintain its "
        "market share or its gross margin.",
    ),
    "EXCO-8-p0000": (
        "8",
        "prose",
        "During fiscal 2025 the Company repurchased 40 million shares of its common "
        "stock for $9.1 billion.",
    ),
}


@pytest.fixture
def tiny_corpus(tmp_path):
    """A labeled company EXCO with four chunks, a retrieval config whose variants use
    only offline fakes, a hashing-embedder FAISS index for its default variant, a
    companies config and both question files. Returns a dict of paths."""
    from groundcheck.chunking.base import Chunk, write_chunks
    from groundcheck.retrieval.embedder import HashingEmbedder
    from groundcheck.retrieval.retriever import build_variant_index
    from groundcheck.retrieval.variants import load_retrieval_config

    chunks = [
        Chunk(
            chunk_id=cid,
            ticker="EXCO",
            accession="0000000001-25-000001",
            item=item,
            section_title="T",
            kind=kind,
            index=0,
            text=text,
            word_count=len(text.split()),
            chunker="fixed-p250-o50-t120",
        )
        for cid, (item, kind, text) in TINY_CHUNKS.items()
    ]
    chunks_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    write_chunks(chunks, chunks_dir / "EXCO.jsonl")
    retrieval_config = tmp_path / "retrieval.yaml"
    retrieval_config.write_text(
        "default: dense\npool: 4\nvariants:\n"
        "  dense: {embedder: hashing}\n"
        "  hybrid-rerank: {embedder: hashing, headers: true, bm25: true, rerank: overlap}\n"
        "  bm25: {embedder: null, bm25: true}\n"
    )
    spec = load_retrieval_config(retrieval_config).variants["dense"]
    build_variant_index("dense", spec, "EXCO", HashingEmbedder(), chunks_dir, index_dir)
    config = tmp_path / "companies.yaml"
    config.write_text(
        "companies:\n"
        "  - {ticker: EXCO, name: Example Corp, cik: 1, role: labeled}\n"
        "  - {ticker: OTHR, name: Other Corp, cik: 2, role: unlabeled}\n"
    )
    questions = tmp_path / "candidate_questions.yaml"
    questions.write_text(
        "version: 1\nquestions:\n"
        "  - {id: EXCO-G01, ticker: EXCO, qtype: numeric_lookup,"
        " question: 'What were total net sales in fiscal 2025?'}\n"
        "  - {id: EXCO-G02, ticker: EXCO, qtype: yes_no,"
        " question: 'Did the company repurchase shares in fiscal 2025?'}\n"
        "  - {id: EXCO-G03, ticker: EXCO, qtype: table, question: ''}\n"
    )
    retrieval = tmp_path / "retrieval_questions.yaml"
    retrieval.write_text(
        "version: 1\nquestions:\n"
        "  - {id: EXCO-01, ticker: EXCO, qtype: table,"
        " question: 'What was operating income in 2025?', gold_evidence: []}\n"
        "  - {id: EXCO-02, ticker: EXCO, qtype: temporal, question: ''}\n"
        "  - {id: OTHR-01, ticker: OTHR, qtype: table, question: 'Ignored?'}\n"
    )
    llm_config = tmp_path / "llm.yaml"
    llm_config.write_text(f"cache_dir: {tmp_path / 'cache'}\ntop_k: 3\n")
    corpus = {
        "root": tmp_path,
        "chunks": chunks_dir,
        "index": index_dir,
        "config": config,
        "questions": questions,
        "retrieval": retrieval,
        "llm_config": llm_config,
        "retrieval_config": retrieval_config,
        "out": tmp_path / "candidates",
    }

    def generate_args(*extra):
        """Arguments for scripts/generate_candidates.py against this corpus."""
        return [
            "--config", str(config),
            "--questions", str(questions),
            "--retrieval-questions", str(retrieval),
            "--chunks-dir", str(chunks_dir),
            "--index-dir", str(index_dir),
            "--llm-config", str(llm_config),
            "--out-dir", str(corpus["out"]),
            "--retrieval-config", str(retrieval_config),
            *extra,
        ]  # fmt: skip

    corpus["generate_args"] = generate_args
    return corpus
