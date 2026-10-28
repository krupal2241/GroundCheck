# GroundCheck

Claim-level hallucination verification for retrieval-augmented generation (RAG).

Given an LLM answer and the source passages it was generated from, GroundCheck splits
the answer into atomic claims and labels each one **supported**, **contradicted**, or
**unsupported**, citing the evidence sentence. The corpus is SEC 10-K filings: long
documents with messy tables and lots of numbers, which is where RAG answers tend to go
wrong.

> **Status:** Weeks 1–2 in progress. Building the RAG baseline and the labeling tooling.
> No verifier model yet. See [docs/PLAN.md](docs/PLAN.md).

## Project goals

1. A RAG baseline over 10-K filings (EDGAR download → parse → chunk → retrieve → answer).
2. A hand-labeled, claim-level test set (every label confirmed by a human).
3. A fine-tuned small verifier, benchmarked against a prompted frontier-model judge and
   open verifiers (MiniCheck, HHEM) on accuracy, latency, and cost per 1,000 checks.

The project runs on a $0 budget: generation uses local models through Ollama, and any
paid API call is refused unless you explicitly pass `--allow-paid`.

## Results

Every number below stays **TBD** until a real run produces it. Nothing here is estimated.

### Retrieval (baseline: fixed-size chunks, bge-small, FAISS)

| Metric | Value |
|---|---|
| recall@1 | TBD |
| recall@5 | TBD |
| recall@10 | TBD |

### Labeled test set

| | Value |
|---|---|
| Items | TBD |
| Claims | TBD |
| Label distribution (S / C / U) | TBD |

### Verifier benchmark

| Verifier | Accuracy | Latency (p50) | Cost / 1,000 checks |
|---|---|---|---|
| Fine-tuned small verifier | TBD | TBD | TBD |
| Prompted frontier judge | TBD | TBD | TBD |
| MiniCheck | TBD | TBD | TBD |
| HHEM | TBD | TBD | TBD |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Full macOS steps (Homebrew, Ollama, SEC user agent) will be in `docs/running_locally.md`.

## Repository layout

```
src/groundcheck/   package: edgar, parsing, chunking, retrieval, llm, generation, labeling
scripts/           standalone runs (e.g. candidate generation)
config/            companies (pinned accession numbers) and pipeline settings
data/              committed: chunks, questions, candidates, labels (raw filings are not)
docs/              PLAN.md, DESIGN.md, labeling_guide.md
tests/             pytest suite (offline; no network, no models)
```

## Docs

- [docs/PLAN.md](docs/PLAN.md): two-week plan, open questions and risks
- [docs/DESIGN.md](docs/DESIGN.md): what each module does, why, and the alternatives considered

## Data and fair access

Filings come from SEC EDGAR's official APIs. Requests send a `SEC_USER_AGENT` header
(read from your environment), are rate-limited to at most 5 per second, and are cached on
disk so no file is downloaded twice.
