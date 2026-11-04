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

### Retrieval (fixed-size chunks, one FAISS index per company)

On the 30 questions in `data/eval/retrieval_questions.yaml`. They were drafted by an AI
from the filings and are not yet reviewed, so differences under about 0.1 are noise.

| Variant | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| `dense` (bge-small, baseline) | 0.23 | 0.50 | 0.62 |
| `bm25` (BM25 alone, diagnostic) | 0.33 | 0.62 | 0.65 |
| other variants | TBD | TBD | TBD |

Variants and how to compare them: [docs/DESIGN.md](docs/DESIGN.md), "Retrieval variants".

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

## Quickstart on a Mac

Tested target: an Apple-silicon MacBook (8 GB RAM is enough for the default
`llama3.2:3b`). Everything runs locally; nothing costs money.

**1. Install the package** (Python 3.11 or newer, e.g. `brew install python@3.12`):

```bash
git clone https://github.com/krupal2241/GroundCheck.git && cd GroundCheck
```

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e ".[dev,embed]"
```

Add `rerank` (`pip install -e ".[dev,embed,rerank]"`) only to try the cross-encoder
reranker variant; the default retriever doesn't need it.

**2. Set your SEC user agent** (SEC's fair-access rules require a name and email):

```bash
cp .env.example .env    # then edit .env: SEC_USER_AGENT="Your Name you@example.com"
```

**3. Install Ollama and pull the model.** Install the Ollama app from
[ollama.com/download](https://ollama.com/download) (or `brew install ollama` and run
`ollama serve` in another terminal), then:

```bash
ollama pull llama3.2:3b
```

```bash
groundcheck llm-info
```

It prints the provider, model, model digest and Ollama version, and a one-word reply.

**4. Download, parse, chunk and index** (filings are pinned in `config/companies.yaml`):

```bash
groundcheck download
```

```bash
groundcheck parse
```

```bash
groundcheck chunk
```

```bash
groundcheck index
```

`parse` ends with "Parsed 8 filing(s) ...; N warning(s)"; JPM and XOM show NOTE lines
for their recovered Items 7 and 8. `index` downloads bge-small once and builds one
FAISS index per company for the default retrieval variant (`config/retrieval.yaml`),
under `data/index/<embedding>/`.

**Optional: compare retrieval variants** (dense, contextual headers, hybrid BM25,
reranker, bge-base). Build every variant's index, then print one comparison table:

```bash
groundcheck index --variant all --only AAPL MSFT AMZN WMT XOM JNJ
```

```bash
groundcheck eval-retrieval --compare
```

It also writes `data/eval/results/compare.json` and `compare.md` and prints the
recommended default (best recall@5, or the fastest variant within 0.05 of it). To adopt
it, set `default:` in `config/retrieval.yaml`, then run `groundcheck index` for it.

**5. Write questions** in `data/questions/candidate_questions.yaml` (ten typed slots
per labeled company) and/or `data/eval/retrieval_questions.yaml`. Questions with text
in either file are used; you don't write a question twice.

**6. Generate candidates** (rehearse first with the fake model, then for real):

```bash
python scripts/generate_candidates.py --dry-run --limit 3
```

```bash
python scripts/generate_candidates.py --batch-size 5 --sleep-between-batches 60
```

Output goes to `data/candidates/<TICKER>.jsonl` (dry runs to `data/cache/dry-run/`).
Interrupt it any time and run the same command again: it resumes where it stopped.
`--company AAPL` limits to one company, `--model` picks another Ollama model. On a
free Colab GPU, use `notebooks/generate_candidates_colab.ipynb` instead.

**7. Label the claims** (see [docs/labeling_guide.md](docs/labeling_guide.md)):

```bash
groundcheck label --only-unlabeled
```

```bash
groundcheck label-stats
```

```bash
groundcheck label-check
```

```bash
groundcheck export-labels
```

Labels autosave to `data/labels/labels.jsonl`; `export-labels` writes only confirmed
labels to `data/eval/gold_labels.jsonl`.

## Repository layout

```
src/groundcheck/   package: edgar, parsing, chunking, retrieval, llm, generation, labeling
scripts/           standalone runs (e.g. candidate generation)
notebooks/         Colab notebook for candidate generation on a free GPU
config/            companies (pinned accession numbers) and pipeline settings
data/              committed: chunks, questions, candidates, labels (raw filings are not)
docs/              PLAN.md, DESIGN.md, labeling_guide.md
tests/             pytest suite (offline; no network, no models)
```

## Docs

- [docs/PLAN.md](docs/PLAN.md): two-week plan, open questions and risks
- [docs/DESIGN.md](docs/DESIGN.md): what each module does, why, and the alternatives considered
- [docs/labeling_guide.md](docs/labeling_guide.md): how to label a claim, with worked edge cases

## Data and fair access

Filings come from SEC EDGAR's official APIs. Requests send a `SEC_USER_AGENT` header
(read from your environment), are rate-limited to at most 5 per second, and are cached on
disk so no file is downloaded twice.
