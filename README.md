# GroundCheck

Claim-level hallucination verification for retrieval-augmented generation (RAG).

Given an LLM answer and the source passages it was generated from, GroundCheck splits
the answer into atomic claims and labels each one **supported**, **contradicted**, or
**unsupported**, citing the evidence sentence. The corpus is SEC 10-K filings: long
documents with messy tables and lots of numbers, which is where RAG answers tend to go
wrong.

> **Status:** the RAG baseline, labeling tooling, verifiers (prompted local LLM, MiniCheck,
> HHEM, fine-tuned), the evaluation and error-report commands, and a local demo are built.
> No benchmark numbers yet: they come from real runs on the labeled set. See
> [docs/PLAN.md](docs/PLAN.md).

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
| Prompted local LLM judge (llama3.2:3b) | TBD | TBD | TBD |
| Prompted frontier judge | not run ($0 budget); cost estimate only | — | TBD (estimate) |
| MiniCheck | TBD | TBD | TBD |
| HHEM | TBD | TBD | TBD |

`groundcheck eval-verifiers` writes the real table to `results/verifier_eval.md`; the
rows above stay TBD until it has run on the confirmed gold labels.

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
FAISS index per company under `data/index/`.

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

## Verifiers: evaluate, analyze errors, fine-tune

**Evaluate** verifiers on the confirmed gold labels. MiniCheck and HHEM need the
`[verifiers]` extra (PyTorch + transformers; they download their weights from Hugging Face
once and run on Apple MPS when available). Run one model at a time on an 8 GB Mac; results
accumulate in one table:

```bash
pip install -e ".[verifiers]"
```

```bash
groundcheck eval-verifiers --verifier prompted
```

```bash
groundcheck eval-verifiers --verifier hhem
```

```bash
groundcheck eval-verifiers --verifier minicheck
```

Each run updates `results/verifier_eval.json` and `results/verifier_eval.md` (the table to
paste above) and writes per-claim predictions to `results/predictions/`. Cost is $0 for
every local model; to see an **estimate** for a hosted judge, type its prices into
`config/prices.yaml` (no API is ever called). Then list every mistake by edge case:

```bash
groundcheck error-report
```

**Fine-tune** a small verifier (deberta-v3-small) on synthetic data built from the chunks
(the gold set is excluded and splits are by company): open
`notebooks/finetune_verifier_colab.ipynb` on a free Colab T4, download the model zip,
unzip it into `models/`, then:

```bash
groundcheck eval-verifiers --verifier finetuned --finetuned-path models/verifier-deberta-v3-small
```

## Demo

A local web page: pick a company, type a question, and see the local model's answer
split into claims, each labeled by the verifier you choose, with the evidence span
highlighted in its passage. Everything runs on your machine (127.0.0.1, no share link).

```bash
pip install -e ".[demo,verifiers]"
```

```bash
groundcheck demo
```

Open http://127.0.0.1:7860. It needs the indexes from step 4 and Ollama running. Offer
fewer verifiers to save memory (`--verifiers prompted hhem`), add your fine-tuned model
with `--finetuned-path models/verifier-deberta-v3-small`, or rehearse without Ollama or
model downloads using `groundcheck demo --dry-run` (fake answers, fake verifier).

## Repository layout

```
src/groundcheck/   package: edgar, parsing, chunking, retrieval, llm, generation, labeling,
                   verify (verifiers, evaluation, error report), training, demo
scripts/           standalone runs (candidate generation, verifier fine-tuning)
notebooks/         Colab notebooks: candidate generation, verifier fine-tuning
config/            companies (pinned accession numbers), LLM, training splits, prices
data/              committed: chunks, questions, candidates, labels (raw filings are not)
results/           verifier benchmark tables, predictions and error report
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
