# GroundCheck Plan: Weeks 1–2

**Scope:** a RAG baseline over SEC 10-K filings, plus tooling for a hand-labeled,
claim-level test set. No GPU, no fine-tuning, **$0 spend**.

## Ground rules

- **No fabricated gold labels.** Claims or questions may be drafted by a model or by
  Claude, but a label only counts once a human has confirmed it. This is enforced in
  code: export keeps only `confirmed=True` claims.
- **No made-up results.** README numbers stay "TBD" until a real run produces them.
- **SEC fair access.** A `SEC_USER_AGENT` header ("Name email") is sent on every
  request, requests stay at ≤5 per second, everything is cached on disk, and no file is
  ever re-downloaded.
- **$0 budget.** Local models run through Ollama. Any provider marked paid is refused
  unless `--allow-paid` is passed.
- **Secrets** live only in env vars or a gitignored `.env`.
- Small, focused commits. Tests are required for the parser, the chunker, and label
  schema validation.

## Decisions

| Topic | Decision |
|---|---|
| Companies | AAPL, MSFT, AMZN, JPM, XOM, JNJ, WMT, BRK-B in `config/companies.yaml` |
| Filings | Each company's latest 10-K, with **pinned accession numbers**. Fiscal years are not aligned; `fiscal_year_end` is recorded per company. |
| Labeled set | Drawn from AAPL, MSFT, AMZN, WMT, XOM, JNJ: **~10 candidate questions per company (~60 items)** |
| BRK-B | Parser stress test only, not in the labeled set |
| JPM | Indexed and parsed (`role: unlabeled`); not in the labeled set |
| Missing Item section | Logged as a WARNING (never silently skipped) |
| Vector store | FAISS `IndexFlatIP` (exact cosine), **one index per company** |
| Embeddings | `BAAI/bge-small-en-v1.5`, on CPU or Apple MPS |
| LLM | Ollama (local) first. No paid adapter in Weeks 1–2. |
| Default model | `llama3.2:3b`: the MacBook Air M3 has 8 GB RAM, so a 3B model is what fits comfortably. |
| Rounding rule | Supported only if the source value rounds correctly to the claim's own precision. Unit or scale mistakes count as **contradicted**. |
| Workflow | One pull request per day of work; you review and merge it yourself |
| Committed data | `data/chunks/*.jsonl`, question files, candidate JSONL, labels |

## Pipeline

```
resolve-filings → download → parse → chunk → index (per company) → retrieve
                                                             ↓
                         generate_candidates.py (Ollama) → candidates JSONL
                                                             ↓
                                       label CLI → labels JSONL → gold export
```

### Modules

1. **edgar.** `requests` plus a small rate limiter (≤5 req/s). `resolve-filings` reads
   `data.sec.gov/submissions/CIK##########.json` and prints each company's latest 10-K
   accession number, filing date, primary document and fiscal year end, which you then
   pin in config. `download` fetches from `/Archives/edgar/data/...`, skips files that
   already exist, writes atomically, and records URL, sha256 and timestamp in
   `meta.json`. It refuses to run while any accession number is unpinned.
2. **parsing.** BeautifulSoup + lxml. Filing HTML becomes clean text; tables become
   pipe-delimited rows inside `[TABLE]…[/TABLE]`, with colspan expanded, `$`/`%` cells
   merged, and `(1,234)` kept verbatim. The text is split into Item sections, handling
   the duplicate headings from the table of contents. Expected items are 1, 1A, 7 and 8.
3. **chunking.** A `Chunker` interface. The baseline is `FixedSizeChunker`: prose is
   chunked at 250 words with 50 words of overlap. Table blocks are chunked separately,
   with a smaller size, split only on row boundaries, and each piece repeats the header
   row. This keeps every chunk under bge-small's 512-token limit. Chunk IDs are
   deterministic, and chunks never cross sections. A later `TableAwareChunker` plugs in
   behind the same interface for comparison.
4. **retrieval.** An `Embedder` interface (bge-small in practice, a hashing fake in
   tests). There is one FAISS index per company, so every question gets its full `top_k`
   from the right filing. Recall@{1,3,5,10} and section-hit@k are measured against
   human-written gold evidence strings.
5. **llm.** An `LLMProvider` interface with `OllamaProvider` (plain HTTP to the local
   server, temperature 0, fixed seed, and model digest and runtime version recorded) and
   `FakeProvider` for tests. The paid-call guard sits in the provider factory. Responses
   are cached on disk by request hash.
6. **generation.** `scripts/generate_candidates.py` is a standalone script that runs on
   your Mac or on Colab. It writes JSONL: question, retrieved passages, answer, model
   name, digest and version, prompt version, and retrieval config. It's resumable and
   batched (`--batch-size`, `--limit`, `--sleep-between-batches`), so the fanless
   MacBook Air can cool down between batches.
7. **labeling.** A pydantic schema plus a terminal CLI that labels claim by claim
   (supported / contradicted / unsupported), with evidence span offsets checked against
   the passage text. The optional `--draft-claims` produces unconfirmed claim splits from
   the local model.

### Candidate question mix

The ~10 questions per labeled company are a deliberate mix of types, each tagged with
`qtype`:

| qtype | Example shape |
|---|---|
| `numeric_lookup` | "What was total net sales in fiscal year X?" |
| `table` | A value that only appears in a financial table |
| `multi_step` | A difference, ratio or growth rate between two reported values |
| `yes_no` | "Did the company repurchase shares during the year?" |
| `temporal` | A comparison across years or periods, or fiscal vs calendar year |
| `unanswerable` | A plausible question the filing does not answer |

Claude may draft question wording for you to edit. **You** write every gold evidence
string.

## Day-by-day

| Day | Work | Handoff to you |
|---|---|---|
| 1 | Scaffold: pyproject, ruff, pytest, CI, README skeleton, this plan, DESIGN.md | — |
| 2 | Config models, EDGAR client, rate limiter, `resolve-filings`, `download` (tests use mocked HTTP) | — |
| 3 | HTML → text, tables → rows, synthetic fixtures, tests | — |
| 4 | Item section splitter (TOC handling, missing-section warnings), tests | Pin accession numbers; download + parse locally; send warnings |
| 5 | Parser fixes from the real filings; fixed-size chunker with table handling; tests | — |
| 6 | Embedder, per-company FAISS indexes, `retrieve` command | — |
| 7 | Recall@k and the question file format | Write ~30 retrieval questions with gold strings; run recall |
| 8 | LLM interface, Ollama adapter, paid-call guard, cache | — |
| 9 | `generate_candidates.py`, Colab notebook, `running_locally.md`; label schema + tests | — |
| 10 | Labeling CLI, gold export, `labeling_guide.md` | Generate the ~60 candidates, commit them, label a ~10-item pilot |

**If we fall behind,** cut `--draft-claims` and the rich terminal UI first (plain
`input()`/`print` is enough).

## Open questions

- [ ] Accession numbers and fiscal year ends: to be filled from `resolve-filings` (Day 4).
- [ ] Retrieval questions and gold evidence strings (Day 7).

## Resolved

- [x] RAM: 8 GB, so the default model is `llama3.2:3b` (Day 2).

## Risks

- **The CIKs in `config/companies.yaml` were typed in by hand**, not looked up here
  (sec.gov is unreachable from the sandbox). `resolve-filings` prints SEC's name for
  each CIK; check it before pinning.

- **This sandbox can't reach sec.gov or huggingface.co** (the proxy rejects the
  connection). Downloading, real parsing checks, embedding and generation all run on
  your machine. In the sandbox the parser is tested only against synthetic fixtures,
  which may miss real-world quirks.
- **HTML differs by filing agent** (e.g. Workiva vs Donnelley). Section detection may
  miss headings; the missing-section warnings are the tripwire.
- **Pipe tables lose multi-level headers,** which may hurt numeric claims. The later
  table-aware chunker comparison should measure this.
- **Word counts only approximate tokens.** The smaller table chunk size reduces the risk
  of truncation; chunk token lengths get checked once the tokenizer is available locally.
- **Gold-string matching for recall is exact** (after whitespace normalization), so
  text normalization has to be identical on both sides.
- **Local model nondeterminism.** Even at temperature 0 with a fixed seed, outputs can
  change across Ollama versions or hardware. The digest and runtime version are
  recorded, and the committed JSONL is the source of truth.
- **Small local models** may refuse or answer tersely, which shifts the mix of claim
  types compared with frontier models.
- **One annotator** means there's no inter-annotator agreement number. Consider
  double-labeling a subset later.
- **Thermal throttling** on the fanless MacBook Air makes generation slow; batching and
  resume address this.
