# Interview notes

The main design decisions, the questions an interviewer is likely to ask about each,
and short, honest answers. Everything here is drawn from `docs/DESIGN.md`, which has the
full reasoning. No results are quoted: the numbers live only in `docs/RESULTS.md`, once
real runs exist.

## The project in one breath

GroundCheck checks, claim by claim, whether a RAG answer about a 10-K filing is backed
by the passages it was given: supported, contradicted or unsupported, with the evidence
span. I built a hand-labeled test set from a small local model's real answers and
benchmarked a prompted local LLM judge, MiniCheck, HHEM and a fine-tuned DeBERTa-small
on accuracy, latency and cost, all on an 8 GB laptop for $0.

## Problem framing

**Why claim-level and not answer-level?**
An answer with four correct numbers and one wrong one is neither "good" nor
"hallucinated". Claims can each be checked against a span, the errors can be counted
by type, and a product can highlight the one bad sentence instead of rejecting the
whole answer.

**Why three labels when MiniCheck and HHEM use two?**
"The passage says otherwise" and "the passage doesn't say" call for different actions:
the first is a wrong answer, the second may be a retrieval miss. I report a binary view
(supported vs not) too, so the binary models are compared fairly; their three-way
scores are low by construction, and I say so wherever they appear.

**Why 10-K filings?**
Dense numbers, units and scales, fiscal years that don't match calendar years, and
facts that live in tables. That's where answers go wrong in ways a lexical check can't
catch, and where existing checkers are least tested.

**What's actually novel?**
Not a model. MiniCheck (small checkers trained on synthetic data) and HHEM exist, and
rule-based negatives go back to FactCC. What's new is the test set (domain, three
labels, evidence spans, edge-case tags, natural hallucinations from three generation
modes) and a reproducible $0 benchmark including latency. I'd rather claim that clearly
than overclaim.

## Data and retrieval

**Why one FAISS index per company?**
Every question is about one company. A shared index would let a big filing crowd out a
small one and return other companies' chunks; per-company indexes give each question its
full top-k from the right filing. Exact `IndexFlatIP` because a few thousand vectors
per company don't need approximate search.

**Why fixed-size chunks with special table handling?**
It's the simplest baseline that respects bge-small's 512-token limit: prose in 250-word
windows with overlap, tables split only on row boundaries with the header repeated, so
a row is never cut from its column labels. A table-aware chunker can plug in behind the
same interface for comparison.

**How do you know retrieval works?**
Recall@k and section-hit@k against gold evidence strings that must occur verbatim in
the chunks (checked by `eval-retrieval --check`). Caveat I'd raise myself: the questions
were AI-drafted from the chunks, so recall is an optimistic upper bound.

## Generation and claims

**Why a 3B local model?**
$0 budget and 8 GB of RAM. It also makes plenty of mistakes, which is useful for a
hallucination test set. Temperature 0, seed 42, and the model digest and Ollama version
are recorded with every answer, because outputs can still change across builds.

**The claim splitter produced fragments like "402 million". What did you do?**
Two layers. A few-shot prompt whose first example *is* that failure with the corrected
two-claim output. Then a deterministic validator: at least 5 words, a verb, no
duplicates or near-duplicates, no "passage [n]", and no number that isn't in the
answer. Rejected claims trigger up to 3 retries with the specific problems listed, and
the filter counts are stored in every record. *Follow-up: why not just trust a better
prompt?* Because a fragment in the gold set is worse than a missing claim, and rules
are cheap, testable and explainable.

**How do you avoid a test set that's 90% "supported"?**
Three generation modes: normal RAG, closed-book (no passages), and wrong-context
(passages from another company or another Item). The errors are the model's own, not
synthetic. Claims are always labeled against the passages retrieved for the correct
question, whatever the model saw. Then `label-plan` balances the sample across company,
question type and mode.

**Isn't a designed label mix unrealistic?**
Yes. Overall accuracy on this set is not deployment accuracy, which is why results are
also broken down by mode.

## Labeling

**Why exact-substring evidence spans with offsets?**
Every supported or contradicted label then points at checkable text. The tool rejects a
span that isn't in a retrieved chunk, and offsets record which occurrence was meant
when "$12,345" appears twice.

**Single annotator: isn't that a problem?**
It is, and I say so in the dataset card and threats document. Mitigations are a written
guide with worked edge cases, mechanical span checks and a confirm step. The fix I'd do
next is double-labeling a random subset and reporting kappa.

**You have verifiers. Why not let them pre-label?**
Anchoring. If the annotator sees "supported", the gold set drifts towards that
verifier, which then scores well on a test it helped write. Label suggestions are off
by default; when on, each label records what was shown and whether I agreed, and
`label-stats` compares labels made with and without suggestions. The evidence-span
suggestion appears only *after* the label key, so it can change which span is cited,
not the label.

## Verifiers and evaluation

**How do cross-encoders handle five passages?**
Each (chunk, claim) pair is scored separately and the maximum support wins: one passage
stating the claim is enough, every input fits in 512 tokens, and that's what MiniCheck
does with long documents. Known limitation: a claim needing two chunks can't be
supported by either alone.

**Did you tune the 0.5 threshold?**
No. Tuning it on the gold set would leak test labels into the verifier. 0.5 is the
models' own decision boundary.

**How do you compare cost when everything is free?**
Local verifiers are "$0 (local)". A hosted judge is never called: its cost is an
estimate from prices you type into `config/prices.yaml`, input tokens counted as
characters / 4 from the real prompt, output tokens measured from the local judge, and
it's labeled "estimate" everywhere it appears.

**How is latency measured?**
Per claim, inside the verifier's `verify` call, covering all passages and the span
lookup; p50 and p95. Model loading is excluded. It's one laptop, so the ranking is more
trustworthy than the milliseconds.

## Fine-tuning

**Where does training data come from without leaking the test set?**
Synthetic: sentences and table rows from the chunks as supported, rule-based
perturbations as contradicted, the same claim with an on-topic different chunk as
unsupported. Every chunk retrieved for any candidate and every gold claim is excluded
in code, and splits are by company.

**Why train only on JPM?**
Earlier I trained on two labeled companies too. Excluding gold chunks prevents literal
leakage, but the rest of the same filing still teaches the model that company's
vocabulary and table layouts. Now no labeled company is in train or dev (BRK-B is dev),
so the gold score is on unseen companies. The cost is less variety, and the tooling
warns below 500 training examples.

**Will the synthetic scores carry over?**
No. The model learns the perturbation rules; synthetic dev/test scores overstate real
performance. Only the gold-set numbers count, and the model card says so.

## Engineering

**How do you make a ten-stage pipeline reproducible?**
`groundcheck pipeline` calls the same entry points as the individual commands. It
fingerprints each stage's input files (sha256) and settings, and skips a stage whose
fingerprint matches the last success, so an unchanged re-parse doesn't trigger
re-indexing. Content hashes rather than mtimes, because git checkouts and Colab
round-trips change mtimes. `--dry-run` runs the whole thing offline with fakes in
seconds.

**How do you keep the README honest?**
`groundcheck report` writes the results section from result files only. If any input
is missing it lists what's missing and leaves the README untouched, so a TBD never
turns into an invented number.

**How do tests run without models or network?**
Fake embedder, fake LLM, fake verifier, stand-in `torch`/`transformers`/`gradio`
modules, an import blocker that fails any test that loads a heavy ML package for real,
and a network blocker. That rule came from a real crash: faiss and torch each bring an
OpenMP runtime, and loading both killed pytest on the Mac.

**Why not LangChain or LiteLLM?**
The HTTP call and its parameters are the part worth being able to explain, and a
provider interface plus a disk cache and a paid-call guard is a few hundred lines.

## What I'd do next

Double-label a subset for agreement; a table-aware chunker compared on recall and on
table-misread errors; a second generator model; and confidence intervals (bootstrap) in
the report.
