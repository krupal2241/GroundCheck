# Dataset card: GroundCheck claim-level gold set

## Summary

Claims extracted from answers that a small local LLM (llama3.2:3b by default) gave to
questions about eight 2025–2026 SEC 10-K filings. Each claim is labeled by a human as
**supported**, **contradicted** or **unsupported** by the passages retrieved for its
question, with an exact evidence span and an optional edge-case tag. The confirmed
labels are exported to `data/eval/gold_labels.jsonl`, which is the test set for every
verifier in this repository.

| | |
|---|---|
| Source documents | Latest 10-K of AAPL, MSFT, AMZN, WMT, XOM, JNJ (labeled), JPM, BRK-B (not labeled), pinned by accession number in `config/companies.yaml` |
| Unit | One claim (an atomic, self-contained sentence) with its question, answer and retrieved passages |
| Labels | supported / contradicted / unsupported, plus "invalid" for malformed claims (never exported) |
| Annotators | **One person** (the project author) |
| Language | English |
| License | MIT (the code and the labels); the filings are public SEC documents |

## How it was made

1. **Questions (60).** Ten per labeled company in a fixed mix of types:
   `numeric_lookup` x2, `table` x2, `multi_step` x2, `yes_no` x1, `temporal` x2,
   `unanswerable` x1. Thirty live in `data/eval/retrieval_questions.yaml` (with gold
   evidence strings for the retrieval evaluation), thirty in
   `data/questions/candidate_questions.yaml`. **All were drafted by an AI assistant
   (Claude) from the committed chunks**, at the author's request, with notes giving the
   expected answer and its source text; the author is responsible for reviewing them.
2. **Retrieval.** bge-small-en-v1.5 embeddings, one exact FAISS index per company, top
   5 chunks per question. Chunks are 250-word prose windows (50-word overlap) and
   table chunks split on row boundaries with the header repeated.
3. **Answers, in three generation modes** (`scripts/generate_candidates.py --mode`):

   | Mode | The model sees | Why |
   |---|---|---|
   | `rag` | the 5 retrieved passages | the normal case; mostly supported claims |
   | `no-context` | nothing (closed-book) | what a small model asserts from memory |
   | `wrong-context` | 5 passages for the same question from another company, or from other Items of the same filing | answers built from the wrong evidence |

   Temperature 0, seed 42; the model digest, runtime version and prompt version are
   stored with every answer.
4. **Claims.** The answer is split by the same model with a few-shot prompt (v2), then
   a deterministic validator rejects fragments, verb-less claims, duplicates, passage
   citations and numbers not in the answer, re-asking up to 3 times. The counts are in
   each candidate's `claim_filter`.
5. **Sampling.** `groundcheck label-plan` picks a sample balanced across company,
   question type and mode.
6. **Labeling.** `groundcheck label`, one claim at a time, against the passages
   **retrieved for the claim's own question** (in every mode, even when the model saw
   different passages or none). Rules are in `docs/labeling_guide.md`. Evidence spans
   must be exact substrings of a retrieved chunk (whitespace-normalized), checked by
   the tool. Label suggestions from a verifier are off by default; if used, each label
   records what was shown and whether the annotator agreed.
7. **Export.** Only confirmed, valid labels are exported, with question, answer,
   passages, mode and generator metadata (`groundcheck export-labels`).

## Label definitions

| Label | Meaning | Evidence |
|---|---|---|
| supported | The retrieved passages state it, or it follows directly (correct rounding and unit conversion included) | required |
| contradicted | The passages state something incompatible: another number, the opposite fact, the wrong period, unit or scale | required |
| unsupported | The passages neither establish nor contradict it | none |
| invalid (`x`) | Not a checkable claim (fragment, duplicate, no subject); kept in the label file, never exported | none |

A bundled claim takes the label of its weakest part. A claim more precise than its
source is unsupported. Edge-case tags: negation, numeric_mismatch, rounding,
units_scale, table_misread, temporal, partial_support, hedging, absent_evidence.

## Class balance

Filled in by `groundcheck report` from the confirmed labels (rows: generation mode).

<!-- BALANCE:START -->
*No confirmed labels yet:* run `groundcheck label`, then `groundcheck report`.
<!-- BALANCE:END -->

## Known biases and limitations

- **Single annotator.** No inter-annotator agreement; systematic misreadings by one
  person go undetected. A double-labeled subset is the obvious next step.
- **AI-drafted questions.** The questions were written by an AI assistant looking at
  the chunks, so they tend to match the chunk wording (easier retrieval) and to favor
  facts that are stated cleanly in one place.
- **Answers from a 3B model.** Error types reflect a small local model (fragmentary
  splitting, copied numbers in the wrong context); a larger generator would produce
  fewer, subtler errors.
- **Modes shape the label mix.** no-context and wrong-context exist to produce
  contradicted and unsupported claims; the mix is designed, not natural, so accuracy on
  the whole set is not "accuracy in deployment". Per-mode results are reported.
- **"Unsupported" is relative to the top 5 passages,** not to the whole filing.
- **Assisted labels** (if `--suggest` was ever used) may lean towards the suggesting
  verifier; `label-stats` reports agreement and the label mix with and without
  suggestions.
- **Eight large US companies, one fiscal year each,** in English.

## Intended use

Evaluating claim-level groundedness checkers on financial-filing RAG output. Not for
training (the fine-tuned verifier never sees it: every gold chunk and claim is excluded
from its training data), and not as a measure of factual truth about the companies.
