# Threats to validity

What could make GroundCheck's results misleading, how much it matters, and what is done
about it. Read this before quoting any number from `docs/RESULTS.md`.

## 1. Small test set

**Threat.** A gold set of a few hundred claims gives wide confidence intervals. With
300 claims, a 95% interval on an accuracy near 70% is roughly ±5 points; per-class,
per-mode and per-edge-case cells hold tens of claims or fewer, so their percentages can
swing by 10–20 points on a handful of labels.

**Mitigation.** Counts (n) are printed next to every breakdown; macro-F1 is reported
because accuracy hides a weak minority class; `label-plan` spreads the sample across
companies, question types and modes so no cell is empty by accident. **Rule of thumb:**
don't treat a gap of under ~5 points between two verifiers on the full set as real, and
read per-tag numbers as qualitative. A paired comparison on the same claims (the error
report lists each verifier's mistakes) is more informative than the two accuracies.

## 2. Single annotator

**Threat.** One person labels everything. Systematic misreadings (a habit of calling
rounding cases supported, missing a scale error in a table) go uncaught, and there is no
inter-annotator agreement to bound label noise.

**Mitigation.** A written guide with worked edge cases (`docs/labeling_guide.md`),
evidence spans that must be exact substrings (so every supported/contradicted label
points at checkable text), a confirm step, and the `x` key for claims that can't be
labeled. Not mitigated: agreement. Double-labeling a random 50-claim subset after a gap
of some days (or by a second person) and reporting Cohen's kappa is the obvious fix.

## 3. AI-drafted questions

**Threat.** The 60 questions and the retrieval gold strings were drafted by an AI
assistant reading the chunks. Such questions echo the filing's wording, which inflates
retrieval recall, and favor facts stated cleanly in one chunk, which may make claims
easier to verify than real user questions would.

**Mitigation.** Each question's note names the expected answer and its source, so a
reviewer can check it; the question types force table, multi-step, temporal and
unanswerable questions. Not mitigated: phrasing bias. Retrieval numbers in particular
should be read as an upper bound for this corpus.

## 4. Retrieval quality

**Threat.** Labels are relative to the top 5 retrieved passages. When retrieval misses
the right chunk, a true claim is labeled unsupported, and the verifier is judged on
whether it agrees that the passages don't say it. Retrieval errors and verifier errors
are entangled, and the share of unsupported labels depends on the retriever.

**Mitigation.** The labeling guide and every verifier use exactly the same passages
(stored with each candidate), so the comparison between verifiers is fair even when
retrieval is imperfect; retrieval recall@k is reported separately so its quality is
visible. Not mitigated: the unsupported share would change with a better retriever.

## 5. Synthetic training data

**Threat.** The fine-tuned verifier is trained on rule-based perturbations of filing
sentences. It can learn the rules' artifacts (an inserted "not", a templated table
sentence) rather than groundedness, so its synthetic dev/test scores overstate how it
handles real model errors; label noise in the rules caps what it can learn.

**Mitigation.** The gold set never enters training (gold chunks, all candidate chunks
and gold claims are excluded in code); no labeled company is in train or dev, so the
gold-set score is on unseen companies; the model card and every report say which
numbers are synthetic. Only gold-set numbers are compared with the other verifiers.

## Other threats

- **Designed label mix.** The no-context and wrong-context modes deliberately produce
  hallucinations, so the class balance is chosen, not natural; overall accuracy is not
  deployment accuracy. Per-mode results are reported for that reason.
- **One generator.** All claims come from one 3B model (or one Colab model); verifier
  rankings could change for a frontier model's subtler errors.
- **Claim-splitting filter.** The validator drops malformed claims; the remaining
  claims are cleaner than raw model output, which may make verification look easier.
  `claim_filter` statistics are kept in every candidate.
- **Binary verifiers on a three-way task.** MiniCheck and HHEM can't predict
  contradicted; their three-way scores are low by construction. Compare on the binary
  column.
- **Latency is hardware-specific.** Measured on one machine (an M-series Air, which
  throttles under sustained load). Relative order is more trustworthy than the
  absolute milliseconds.
- **Anchoring.** If label suggestions were used, labels may lean towards the
  suggesting verifier; `label-stats` reports how many labels were assisted.
