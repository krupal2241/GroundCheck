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

**Mitigation.** The gold set never enters training (gold chunks, all candidate chunks,
every other chunk containing a gold evidence span, and gold claims are excluded in
code); no labeled company is in train or dev, and wrong-context generation never
borrows the training companies' passages, so the gold-set score is on unseen companies.
Not mitigated: dev is held-out Items of the two training filings, so it is not
independent of train (it only picks the epoch); the model card and every report say which
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
  column. The per-tag, per-company and per-mode tables show supported-vs-not accuracy
  for them (marked †), so a † cell and a three-way cell in one row are different
  metrics; the per-verifier sections of `verifier_eval.md` have both columns.
- **Parse errors.** A prompted verdict with no parseable JSON is scored as unsupported,
  a free correct answer when the gold label is unsupported; accuracy with parse errors
  counted as wrong is reported next to it.
- **Subset runs.** Rows computed with `--company` or `--limit` are flagged (different
  n or claim-set hash); do not compare them with full runs.
- **Evidence agreement is partly circular.** A gold span accepted from the one-key
  suggestion was picked by the same lexical rule the cross-encoder verifiers cite spans
  with; agreement is reported separately for hand-pasted spans, the independent figure.
- **The numeric baseline does not read.** `NumericVerifier` matches numbers, periods
  and row labels only; it is unsupported on every claim without a number. It is a
  floor for numeric claims, not a fourth competitor overall.
- **Latency is hardware-specific.** Measured on one machine (an M-series Air, which
  throttles under sustained load). Relative order is more trustworthy than the
  absolute milliseconds.
- **Anchoring.** If label suggestions were used, labels may lean towards the
  suggesting verifier; `label-stats` reports how many labels were assisted and
  `eval-verifiers` reports each verifier's score on labels made with and without a
  suggestion on screen. Compare verifiers on the "not shown" row.
