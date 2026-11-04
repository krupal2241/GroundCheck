# Labeling guide

How to label one claim with `groundcheck label`. Every label in the gold set is a human
judgment; the tool only checks that the evidence you cite really is in the passages.

All examples below use **made-up text, not from a real filing**. The passages are
written the way `groundcheck` shows them: prose as paragraphs, tables as rows with
cells separated by ` | `.

## The three labels

Judge the claim **only against the retrieved chunks shown on screen**. Not against the
rest of the filing, not against what you know about the company, not against the
model's answer.

| Label | Meaning | Evidence |
|---|---|---|
| `supported` | The chunks state it, or it follows directly from what they state (including correct rounding and unit conversion). | Required: the span(s) that support it. |
| `contradicted` | The chunks state something incompatible with it: a different number, the opposite fact, the wrong year, the wrong unit or scale. | Required: the span(s) it conflicts with. |
| `unsupported` | The chunks neither establish nor contradict it. | None. |

When in doubt between `contradicted` and `unsupported`: `contradicted` needs a span you
can point at that makes the claim false. If you can only say "I can't find this",
it's `unsupported`.

**Evidence spans.** Copy the shortest span that does the job (a phrase or one table
row), straight from the chunk text on screen, and finish the paste with an empty line.
Whitespace and line breaks don't matter; everything else must match exactly (case,
punctuation, `$`, `%`, `(1,234)`, ` | `). For a table value, copy the whole row so the
row label is part of the evidence: `Total net sales | $12,345 | $11,646`. A claim that
combines two values (a difference, a growth rate) cites one span per value. If the
same text appears in two chunks, the tool asks which one you mean.

**Confirm.** Answer `Y` only when you're sure. An unconfirmed label is kept as a draft
and is left out of `export-labels`.

**Invalid claims (`x`).** If the "claim" is not a checkable statement at all (a
fragment like "402 million" or "its common stock", a duplicate of another claim of the
same answer, or a sentence with no subject), press `x` instead of labeling it and give
a short reason. It is saved as invalid: no label, no evidence, never exported, and
counted separately by `label-stats`. Don't use `x` for a claim that is merely wrong or
oddly worded; that still gets a label.

**Edge-case tag.** Pick at most one tag, the one that decided the label. Use none for
an ordinary claim.

## Edge cases

The tag names in brackets are what the tool's edge-case menu shows.

### 1. Negation (`negation`)

The claim flips a "not", "no", "never", "neither", or a negative verb (declined,
decreased, failed to).

> Chunk: *"The Company did not repurchase any shares of its common stock during fiscal 2025."*
>
> Claim: *"The company repurchased shares in fiscal 2025."* → **contradicted**,
> evidence `The Company did not repurchase any shares of its common stock during fiscal 2025`.
>
> Claim: *"The company did not buy back stock in fiscal 2025."* → **supported**, same
> evidence, no tag needed (the negation was handled correctly; tag only when it decided
> the label or was the trap).

### 2. Numeric mismatch (`numeric_mismatch`)

A number in the claim differs from the source beyond what rounding explains:
transposed digits, a different row's value, a made-up figure.

> Chunk: `Total net sales | $12,345 | $11,646`, header row ` | 2025 | 2024`, caption
> *"(in millions)"*.
>
> Claim: *"Total net sales were $12,435 million in 2025."* → **contradicted**,
> evidence `Total net sales | $12,345 | $11,646`.

### 3. Rounding (`rounding`)

Rule (PLAN.md): a rounded claim is **supported only if the source value rounds
correctly to the claim's own precision.**

> Chunk: *"Net sales increased 6.4% to $12,345 million."*
>
> | Claim | Label | Why |
> |---|---|---|
> | "about $12.3 billion" | supported | 12,345 million = 12.345 billion → 12.3 at one decimal |
> | "$12 billion" | supported | rounds to 12 at zero decimals |
> | "$12.4 billion" | contradicted | 12.345 rounds to 12.3, not 12.4 |
> | "6%" | supported | 6.4 → 6 |
> | "7%" | contradicted | 6.4 does not round to 7 |
> | "$12,345.6 million" | unsupported | more precise than the source; the extra digit can't be checked |

Hedges like "about", "approximately", "roughly" don't loosen the rule: "approximately
$12.4 billion" is still contradicted.

### 4. Units and scale (`units_scale`)

Rule (PLAN.md): **a unit or scale mistake is contradicted**, never "close enough".
Look for the scale caption above a table (*"(in millions)"*, *"(millions of dollars)"*,
*"(in thousands, except per share amounts)"*).

> Chunk: *"(in millions, except per share amounts)"* then `Net income | $3,210 | $2,987`
> and `Diluted earnings per share | $2.10 | $1.95`.
>
> | Claim | Label |
> |---|---|
> | "Net income was $3,210 million." | supported |
> | "Net income was $3.21 billion." | supported (scale converted correctly) |
> | "Net income was $3,210." | contradicted (scale dropped) |
> | "Net income was $3,210 billion." | contradicted (wrong scale) |
> | "Diluted EPS was $2.10 million." | contradicted (EPS is per share, not in millions) |
> | "Gross margin rose 2%" when the chunk says "rose 2 percentage points" | contradicted (percent vs percentage points) |

Cite the value row; if the caption is in the same chunk and it is what makes the claim
wrong, cite the caption as a second span.

### 5. Table misreads (`table_misread`)

The value exists in the table but belongs to another column (usually the wrong year)
or another row (a subtotal, a neighbouring line item).

> Chunk:
> ```
>  | 2025 | 2024
> Products | $9,000 | $8,700
> Services | $3,345 | $2,946
> Total net sales | $12,345 | $11,646
> ```
> Claim: *"Services revenue was $2,946 million in 2025."* → **contradicted**
> (that is the 2024 column), evidence `Services | $3,345 | $2,946`.
>
> Claim: *"Products revenue was $12,345 million in 2025."* → **contradicted** (that is
> the total row), evidence `Products | $9,000 | $8,700`.

### 6. Temporal and fiscal-year claims (`temporal`)

Fiscal years don't match calendar years (Apple's fiscal 2025 ends in late September
2025; Walmart's fiscal 2026 ends January 31, 2026). Rules:

- A year in a claim means the filing's own fiscal-year label ("2025" = "fiscal 2025")
  unless the claim says "calendar".
- A value attached to the wrong period is **contradicted**.
- A claim about a period the chunks don't cover (a quarter when only annual figures
  are shown; calendar 2025 when only fiscal 2026 is reported) is **unsupported**.
- "Increased", "decreased", "compared with" claims need both periods' values (cite both).

> Chunk (fiscal year ends January 31): *"Net sales were $680 billion in fiscal 2026,
> compared with $648 billion in fiscal 2025."*
>
> | Claim | Label |
> |---|---|
> | "Net sales were $680 billion in fiscal 2026." | supported |
> | "Net sales were $680 billion in fiscal 2025." | contradicted |
> | "Net sales were $680 billion in calendar 2026." | unsupported |
> | "Net sales grew from fiscal 2025 to fiscal 2026." | supported (cite the sentence) |
> | "Fourth-quarter net sales were $180 billion." | unsupported |

### 7. Partial support (`partial_support`)

The claim bundles two facts and only one checks out (the claim splitter should have
split it, but small models don't always). Label the claim by its weakest part:

- any part contradicted → **contradicted** (cite the conflicting span);
- otherwise any part unsupported → **unsupported**;
- **supported** only if every part is supported.

> Chunk: *"Net sales increased 6% to $12,345 million."*
>
> Claim: *"Net sales increased 6% to $12,345 million, driven by strong iPhone demand."*
> → **unsupported** (the cause is not in the chunks), tag `partial_support`.
>
> Claim: *"Net sales increased 8% to $12,345 million."* → **contradicted** (the 8%),
> evidence `Net sales increased 6% to $12,345 million`.

Write which part failed in the note.

### 8. Hedging (`hedging`)

Hedged claims ("may", "could", "is expected to", "approximately") and hedged sources
(risk factors, forward-looking statements).

- A hedge in the claim doesn't rescue a wrong fact (see rounding).
- A claim that states as fact what the chunk only states as a possibility or an
  expectation is **unsupported**: the chunk does not establish it.
- A claim that correctly reports the hedge is **supported**.

> Chunk: *"We expect capital expenditures to increase in fiscal 2026."*
>
> | Claim | Label |
> |---|---|
> | "The company expects capital expenditures to rise in fiscal 2026." | supported |
> | "Capital expenditures increased in fiscal 2026." | unsupported |
>
> Chunk (a risk factor): *"Competition may reduce our gross margin."*
>
> Claim: *"Competition reduced the company's gross margin."* → **unsupported**.

### 9. Absent evidence = unsupported (`absent_evidence`)

The retrieved chunks don't address the claim at all. It is **unsupported**, even if
you know it is true, and even if the fact is elsewhere in the filing: the test set
measures whether the answer is grounded in what the model was given. No evidence span.
`absent_evidence` is only allowed with `unsupported`.

> Chunks: four passages about revenue and share repurchases.
>
> Claim: *"The company has about 160,000 employees."* → **unsupported**,
> tag `absent_evidence`.

## Workflow

```bash
groundcheck label --only-unlabeled          # label; x = invalid claim; q quits, run again to resume
groundcheck label --company AAPL            # one company; Enter keeps existing labels
groundcheck label-stats                     # counts by label, company, qtype, edge case
groundcheck label-check                     # validate every record (exit 1 on problems)
groundcheck export-labels                   # confirmed labels -> data/eval/gold_labels.jsonl
```

Commit `data/labels/labels.jsonl` after each session. If `label-check` reports a
problem, fix the line in an editor (or delete it and relabel the claim).
