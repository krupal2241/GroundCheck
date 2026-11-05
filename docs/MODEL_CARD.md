# Model card: GroundCheck fine-tuned verifier

The trained model directory also gets its own `README.md` from `scripts/train_verifier.py`
with the exact counts and metrics of that run; this card describes the design.

## Model

| | |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Task | three-way sequence-pair classification: (evidence chunk, claim) -> supported / contradicted / unsupported |
| Input | the evidence chunk and the claim as a sentence pair, max 512 tokens, only the evidence truncated |
| Aggregation over passages | supported if any chunk's top class is supported; else contradicted if any chunk's is; else unsupported. `score` is the highest P(supported) |
| Training | Hugging Face `Trainer`, 3 epochs, learning rate 3e-5, 10% warmup, weight decay 0.01, fp16 on CUDA, best epoch by dev macro-F1, fixed seeds |
| Where it runs | trained on a free Colab T4; inference on CPU or Apple MPS |
| Cost | $0 |

## Training data: synthetic, split by company

Built by `groundcheck build-training-data` from the committed chunks
(`config/training.yaml`):

- **supported:** a prose sentence (8–45 words) from a chunk, or a templated table
  claim ("<row> was <value> for <column>."), paired with its own chunk; half reworded
  (synonyms, rescaling with correct rounding).
- **contradicted:** one rule-based perturbation of a supported claim with the same
  evidence: a changed number, wrong scale, inserted negation, a swapped year that is
  present in the evidence, or a swapped entity or row label.
- **unsupported:** the unchanged claim paired with a different, on-topic chunk (another
  Item of the same company or another company in the same split) that contains none of
  its numbers.

**Splits by company** (no company's text crosses splits):

| Split | Companies | Use |
|---|---|---|
| train | JPM | fitting |
| dev | BRK-B | choosing the epoch |
| test (synthetic) | AAPL, MSFT, AMZN, WMT, XOM, JNJ | synthetic sanity check |

No company with hand-labeled questions is used for training or model selection, so the
gold-set score measures companies the model has never seen. In addition, every chunk
retrieved for any candidate (in any mode) and every gold claim is removed from all
splits. A warning is printed when the train split has fewer than 500 examples.

## Intended use

A cheap, local groundedness check for claims about financial filings, evaluated
against the hand-labeled gold set alongside a prompted LLM judge, MiniCheck and HHEM.
A research baseline, not a production fact-checker.

## Limitations

- **Synthetic-data caveat.** The model learns the perturbation rules. Its scores on the
  synthetic dev and test splits measure how well it learned those rules and will
  overstate accuracy on real model errors. Only the gold-set numbers in
  `docs/RESULTS.md` say how it does on real hallucinations.
- **Label noise** from the rules (an inserted "not" that reads oddly, a different
  section that restates the same fact in words) caps what the synthetic data can teach.
- **One training company.** Training only on JPM keeps the evaluation honest but gives
  little variety in filing style (a bank's vocabulary and tables).
- **Single-chunk evidence.** A claim whose two numbers sit in different chunks can't be
  supported by any one (chunk, claim) pair.
- **512 tokens.** Long chunks are truncated (only the evidence side).
- **English 10-K text only.**

## Metrics

Not reported here until a real training run and gold-set evaluation exist. The
training script writes the synthetic metrics into the model directory; `groundcheck
eval-verifiers --verifier finetuned` and `groundcheck report` put the gold-set numbers
in `docs/RESULTS.md`.
