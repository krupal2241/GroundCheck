#!/usr/bin/env python3
"""Fine-tune a small encoder (default microsoft/deberta-v3-small) as a claim verifier.

Reads data/training/{train,dev,test}.jsonl from `groundcheck build-training-data`
(synthetic, split by company, gold set excluded) and writes the model, metrics.json
and a model card to models/verifier-deberta-v3-small/.

Examples:
    python scripts/train_verifier.py --fp16                    # Colab T4
    python scripts/train_verifier.py --limit-train 200 --epochs 1   # smoke test on a Mac

Needs the [train] extra (`pip install -e ".[train]"`). On a free Colab GPU, use
notebooks/finetune_verifier_colab.ipynb.
"""

from groundcheck.training.train import main

if __name__ == "__main__":
    raise SystemExit(main())
