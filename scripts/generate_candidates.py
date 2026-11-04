#!/usr/bin/env python3
"""Generate candidate answers and claims for the labeled test set.

Per question: retrieve the top-k chunks from the company's FAISS index, answer from
only those chunks with a local LLM (Ollama by default), split the answer into atomic
claims, and append one record to data/candidates/<TICKER>.jsonl. Resumable: re-running
skips questions that already have a record for the same provider and model.

Examples:
    python scripts/generate_candidates.py --dry-run --limit 3
    python scripts/generate_candidates.py --company AAPL --batch-size 5 --sleep-between-batches 60
    python scripts/generate_candidates.py --provider hf-local --model Qwen/Qwen2.5-3B-Instruct

Run `python scripts/generate_candidates.py --help` for every option. Needs the package
installed (`pip install -e .`).
"""

from groundcheck.generation.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
