#!/usr/bin/env python3
"""Generate candidate answers and claims for the labeled test set.

Per question: retrieve the top-k chunks with the configured retrieval variant
(config/retrieval.yaml `default`, or --variant), answer from only those chunks with a
local LLM (Ollama by default), split the answer into atomic claims, and append one
record to data/candidates/<TICKER>.jsonl; each record names the retrieval variant that
produced it. Resumable: re-running skips questions that already
have a record for the same provider and model (and refuses to mix in another variant).

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
