# Common tasks. Every target is a thin wrapper around a documented command.
# `make help` lists them. $0 policy: nothing here calls a paid API.

PYTHON ?= python

.PHONY: help install dev lint format test coverage dry-run pipeline generate label \
        label-plan label-stats report demo precommit clean-cache

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## install the package with the local-model extras (embeddings, verifiers, report)
	$(PYTHON) -m pip install -e ".[dev,embed,verifiers,report]"

dev:  ## install only what tests and lint need (what CI installs)
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## ruff lint and format check, as in CI
	ruff check .
	ruff format --check .

format:  ## apply ruff fixes and formatting
	ruff check --fix .
	ruff format .

test:  ## run the test suite (offline, no models)
	$(PYTHON) -m pytest -q

coverage:  ## tests with a coverage report (needs pytest-cov)
	$(PYTHON) -m pytest -q --cov=groundcheck --cov-report=term

dry-run:  ## the whole pipeline offline with fakes, into data/cache/dry-run/pipeline
	groundcheck pipeline --dry-run

pipeline:  ## the real pipeline (needs SEC_USER_AGENT, Ollama and the extras); skips unchanged stages
	groundcheck pipeline

generate:  ## candidates in all three modes, in cool-down batches for a fanless laptop
	for mode in rag no-context wrong-context; do \
	  $(PYTHON) scripts/generate_candidates.py --mode $$mode --batch-size 5 --sleep-between-batches 60 || exit 1; \
	done

label-plan:  ## propose a balanced 300-claim labeling sample
	groundcheck label-plan --claims 300

label:  ## label the planned sample, resuming where you stopped
	groundcheck label --plan data/labels/label_plan.json --only-unlabeled

label-stats:  ## label counts by mode, label, company and question type
	groundcheck label-stats

report:  ## regenerate README results, docs/RESULTS.md and docs/figures/ from result files
	groundcheck report

demo:  ## local Gradio demo on 127.0.0.1:7860
	groundcheck demo

precommit:  ## install the pre-commit hooks (ruff)
	$(PYTHON) -m pip install pre-commit && pre-commit install

clean-cache:  ## delete re-creatable caches (LLM replies, dry runs, pipeline state)
	rm -rf data/cache .pytest_cache .ruff_cache
