# Design notes

For each module: what it does, why it is built this way, and what the alternatives were.
Each section is written when its module is built.

## Project scaffolding (Day 1)

**What:** a Python package in a `src/` layout, configured by `pyproject.toml`, tested
with pytest, linted and formatted with ruff, and checked by GitHub Actions on every push.

**Why a `src/` layout:** the package lives in `src/groundcheck/`, not at the repo root.
That means `import groundcheck` only works once the package is *installed*
(`pip install -e .`), so the tests exercise the installed package exactly as a user
would get it. With a flat layout, Python can silently import from the working directory
and hide packaging mistakes, such as a missing file or a wrong package path, until
someone else tries to install the project.

**Why `pyproject.toml` with setuptools:** it's the standard place for metadata,
dependencies and tool settings (PEP 621), so one file replaces `setup.py`,
`requirements.txt`, `setup.cfg` and separate tool configs. Setuptools is the most widely
known build backend. *Alternatives:* Poetry or Hatch, which add lock files and
environment management. They're nice, but they're one more tool to learn and explain,
and nothing here needs them yet.

**Why runtime dependencies are added only when needed:** `dependencies = []` grows as
each module is written, so every listed package is something the code really imports. A
reviewer can read the list and trust it.

**Why optional extras (`[dev]`, later `[embed]`):** the embedding stack
(sentence-transformers → PyTorch) is hundreds of MB. Tests don't need it, because they
use a fake embedder. Keeping it in an extra makes CI fast, and it means CI never
downloads a model.

**Why pytest:** plain `assert` statements, fixtures for setup, and `capsys`/`caplog` to
check printed output and log warnings (we'll use `caplog` to test the missing-section
warning). *Alternative:* the built-in `unittest`, which is more verbose and class-based.

**Why ruff:** one fast tool that does both linting (pyflakes, pycodestyle, import
sorting, bug patterns) and formatting (Black-compatible). *Alternative:* flake8 + isort +
black, which are three tools and three configs that do the same job.

**Why argparse for the CLI:** it's in the standard library and explicit; each subcommand
is a few readable lines. The CLI only parses arguments and calls functions in the
modules, so all the logic stays testable without the CLI. *Alternatives:* Typer or Click
are shorter to write but rely on decorators and type-hint magic that's harder to explain
line by line.

**Why CI is split into a lint job and a test matrix:** a lint failure and a test failure
show up as separate red marks, so it's obvious which one broke. Tests run on Python 3.11
(the minimum we support) and 3.12, to catch use of newer syntax. Tests never use the
network or models, so CI is fast and deterministic.

**Secrets:** `.env` is gitignored and `.env.example` documents the variable names with
empty values. Code reads secrets only from environment variables.
