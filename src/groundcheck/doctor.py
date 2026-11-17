"""`groundcheck doctor`: is this machine ready to run the pipeline?

It checks, without loading a model or downloading anything:

- the Python version (pyproject's `requires-python`);
- which optional extras are installed (by looking for their packages, not importing
  them: importing torch just to say it is there would take seconds and memory);
- that Ollama answers on the configured URL and has the configured model pulled;
- free disk space (models and indexes need a few GB);
- that `SEC_USER_AGENT` is set and well-formed (needed only by `download`);
- that the data folders each stage writes exist, with the command that creates them;
- that no `*_API_KEY` environment variable is set. The project's policy is $0: nothing
  here calls a paid API, and a key in the environment is how one would start to by
  accident (a library picking it up, a provider added later). Only the variable NAMES
  are printed, never their values.

Each line is `[ OK ]`, `[WARN]` or `[FAIL]`, followed by what to do about it. The exit
code is 1 if anything failed, 0 otherwise (warnings do not fail: an extra you do not
use, or Ollama not running while you only evaluate retrieval, is fine).

Everything that touches the outside (environment, HTTP, disk, package lookup) is
passed in, so tests run it without a network or real packages.
"""

import importlib.util
import os
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from groundcheck.config import REPO_ROOT, ConfigError, LLMConfig, get_sec_user_agent

MIN_PYTHON = (3, 11)
WARN_FREE_GB = 5.0  # MiniCheck ~1.7 GB, HHEM ~0.4 GB, llama3.2:3b ~2 GB, indexes, cache
FAIL_FREE_GB = 1.0

# extra -> (what it is for, the packages that show it is installed)
EXTRAS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dev": ("tests and lint", ("pytest", "ruff", "hypothesis")),
    "embed": ("the bge embedding models (index, hybrid retrieval)", ("sentence_transformers",)),
    "verifiers": ("MiniCheck, HHEM and the fine-tuned verifier", ("torch", "transformers")),
    "report": ("charts in `groundcheck report`", ("matplotlib",)),
    "demo": ("the local Gradio demo", ("gradio",)),
    "train": ("fine-tuning the verifier (best on Colab)", ("accelerate", "torch", "transformers")),
}

# folder (under the repo) -> (what writes it, the command)
DATA_FOLDERS: dict[str, tuple[str, str]] = {
    "data/raw": ("downloaded filings", "groundcheck download"),
    "data/parsed": ("parsed Item sections", "groundcheck parse"),
    "data/chunks": ("chunks", "groundcheck chunk"),
    "data/index": ("dense indexes", "groundcheck index"),
    "data/candidates": ("generated answers and claims", "python scripts/generate_candidates.py"),
    "data/labels": ("your labels", "groundcheck label"),
    "data/eval": ("retrieval questions and the gold export", "groundcheck export-labels"),
}

Status = Literal["ok", "warn", "fail"]
MARK = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""

    def lines(self) -> list[str]:
        out = [f"{MARK[self.status]} {self.name}: {self.detail}"]
        if self.fix and self.status != "ok":
            out.append(f"       fix: {self.fix}")
        return out


def _has_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _http_get(url: str, timeout: float = 3.0) -> dict[str, Any]:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def check_python(version: tuple[int, ...] = sys.version_info[:3]) -> Check:
    text = ".".join(str(v) for v in version)
    if tuple(version[:2]) >= MIN_PYTHON:
        return Check("Python", "ok", text)
    need = ".".join(str(v) for v in MIN_PYTHON)
    return Check(
        "Python", "fail", f"{text}, but GroundCheck needs {need} or newer",
        f"install Python {need}+ and create a new virtual environment with it",
    )  # fmt: skip


def check_extras(has_package: Callable[[str], bool] = _has_package) -> list[Check]:
    checks = []
    for extra, (purpose, packages) in EXTRAS.items():
        missing = [p for p in packages if not has_package(p)]
        if not missing:
            checks.append(Check(f"extra [{extra}]", "ok", f"installed ({purpose})"))
        else:
            checks.append(
                Check(
                    f"extra [{extra}]",
                    "warn",
                    f"not installed (missing {', '.join(missing)}); needed for {purpose}",
                    f'pip install -e ".[{extra}]"',
                )  # fmt: skip
            )
    return checks


def check_ollama(config: LLMConfig, http_get: Callable[[str], dict] = _http_get) -> list[Check]:
    if config.provider != "ollama":
        detail = f"provider is {config.provider!r} in the LLM config, so Ollama is not needed"
        return [Check("Ollama", "ok", detail)]
    url = config.ollama_url.rstrip("/")
    try:
        version = http_get(f"{url}/api/version").get("version", "unknown")
    except Exception as exc:  # noqa: BLE001  any failure means "not reachable"
        return [
            Check(
                "Ollama",
                "warn",
                f"not reachable at {url} ({exc.__class__.__name__})",
                "start the Ollama app or run `ollama serve` (needed for candidate "
                "generation and the prompted verifier, not for retrieval or labeling)",
            )  # fmt: skip
        ]
    checks = [Check("Ollama", "ok", f"version {version} at {url}")]
    try:
        models = http_get(f"{url}/api/tags").get("models", [])
    except Exception as exc:  # noqa: BLE001
        detail = f"could not list models ({exc.__class__.__name__})"
        return [*checks, Check("Ollama model", "warn", detail, "run `ollama list`")]
    names = {m.get("name") for m in models} | {m.get("model") for m in models}
    wanted = {config.model, f"{config.model}:latest"}
    if names & wanted:
        checks.append(Check("Ollama model", "ok", f"{config.model} is pulled"))
    else:
        have = ", ".join(sorted(n for n in names if n)) or "none"
        checks.append(
            Check(
                "Ollama model",
                "warn",
                f"{config.model} is not pulled (installed: {have})",
                f"ollama pull {config.model}",
            )  # fmt: skip
        )
    return checks


def check_disk(root: Path = REPO_ROOT, free_bytes: int | None = None) -> Check:
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    gb = free / 1e9
    detail = f"{gb:.1f} GB free on the disk holding {root}"
    if gb < FAIL_FREE_GB:
        fix = "free some space: models, indexes and the LLM cache need a few GB"
        return Check("Disk space", "fail", detail, fix)
    if gb < WARN_FREE_GB:
        fix = (
            f"under {WARN_FREE_GB:g} GB: enough for retrieval, tight for the verifier models "
            "(MiniCheck is about 1.7 GB); `make clean-cache` frees the LLM cache"
        )
        return Check("Disk space", "warn", detail, fix)
    return Check("Disk space", "ok", detail)


def check_user_agent(env_file: Path | None = None) -> Check:
    try:
        get_sec_user_agent(env_file)
    except ConfigError as exc:
        return Check(
            "SEC_USER_AGENT", "warn", str(exc).split(". ")[0].rstrip("."),
            'add SEC_USER_AGENT="Your Name you@example.com" to .env (see .env.example); '
            "only `groundcheck download` needs it",
        )  # fmt: skip
    return Check("SEC_USER_AGENT", "ok", "set and well-formed")


def check_data(root: Path = REPO_ROOT) -> list[Check]:
    checks = []
    for folder, (what, command) in DATA_FOLDERS.items():
        path = root / folder
        if path.is_dir() and any(path.iterdir()):
            checks.append(Check(folder, "ok", f"present ({what})"))
        else:
            state = "empty" if path.is_dir() else "missing"
            checks.append(Check(folder, "warn", f"{state} ({what})", f"run `{command}`"))
    return checks


def check_api_keys(environ: Mapping[str, str] = os.environ) -> Check:
    """Names (never values) of `*_API_KEY` variables that are set and non-empty."""
    names = sorted(k for k, v in environ.items() if k.upper().endswith("_API_KEY") and v.strip())
    if not names:
        return Check("Paid API keys", "ok", "no *_API_KEY variable is set ($0 policy)")
    return Check(
        "Paid API keys", "warn",
        f"{len(names)} API key variable(s) set in this environment: {', '.join(names)}. "
        "This project's policy is $0 and none of its commands calls a paid API, but a "
        "library or a provider added later could pick a key up",
        "unset " + " ".join(names) + "   (and remove them from .env / your shell profile "
        "for this project)",
    )  # fmt: skip


def run_checks(
    llm_config: LLMConfig,
    *,
    root: Path = REPO_ROOT,
    env_file: Path | None = None,
    environ: Mapping[str, str] = os.environ,
    has_package: Callable[[str], bool] | None = None,
    http_get: Callable[[str], dict] | None = None,
    python_version: tuple[int, ...] = sys.version_info[:3],
    free_bytes: int | None = None,
) -> list[Check]:
    return [
        check_python(python_version),
        *check_extras(has_package or _has_package),
        *check_ollama(llm_config, http_get or _http_get),
        check_disk(root, free_bytes),
        check_user_agent(env_file),
        *check_data(root),
        check_api_keys(environ),
    ]


def format_checks(checks: list[Check]) -> str:
    lines = [line for check in checks for line in check.lines()]
    counts = {s: sum(c.status == s for c in checks) for s in MARK}
    verdict = (
        "Not ready: fix the [FAIL] lines first."
        if counts["fail"]
        else "Ready, with warnings: each [WARN] line says when it matters."
        if counts["warn"]
        else "Ready."
    )
    summary = f"{counts['ok']} ok, {counts['warn']} warning(s), {counts['fail']} failure(s)."
    return "\n".join([*lines, "", f"{summary} {verdict}"])
