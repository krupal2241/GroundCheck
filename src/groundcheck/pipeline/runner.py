"""`groundcheck pipeline`: every stage in order, skipping stages whose inputs haven't changed.

    download -> parse -> chunk -> index -> eval-retrieval -> generate -> export-labels
             -> eval-verifiers -> error-report -> report

Each stage runs the same code as its own command (`groundcheck parse`, ...,
`scripts/generate_candidates.py`), so the pipeline is a driver, not a second
implementation. Skipping: before a stage runs, a fingerprint is computed from the
sha256 of every input file, the stage's settings and a stage version. If it equals the
fingerprint stored after the stage last succeeded (`<state dir>/<stage>.json`) and
every output still exists, the stage is skipped as "unchanged". A failed stage stops
the pipeline and stores nothing, so it runs again next time. `--force` ignores the
stored fingerprints.

`--dry-run` runs everything offline with fakes and writes only to a scratch directory
(`data/cache/dry-run/pipeline/`, gitignored): a fake SEC client serving synthetic
filings built from the committed chunks, no tokenizer limit, the hashing embedder, the
fake LLM and the fake verifier. Labels are never invented: if there is no label file,
export-labels and the stages after it report "nothing to do".
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from groundcheck.chunking.base import DEFAULT_CHUNKS_DIR
from groundcheck.config import DEFAULT_COMPANIES_PATH, REPO_ROOT, ConfigError, load_companies
from groundcheck.edgar.download import DEFAULT_RAW_DIR
from groundcheck.generation.modes import MODES
from groundcheck.generation.pipeline import DEFAULT_CANDIDATES_DIR
from groundcheck.generation.questions import DEFAULT_CANDIDATE_QUESTIONS_PATH
from groundcheck.labeling.store import DEFAULT_EXPORT_PATH, DEFAULT_LABELS_PATH
from groundcheck.parsing.filing import DEFAULT_PARSED_DIR
from groundcheck.retrieval.evaluate import DEFAULT_QUESTIONS_PATH
from groundcheck.retrieval.index import DEFAULT_INDEX_DIR
from groundcheck.retrieval.variants import DEFAULT_RETRIEVAL_CONFIG_PATH, load_retrieval_config
from groundcheck.verify.evaluate import DEFAULT_RESULTS_DIR

STAGES: tuple[str, ...] = (
    "download",
    "parse",
    "chunk",
    "index",
    "eval-retrieval",
    "generate",
    "export-labels",
    "eval-verifiers",
    "error-report",
    "report",
)
STAGE_VERSION = 1  # bump to invalidate every stored fingerprint
DRY_RUN_DIR = REPO_ROOT / "data" / "cache" / "dry-run" / "pipeline"
DEFAULT_STATE_DIR = REPO_ROOT / "data" / "cache" / "pipeline"
DEFAULT_VERIFIERS = ("prompted", "hhem", "minicheck")
NOTHING_TO_DO = 3  # a stage's "ran, but had no input" code (never stored as done)


@dataclass
class Paths:
    config: Path = DEFAULT_COMPANIES_PATH
    raw: Path = DEFAULT_RAW_DIR
    parsed: Path = DEFAULT_PARSED_DIR
    chunks: Path = DEFAULT_CHUNKS_DIR
    index: Path = DEFAULT_INDEX_DIR
    retrieval_questions: Path = DEFAULT_QUESTIONS_PATH
    candidate_questions: Path = DEFAULT_CANDIDATE_QUESTIONS_PATH
    retrieval_results: Path = DEFAULT_QUESTIONS_PATH.parent / "results"
    candidates: Path = DEFAULT_CANDIDATES_DIR
    labels: Path = DEFAULT_LABELS_PATH
    gold: Path = DEFAULT_EXPORT_PATH
    results: Path = DEFAULT_RESULTS_DIR
    llm_config: Path | None = None
    retrieval_config: Path = DEFAULT_RETRIEVAL_CONFIG_PATH
    readme: Path = REPO_ROOT / "README.md"
    docs: Path = REPO_ROOT / "docs"
    state: Path = DEFAULT_STATE_DIR

    @classmethod
    def dry_run(cls, root: Path) -> "Paths":
        """Everything under `root`, except the read-only inputs (config, questions,
        committed chunks used to build the synthetic filings, the real label file)."""
        return cls(
            raw=root / "raw",
            parsed=root / "parsed",
            chunks=root / "chunks",
            index=root / "index",
            retrieval_results=root / "eval-results",
            candidates=root / "candidates",
            labels=root / "labels.jsonl",  # never the real labels: IDs wouldn't match
            gold=root / "gold_labels.jsonl",
            results=root / "results",
            llm_config=None,
            readme=root / "README.md",
            docs=root / "docs",
            state=root / "state",
        )


@dataclass
class Options:
    dry_run: bool = False
    force: bool = False
    modes: tuple[str, ...] = MODES
    verifiers: tuple[str, ...] = DEFAULT_VERIFIERS
    # Retrieval: the variant from config/retrieval.yaml (None: its `default`), and model
    # overrides that keep the variant's stages (None: the variant's own models).
    variant: str | None = None
    compare: bool = False  # index every variant and run eval-retrieval --compare
    embedder: str | None = None
    reranker: str | None = None
    tokenizer: str = "bge-small"
    device: str | None = None
    generate_args: tuple[str, ...] = ()  # passed through to generate_candidates.py
    committed_chunks: Path = DEFAULT_CHUNKS_DIR  # dry run: source of the synthetic filings


@dataclass
class Stage:
    name: str
    inputs: Callable[["Context"], list[Path]]
    outputs: Callable[["Context"], list[Path]]
    run: Callable[["Context"], int]
    settings: Callable[["Context"], dict] = lambda ctx: {}


@dataclass
class Context:
    paths: Paths
    options: Options
    echo: Callable[[str], None] = print


@dataclass
class StageResult:
    name: str
    status: str  # ran, unchanged, nothing to do, failed, not run
    seconds: float = 0.0
    detail: str = ""


@dataclass
class PipelineResult:
    results: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status != "failed" for r in self.results)


# --- fingerprints --------------------------------------------------------------------


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and not p.name.endswith(".part"))
    return []


def fingerprint(stage: str, inputs: list[Path], settings: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"stage": stage, "version": STAGE_VERSION, **settings},
                             sort_keys=True, default=str).encode())  # fmt: skip
    for root in inputs:
        digest.update(f"\0root:{root}".encode())
        if not root.exists():
            digest.update(b"\0missing")
        for path in _files(root):
            digest.update(f"\0{path.relative_to(root) if root.is_dir() else path.name}".encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _stamp_path(ctx: Context, stage: str) -> Path:
    return ctx.paths.state / f"{stage}.json"


def is_unchanged(ctx: Context, stage: Stage, current: str) -> bool:
    stamp = _stamp_path(ctx, stage.name)
    if ctx.options.force or not stamp.is_file():
        return False
    try:
        stored = json.loads(stamp.read_text())
    except json.JSONDecodeError:
        return False
    return stored.get("fingerprint") == current and all(p.exists() for p in stage.outputs(ctx))


def store_stamp(ctx: Context, stage: Stage, current: str, seconds: float) -> None:
    ctx.paths.state.mkdir(parents=True, exist_ok=True)
    stamp = {
        "stage": stage.name,
        "fingerprint": current,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(seconds, 2),
        "outputs": [str(p) for p in stage.outputs(ctx)],
    }
    _stamp_path(ctx, stage.name).write_text(json.dumps(stamp, indent=2) + "\n")


# --- the stages ----------------------------------------------------------------------


def _cli(argv: list[str]) -> int:
    from groundcheck.cli import main

    try:
        return main(argv)
    except SystemExit as exc:  # argparse errors
        return exc.code if isinstance(exc.code, int) else 2


def run_download(ctx: Context) -> int:
    p = ctx.paths
    if not ctx.options.dry_run:
        return _cli(["download", "--config", str(p.config), "--raw-dir", str(p.raw)])
    from groundcheck.edgar.download import download_all
    from groundcheck.pipeline.fake import FakeSecClient

    companies = load_companies(p.config).companies
    client = FakeSecClient(companies, ctx.options.committed_chunks)
    results = download_all(client, companies, p.raw)
    ctx.echo(f"Dry run: wrote {sum(results.values())} synthetic filing(s) (no network).")
    return 0


def run_parse(ctx: Context) -> int:
    p = ctx.paths
    return _cli(["parse", "--config", str(p.config), "--raw-dir", str(p.raw),
                 "--parsed-dir", str(p.parsed)])  # fmt: skip


def _tokenizer(ctx: Context) -> str:
    return "none" if ctx.options.dry_run else ctx.options.tokenizer


def _variant(ctx: Context) -> str:
    """The variant the retrieval stages use: --variant, else the config's default."""
    return load_retrieval_config(ctx.paths.retrieval_config).get(ctx.options.variant)[0]


def _retrieval_args(ctx: Context, variant: str | None = None) -> list[str]:
    """--retrieval-config, --variant and the model overrides. A dry run swaps in the
    offline fakes (hashing embedder, overlap reranker) but keeps the variant's stages."""
    o = ctx.options
    args = ["--retrieval-config", str(ctx.paths.retrieval_config)]
    if variant or o.variant:
        args += ["--variant", variant or o.variant]
    embedder = "hashing" if o.dry_run else o.embedder
    reranker = "overlap" if o.dry_run else o.reranker
    if embedder:
        args += ["--embedder", embedder]
    if reranker:
        args += ["--reranker", reranker]
    return args


def _retrieval_settings(ctx: Context) -> dict:
    o = ctx.options
    return {
        "variant": "all" if o.compare else _variant(ctx),
        "embedder": "hashing" if o.dry_run else o.embedder,
        "reranker": "overlap" if o.dry_run else o.reranker,
    }


def _device(ctx: Context) -> list[str]:
    return ["--device", ctx.options.device] if ctx.options.device else []


def run_chunk(ctx: Context) -> int:
    p = ctx.paths
    return _cli(["chunk", "--config", str(p.config), "--parsed-dir", str(p.parsed),
                 "--chunks-dir", str(p.chunks), "--tokenizer", _tokenizer(ctx)])  # fmt: skip


def run_index(ctx: Context) -> int:
    p = ctx.paths
    # --compare evaluates every variant, so every variant's index is built (variants that
    # embed the same texts share one).
    variant = "all" if ctx.options.compare else None
    return _cli(["index", "--config", str(p.config), "--chunks-dir", str(p.chunks),
                 "--index-dir", str(p.index), *_retrieval_args(ctx, variant),
                 *_device(ctx)])  # fmt: skip


def run_eval_retrieval(ctx: Context) -> int:
    p = ctx.paths
    argv = ["eval-retrieval", "--questions", str(p.retrieval_questions),
            "--config", str(p.config), "--chunks-dir", str(p.chunks),
            "--index-dir", str(p.index), "--results-dir", str(p.retrieval_results),
            *_device(ctx)]  # fmt: skip
    if ctx.options.compare:
        if ctx.options.variant:
            raise ConfigError("--compare runs every configured variant; drop --variant")
        argv += [*_retrieval_args(ctx), "--compare"]
    else:
        argv += _retrieval_args(ctx)
    return _cli(argv)


def run_generate(ctx: Context) -> int:
    from groundcheck.generation import runner

    p = ctx.paths
    base = ["--config", str(p.config), "--questions", str(p.candidate_questions),
            "--retrieval-questions", str(p.retrieval_questions), "--chunks-dir", str(p.chunks),
            "--index-dir", str(p.index), "--out-dir", str(p.candidates),
            *_retrieval_args(ctx), *_device(ctx)]  # fmt: skip
    if p.llm_config is not None:
        base += ["--llm-config", str(p.llm_config)]
    if ctx.options.dry_run:
        base.append("--dry-run")
    worst = 0
    for mode in ctx.options.modes:
        ctx.echo(f"-- generate --mode {mode}")
        code = runner.main([*base, "--mode", mode, *ctx.options.generate_args])
        worst = max(worst, code)
    return worst


def run_export_labels(ctx: Context) -> int:
    p = ctx.paths
    if not p.labels.is_file():
        ctx.echo(f"No label file at {p.labels} yet: nothing to export (labels are never made up).")
        return NOTHING_TO_DO
    return _cli(["export-labels", "--candidates-dir", str(p.candidates), "--labels",
                 str(p.labels), "--out", str(p.gold)])  # fmt: skip


def _has_gold(ctx: Context) -> bool:
    return ctx.paths.gold.is_file() and bool(ctx.paths.gold.read_text().strip())


def _verifiers(ctx: Context) -> list[str]:
    return ["fake"] if ctx.options.dry_run else list(ctx.options.verifiers)


def run_eval_verifiers(ctx: Context) -> int:
    p = ctx.paths
    if not _has_gold(ctx):
        ctx.echo(f"No gold labels in {p.gold}: nothing to evaluate.")
        return NOTHING_TO_DO
    argv = ["eval-verifiers", "--verifier", *_verifiers(ctx), "--gold", str(p.gold),
            "--results-dir", str(p.results), *_device(ctx)]  # fmt: skip
    if ctx.options.dry_run:
        argv += ["--provider", "fake"]
    return _cli(argv)


def run_error_report(ctx: Context) -> int:
    p = ctx.paths
    if not (p.results / "predictions").is_dir() or not _has_gold(ctx):
        ctx.echo("No verifier predictions yet: nothing to report.")
        return NOTHING_TO_DO
    return _cli(["error-report", "--gold", str(p.gold), "--results-dir", str(p.results)])


def run_report(ctx: Context) -> int:
    p = ctx.paths
    return _cli(["report", "--results-dir", str(p.results), "--retrieval-results-dir",
                 str(p.retrieval_results), "--readme", str(p.readme), "--docs-dir",
                 str(p.docs), "--labels", str(p.labels), "--candidates-dir",
                 str(p.candidates)])  # fmt: skip


def build_stages() -> dict[str, Stage]:
    def companies(ctx):
        return [ctx.paths.config]

    stages = [
        Stage("download", companies, lambda c: [c.paths.raw], run_download,
              lambda c: {"dry_run": c.options.dry_run}),
        Stage("parse", lambda c: [c.paths.config, c.paths.raw], lambda c: [c.paths.parsed],
              run_parse),
        Stage("chunk", lambda c: [c.paths.config, c.paths.parsed], lambda c: [c.paths.chunks],
              run_chunk, lambda c: {"tokenizer": _tokenizer(c)}),
        Stage("index",
              lambda c: [c.paths.config, c.paths.chunks, c.paths.retrieval_config],
              lambda c: [c.paths.index], run_index, _retrieval_settings),
        Stage("eval-retrieval",
              lambda c: [c.paths.retrieval_questions, c.paths.chunks, c.paths.index,
                         c.paths.retrieval_config],
              lambda c: [c.paths.retrieval_results], run_eval_retrieval, _retrieval_settings),
        Stage("generate",
              lambda c: [c.paths.config, c.paths.candidate_questions,
                         c.paths.retrieval_questions, c.paths.chunks, c.paths.index,
                         c.paths.retrieval_config,
                         c.paths.llm_config or REPO_ROOT / "config" / "llm.yaml"],
              lambda c: [c.paths.candidates], run_generate,
              lambda c: {"modes": list(c.options.modes), "dry_run": c.options.dry_run,
                         "args": list(c.options.generate_args),
                         **_retrieval_settings(c), "variant": _variant(c)}),
        Stage("export-labels", lambda c: [c.paths.labels, c.paths.candidates],
              lambda c: [c.paths.gold], run_export_labels),
        Stage("eval-verifiers", lambda c: [c.paths.gold], lambda c: [c.paths.results],
              run_eval_verifiers, lambda c: {"verifiers": _verifiers(c)}),
        Stage("error-report", lambda c: [c.paths.gold, c.paths.results / "predictions"],
              lambda c: [c.paths.results / "error_report.md"], run_error_report),
        Stage("report",
              lambda c: [c.paths.results / "verifier_eval.json", c.paths.retrieval_results,
                         c.paths.labels, c.paths.candidates],
              lambda c: [c.paths.docs / "RESULTS.md"], run_report),
    ]  # fmt: skip
    return {s.name: s for s in stages}


def select(start: str | None, end: str | None) -> list[str]:
    for name in (start, end):
        if name is not None and name not in STAGES:
            raise ConfigError(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    first = STAGES.index(start) if start else 0
    last = STAGES.index(end) if end else len(STAGES) - 1
    if first > last:
        raise ConfigError(f"--from {start} comes after --to {end}")
    return list(STAGES[first : last + 1])


def run_pipeline(ctx: Context, start: str | None = None, end: str | None = None) -> PipelineResult:
    stages = build_stages()
    chosen = select(start, end)
    result = PipelineResult()
    failed = False
    for name in chosen:
        stage = stages[name]
        if failed:
            result.results.append(StageResult(name, "not run", detail="an earlier stage failed"))
            continue
        current = fingerprint(name, stage.inputs(ctx), stage.settings(ctx))
        if is_unchanged(ctx, stage, current):
            ctx.echo(f"== {name}: skipped (inputs unchanged since the last successful run)")
            result.results.append(StageResult(name, "unchanged"))
            continue
        ctx.echo(f"== {name}")
        started = time.perf_counter()
        code = stage.run(ctx)
        seconds = time.perf_counter() - started
        if code == 0:
            store_stamp(ctx, stage, current, seconds)
            result.results.append(StageResult(name, "ran", seconds))
        elif code == NOTHING_TO_DO:
            result.results.append(StageResult(name, "nothing to do", seconds))
        else:
            failed = True
            result.results.append(StageResult(name, "failed", seconds, f"exit code {code}"))
    return result


def format_summary(result: PipelineResult) -> str:
    lines = ["", "Pipeline summary:", f"  {'stage':<16} {'status':<14} {'seconds':>8}"]
    for r in result.results:
        detail = f"  {r.detail}" if r.detail else ""
        lines.append(f"  {r.name:<16} {r.status:<14} {r.seconds:>8.1f}{detail}")
    return "\n".join(lines)
