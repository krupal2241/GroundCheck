"""Command line for `scripts/generate_candidates.py` (kept in the package so it is tested).

    python scripts/generate_candidates.py --company AAPL --batch-size 5 --limit 10
    python scripts/generate_candidates.py --dry-run          # FakeProvider, no model
    python scripts/generate_candidates.py --mode no-context  # closed-book answers
    python scripts/generate_candidates.py --mode wrong-context --wrong-context-source mixed

`--dry-run` uses the FakeProvider and writes to `data/cache/dry-run/candidates/`
(gitignored), so a rehearsal never mixes into the committed candidates.
"""

import argparse
import logging
import sys
from pathlib import Path

from groundcheck.chunking.base import DEFAULT_CHUNKS_DIR, chunks_path
from groundcheck.cli import add_llm_options
from groundcheck.config import (
    DEFAULT_COMPANIES_PATH,
    ConfigError,
    load_companies,
    load_llm_config,
)
from groundcheck.generation.modes import MODES, WRONG_SOURCES, WrongContextPicker
from groundcheck.generation.pipeline import (
    DEFAULT_CANDIDATES_DIR,
    DRY_RUN_CANDIDATES_DIR,
    Generator,
    RetrievalInfo,
    run,
)
from groundcheck.generation.prompts import fake_responder
from groundcheck.generation.questions import (
    DEFAULT_CANDIDATE_QUESTIONS_PATH,
    collect_questions,
    ensure_template,
)
from groundcheck.llm.base import LLMError
from groundcheck.llm.factory import make_provider
from groundcheck.retrieval.embedder import make_embedder
from groundcheck.retrieval.evaluate import DEFAULT_QUESTIONS_PATH
from groundcheck.retrieval.index import DEFAULT_INDEX_DIR, CompanyIndex, file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_candidates.py",
        description="Retrieve, answer and split into claims; append to data/candidates/.",
    )
    parser.add_argument("--company", nargs="+", metavar="TICKER", help="limit to tickers")
    parser.add_argument("--batch-size", type=int, default=10, help="questions per batch")
    parser.add_argument("--limit", type=int, default=None, help="at most N new questions")
    parser.add_argument(
        "--sleep-between-batches",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="pause between batches (lets a fanless laptop cool down)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the FakeProvider (no model) and write under data/cache/dry-run/",
    )
    add_llm_options(parser)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="rag",
        help="rag: answer from the retrieved passages; no-context: closed-book; "
        "wrong-context: answer from passages of another company or another Item",
    )
    parser.add_argument(
        "--wrong-context-source",
        choices=WRONG_SOURCES,
        default="mixed",
        help="wrong-context only: other-company, other-item, or mixed (per question)",
    )
    parser.add_argument("--no-cache", action="store_true", help="don't use data/cache/llm")
    parser.add_argument("--top-k", type=int, default=None, help="default: from config (5)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_CANDIDATE_QUESTIONS_PATH)
    parser.add_argument("--retrieval-questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: data/candidates")
    parser.add_argument("--embedder", choices=["bge-small", "hashing"], default="bge-small")
    parser.add_argument("--device", default=None, help="embedder device: cpu, mps or cuda")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.batch_size < 1 or (args.limit is not None and args.limit < 0):
        raise ConfigError("--batch-size must be at least 1 and --limit non-negative")
    companies = load_companies(args.config)
    llm_config = load_llm_config(args.llm_config)
    if args.top_k is not None:
        llm_config = llm_config.model_copy(update={"top_k": args.top_k})
    if ensure_template(args.questions, companies):
        print(f"Created the question template {args.questions}; fill in the questions.")
    questions = collect_questions(
        companies, args.questions, args.retrieval_questions, tickers=args.company
    )
    if not questions:
        print(
            f"No questions with text yet in {args.questions} or {args.retrieval_questions}; "
            "nothing to generate."
        )
        return 1

    if args.dry_run:
        provider = make_provider(
            llm_config, "fake", args.model, use_cache=False, responder=fake_responder
        )
    else:
        provider = make_provider(
            llm_config,
            args.provider,
            args.model,
            allow_paid=args.allow_paid,
            use_cache=not args.no_cache,
        )
    out_dir = args.out_dir or (DRY_RUN_CANDIDATES_DIR if args.dry_run else DEFAULT_CANDIDATES_DIR)

    # Load every index first, so a missing one fails before any model call.
    tickers = sorted({q.ticker for q in questions})
    indexes = {t: CompanyIndex.load(t, args.chunks_dir, args.index_dir) for t in tickers}
    picker = None
    if args.mode == "wrong-context":
        # Other companies' indexes too (any role), for "another company's passages".
        for company in companies.companies:
            folder = args.index_dir / company.ticker
            if company.ticker not in indexes and (folder / "meta.json").is_file():
                indexes[company.ticker] = CompanyIndex.load(
                    company.ticker, args.chunks_dir, args.index_dir
                )
    embedder = make_embedder(args.embedder, args.device)
    infos = {
        t: RetrievalInfo(
            embedder=embedder.name,
            chunker=indexes[t].chunks[0].chunker,
            top_k=llm_config.top_k,
            chunks_sha256=file_sha256(chunks_path(t, args.chunks_dir)),
        )
        for t in tickers
    }

    def retrieve(ticker: str, question: str, k: int):
        return indexes[ticker].search(embedder, question, k)

    if args.mode == "wrong-context":
        picker = WrongContextPicker(retrieve, sorted(indexes), args.wrong_context_source)

    identity = provider.identity()
    print(
        f"Model: {identity.provider} {identity.model} (digest {identity.model_digest[:19]}, "
        f"{identity.runtime} {identity.runtime_version}); embedder {embedder.name}, "
        f"top_k {llm_config.top_k}; mode {args.mode}; output {out_dir}"
    )
    names = {c.ticker: c.name for c in companies.companies}
    generator = Generator(
        provider, retrieve, infos.__getitem__, llm_config, names, args.mode, picker
    )
    summary = run(
        generator,
        questions,
        out_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        sleep_between_batches=args.sleep_between_batches,
    )
    print(
        f"Wrote {summary.written} candidate(s) with {summary.claims} claim(s); "
        f"{summary.already_done} already done; {len(summary.failed)} failed."
    )
    if summary.claims_rejected or summary.claim_retries:
        print(
            f"Claim validator: {summary.claims_rejected} claim(s) rejected, "
            f"{summary.claim_retries} retry request(s) (details in each record's claim_filter)"
        )
    hits = getattr(provider, "hits", None)
    if hits is not None:
        print(f"LLM cache: {hits} hit(s), {provider.misses} miss(es)")
    return 1 if summary.failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return _run(args)
    except (ConfigError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
