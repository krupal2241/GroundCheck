"""Command-line entry point: `groundcheck <subcommand>`.

This file only parses arguments and calls into the real modules. Keeping the
logic out of the CLI means every step can also be called from Python (tests,
notebooks, scripts) without going through argparse.

Subcommands are added as each module is built.
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from groundcheck import __version__
from groundcheck.chunking.base import (
    DEFAULT_CHUNKS_DIR,
    chunks_path,
    is_split_part,
    read_chunks,
    write_chunks,
)
from groundcheck.chunking.fixed import DEFAULT_MAX_TOKENS, FixedSizeChunker
from groundcheck.config import (
    DEFAULT_COMPANIES_PATH,
    ConfigError,
    get_sec_user_agent,
    load_companies,
)
from groundcheck.edgar.client import SecClient, SecRequestError
from groundcheck.edgar.download import DEFAULT_RAW_DIR, download_all, resolve_latest_10k
from groundcheck.parsing.filing import DEFAULT_PARSED_DIR, load_parsed, parse_company
from groundcheck.retrieval.embedder import (
    BGE_MAX_TOKENS,
    load_bge_token_counter,
    make_embedder,
)
from groundcheck.retrieval.evaluate import (
    DEFAULT_KS,
    DEFAULT_QUESTIONS_PATH,
    check_questions,
    evaluate,
    format_report,
    load_questions,
)
from groundcheck.retrieval.index import DEFAULT_INDEX_DIR, CompanyIndex, build_company_index


def cmd_resolve_filings(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    client = SecClient(get_sec_user_agent(args.env_file))
    companies = [config.get(t) for t in args.only] if args.only else config.companies

    print("# Paste these values into config/companies.yaml.")
    print("# First check that each 'SEC name' matches the company you meant.\n")
    for company in companies:
        info = resolve_latest_10k(client, company, args.raw_dir, refresh=args.refresh)
        print(f"# {company.ticker}: SEC name = {info.sec_name!r}, period = {info.report_date}")
        print(f"  - ticker: {company.ticker}")
        print(f"    accession: {info.accession}")
        print(f"    primary_document: {info.primary_document}")
        print(f'    filing_date: "{info.filing_date}"')
        print(f'    fiscal_year_end: "{info.fiscal_year_end}"')
        print(f'    period: "{info.report_date}"\n')
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    companies = [config.get(t) for t in args.only] if args.only else config.companies
    client = SecClient(get_sec_user_agent(args.env_file))
    results = download_all(client, companies, args.raw_dir)
    fetched = sum(results.values())
    print(f"Downloaded {fetched} filing(s); {len(results) - fetched} already cached.")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    companies = [config.get(t) for t in args.only] if args.only else config.companies
    total_warnings = 0
    for company in companies:
        parsed = parse_company(company, args.raw_dir, args.parsed_dir)
        print(
            f"{company.ticker}  {parsed.accession}  page-furniture blocks removed: "
            f"{parsed.removed_furniture_blocks}, repeated header lines removed: "
            f"{parsed.removed_header_lines}, front matter: {parsed.front_matter_chars:,} chars"
        )
        for section in parsed.sections:
            label = (
                "Items " + ", ".join(section.covers) if section.covers else f"Item {section.item}"
            )
            words = len(section.text.split())
            marker = "  recovered=true" if section.recovered else ""
            print(
                f"  {label:<14} {section.title[:48]:<48} {len(section.text):>9,} chars"
                f" {words:>8,} words{marker}"
            )
        for message in parsed.notes:
            print(f"  NOTE: {message}")
        for message in parsed.warnings:
            print(f"  WARNING: {message}")
        total_warnings += len(parsed.warnings)
        print()
    print(f"Parsed {len(companies)} filing(s) into {args.parsed_dir}; {total_warnings} warning(s).")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    companies = [config.get(t) for t in args.only] if args.only else config.companies
    try:
        settings = (args.prose_words, args.overlap, args.table_words, args.max_tokens)
        chunker = FixedSizeChunker(*settings)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    count_tokens = load_bge_token_counter() if args.tokenizer == "bge-small" else None
    chunker = dataclasses.replace(chunker, count_tokens=count_tokens)
    limit = f", max {chunker.max_tokens} tokens" if count_tokens else ", no token limit"
    print(f"Chunker: {chunker.name}{limit}")
    for company in companies:
        chunks = chunker.chunk(load_parsed(company.ticker, args.parsed_dir))
        path = chunks_path(company.ticker, args.chunks_dir)
        write_chunks(chunks, path)
        prose = [c for c in chunks if c.kind == "prose"]
        tables = [c for c in chunks if c.kind == "table"]
        longest = max((c.word_count for c in chunks), default=0)
        tokens = ""
        if count_tokens:
            counts = [count_tokens(c.text) for c in chunks]
            parts = sum(is_split_part(c.chunk_id) for c in chunks)
            tokens = f", {max(counts, default=0)} tokens; {parts} split part(s)"
        print(
            f"  {company.ticker:<6} {len(prose):>5} prose + {len(tables):>5} table chunks, "
            f"longest {longest} words{tokens} -> {path}"
        )
        if count_tokens:
            for chunk, n in zip(chunks, counts, strict=True):
                if n > chunker.max_tokens:  # only a single unsplittable "word" gets here
                    print(f"  WARNING: {chunk.chunk_id} still has {n} tokens")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    companies = [config.get(t) for t in args.only] if args.only else config.companies
    embedder = make_embedder(args.embedder, args.device)
    print(f"Embedder: {embedder.name} (dim {embedder.dim})")
    for company in companies:
        if hasattr(embedder, "token_counts"):
            chunks = read_chunks(chunks_path(company.ticker, args.chunks_dir))
            counts = embedder.token_counts([c.text for c in chunks])
            over = [
                (c.chunk_id, n) for c, n in zip(chunks, counts, strict=True) if n > BGE_MAX_TOKENS
            ]
            print(f"  {company.ticker}: longest chunk {max(counts, default=0)} tokens")
            for chunk_id, n in over:
                print(f"  WARNING: {chunk_id} has {n} tokens; truncated to {BGE_MAX_TOKENS}")
        n = build_company_index(company.ticker, embedder, args.chunks_dir, args.index_dir)
        print(f"  {company.ticker:<6} {n:>6} vectors -> {args.index_dir / company.ticker}")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    embedder = make_embedder(args.embedder, args.device)
    index = CompanyIndex.load(args.ticker, args.chunks_dir, args.index_dir)
    for hit in index.search(embedder, args.question, args.k):
        preview = " ".join(hit.chunk.text.split())[:160]
        print(f"{hit.rank:>2}. {hit.score:.3f}  {hit.chunk.chunk_id}  {preview}")
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    ks = tuple(sorted(set(args.k)))
    if ks[0] < 1:
        raise ConfigError("-k values must be positive")
    questions = load_questions(args.questions).questions
    filled = [q for q in questions if q.is_filled]
    print(f"{len(filled)} of {len(questions)} question(s) filled in {args.questions}")
    tickers = sorted({q.ticker for q in filled})
    corpora = {
        t: read_chunks(chunks_path(t, args.chunks_dir))
        for t in tickers
        if chunks_path(t, args.chunks_dir).is_file()
    }
    problems = check_questions(questions, corpora)
    for problem in problems:
        print(f"WARNING: {problem}")
    if args.check:
        print("Gold evidence check: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
        return 1 if problems else 0
    if not filled:
        print("Nothing to evaluate yet: fill in questions and gold_evidence first.")
        return 1

    embedder = make_embedder(args.embedder, args.device)
    indexes = {t: CompanyIndex.load(t, args.chunks_dir, args.index_dir) for t in corpora}

    def retrieve(ticker: str, question: str, k: int):
        return [hit.chunk for hit in indexes[ticker].search(embedder, question, k)]

    report = evaluate(questions, corpora, retrieve, embedder.name, ks)
    print(format_report(report))
    out = args.out or args.results_dir / f"retrieval_{embedder.name.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n")
    print(f"Per-question results: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundcheck",
        description="Claim-level hallucination verification for RAG over SEC 10-K filings.",
    )
    parser.add_argument("--version", action="version", version=f"groundcheck {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO log messages")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Options shared by the EDGAR commands.
    edgar_options = argparse.ArgumentParser(add_help=False)
    edgar_options.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    edgar_options.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    edgar_options.add_argument("--only", nargs="+", metavar="TICKER", help="limit to tickers")
    edgar_options.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="read SEC_USER_AGENT from this file (default: ./.env in the current directory)",
    )

    resolve = subparsers.add_parser(
        "resolve-filings",
        parents=[edgar_options],
        help="print each company's latest 10-K so you can pin it in config",
    )
    resolve.add_argument("--refresh", action="store_true", help="re-fetch SEC's submissions index")
    resolve.set_defaults(func=cmd_resolve_filings)

    download = subparsers.add_parser(
        "download", parents=[edgar_options], help="download the pinned 10-K filings"
    )
    download.set_defaults(func=cmd_download)

    parse = subparsers.add_parser(
        "parse", help="split downloaded filings into Item sections (data/raw -> data/parsed)"
    )
    parse.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    parse.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parse.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parse.add_argument("--only", nargs="+", metavar="TICKER", help="limit to tickers")
    parse.set_defaults(func=cmd_parse)

    chunk = subparsers.add_parser(
        "chunk", help="chunk parsed sections (data/parsed -> data/chunks/<TICKER>.jsonl)"
    )
    chunk.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    chunk.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    chunk.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    chunk.add_argument("--only", nargs="+", metavar="TICKER", help="limit to tickers")
    chunk.add_argument("--prose-words", type=int, default=250)
    chunk.add_argument("--overlap", type=int, default=50)
    chunk.add_argument("--table-words", type=int, default=120)
    chunk.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"split any chunk over this many tokens (default {DEFAULT_MAX_TOKENS})",
    )
    chunk.add_argument(
        "--tokenizer",
        choices=["bge-small", "none"],
        default="bge-small",
        help="count tokens with bge-small's tokenizer (needs the [embed] extra), or "
        "none to skip the token limit",
    )
    chunk.set_defaults(func=cmd_chunk)

    # Options shared by the commands that embed text.
    embed_options = argparse.ArgumentParser(add_help=False)
    embed_options.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    embed_options.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    embed_options.add_argument(
        "--embedder",
        choices=["bge-small", "hashing"],
        default="bge-small",
        help="bge-small (real model, needs the [embed] extra) or hashing (offline fake)",
    )
    embed_options.add_argument(
        "--device", default=None, help="cpu or mps (default: mps if available, else cpu)"
    )

    index = subparsers.add_parser(
        "index", parents=[embed_options], help="build one FAISS index per company from chunks"
    )
    index.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    index.add_argument("--only", nargs="+", metavar="TICKER", help="limit to tickers")
    index.set_defaults(func=cmd_index)

    retrieve = subparsers.add_parser(
        "retrieve", parents=[embed_options], help="show the top-k chunks for a question"
    )
    retrieve.add_argument("ticker")
    retrieve.add_argument("question")
    retrieve.add_argument("-k", type=int, default=5)
    retrieve.set_defaults(func=cmd_retrieve)

    evaluation = subparsers.add_parser(
        "eval-retrieval",
        parents=[embed_options],
        help="recall@k and section-hit@k against the gold evidence in the question file",
    )
    evaluation.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    evaluation.add_argument("-k", type=int, nargs="+", default=list(DEFAULT_KS))
    evaluation.add_argument(
        "--results-dir", type=Path, default=DEFAULT_QUESTIONS_PATH.parent / "results"
    )
    evaluation.add_argument("--out", type=Path, default=None, help="report JSON path")
    evaluation.add_argument(
        "--check",
        action="store_true",
        help="only check that every gold string occurs in the chunks (no model needed)",
    )
    evaluation.set_defaults(func=cmd_eval_retrieval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except (ConfigError, SecRequestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
