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
    load_llm_config,
)
from groundcheck.edgar.client import SecClient, SecRequestError
from groundcheck.edgar.download import DEFAULT_RAW_DIR, download_all, resolve_latest_10k
from groundcheck.generation.pipeline import DEFAULT_CANDIDATES_DIR
from groundcheck.labeling.report import check_labels, export_labels, label_stats
from groundcheck.labeling.session import Labeler
from groundcheck.labeling.store import (
    DEFAULT_EXPORT_PATH,
    DEFAULT_LABELS_PATH,
    LabelStore,
    claim_items,
    load_candidates,
)
from groundcheck.llm.base import LLMError, LLMRequest
from groundcheck.llm.factory import PROVIDERS, make_provider
from groundcheck.parsing.filing import DEFAULT_PARSED_DIR, load_parsed, parse_company
from groundcheck.retrieval.compare import write_comparison
from groundcheck.retrieval.embedder import (
    BGE_MAX_TOKENS,
    EMBEDDERS,
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
from groundcheck.retrieval.index import DEFAULT_INDEX_DIR
from groundcheck.retrieval.rerank import RERANKERS, make_reranker
from groundcheck.retrieval.retriever import build_variant_index, open_retriever
from groundcheck.retrieval.variants import (
    DEFAULT_RETRIEVAL_CONFIG_PATH,
    RetrievalConfig,
    VariantSpec,
    fiscal_years,
    load_retrieval_config,
    model_text,
)


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


def _retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return load_retrieval_config(args.retrieval_config)


def _variant(
    args: argparse.Namespace, config: RetrievalConfig, name: str | None = None
) -> tuple[str, VariantSpec]:
    """The chosen variant, with --embedder / --reranker overrides applied."""
    name, spec = config.get(name or args.variant)
    return name, spec.with_overrides(args.embedder, args.reranker)


def _open_retriever(args, config, name, spec, tickers, embedder=None, reranker=None):
    return open_retriever(
        name,
        spec,
        config,
        tickers,
        args.chunks_dir,
        args.index_dir,
        fiscal_years(load_companies(args.config)),
        embedder=embedder,
        reranker=reranker,
        device=args.device,
    )


def cmd_index(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    companies = [config.get(t) for t in args.only] if args.only else config.companies
    years = fiscal_years(config)
    retrieval = _retrieval_config(args)
    names = list(retrieval.variants) if args.variant == "all" else [_variant(args, retrieval)[0]]
    embedders: dict[str, object] = {}
    built: dict[tuple[str, bool], str] = {}
    for name in names:
        _, spec = _variant(args, retrieval, name)
        if spec.embedder is None:
            print(f"Variant {name}: no dense stage, so no index to build")
            continue
        if (spec.embedder, spec.headers) in built:
            print(f"Variant {name}: uses the same index as {built[spec.embedder, spec.headers]}")
            continue
        built[spec.embedder, spec.headers] = name
        if spec.embedder not in embedders:
            embedders[spec.embedder] = make_embedder(spec.embedder, args.device)
        embedder = embedders[spec.embedder]
        headers = " + contextual headers" if spec.headers else ""
        print(f"Variant {name}: embedder {embedder.name} (dim {embedder.dim}){headers}")
        for company in companies:
            if hasattr(embedder, "token_counts"):
                chunks = read_chunks(chunks_path(company.ticker, args.chunks_dir))
                texts = [model_text(c, spec.headers, years.get(company.ticker)) for c in chunks]
                counts = embedder.token_counts(texts)
                over = [
                    (c.chunk_id, n)
                    for c, n in zip(chunks, counts, strict=True)
                    if n > BGE_MAX_TOKENS
                ]
                print(f"  {company.ticker}: longest chunk {max(counts, default=0)} tokens")
                for chunk_id, n in over:
                    print(f"  WARNING: {chunk_id} has {n} tokens; truncated to {BGE_MAX_TOKENS}")
            n, folder = build_variant_index(
                name,
                spec,
                company.ticker,
                embedder,
                args.chunks_dir,
                args.index_dir,
                years.get(company.ticker),
            )
            print(f"  {company.ticker:<6} {n:>6} vectors -> {folder}")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    config = _retrieval_config(args)
    name, spec = _variant(args, config)
    retriever = _open_retriever(args, config, name, spec, [args.ticker])
    print(f"Variant {name}: {retriever.description}")
    for hit in retriever.search(args.ticker, args.question, args.k):
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

    config = _retrieval_config(args)
    if args.compare and args.variant:
        raise ConfigError("--compare runs every configured variant; drop --variant")
    names = list(config.variants) if args.compare else [config.get(args.variant)[0]]
    tickers = sorted(corpora)
    # Open every retriever first, so a missing index fails before any evaluation.
    # Models are shared between variants that use the same one.
    embedders: dict[str, object] = {}
    rerankers: dict[str, object] = {}
    retrievers = {}
    for name in names:
        _, spec = _variant(args, config, name)
        if spec.embedder and spec.embedder not in embedders:
            embedders[spec.embedder] = make_embedder(spec.embedder, args.device)
        if spec.rerank and spec.rerank not in rerankers:
            rerankers[spec.rerank] = make_reranker(spec.rerank, args.device)
        retrievers[name] = _open_retriever(
            args,
            config,
            name,
            spec,
            tickers,
            embedder=embedders.get(spec.embedder) if spec.embedder else None,
            reranker=rerankers.get(spec.rerank) if spec.rerank else None,
        )

    reports = {}
    for name, retriever in retrievers.items():
        for ticker in tickers:  # warm-up, untimed: first-call overhead (e.g. MPS kernels)
            retriever.search(ticker, "warm-up query", max(ks))

        def retrieve(ticker: str, question: str, k: int, retriever=retriever):
            return [hit.chunk for hit in retriever.search(ticker, question, k)]

        report = evaluate(questions, corpora, retrieve, retriever.description, ks, variant=name)
        reports[name] = report
        out = args.out if args.out and not args.compare else None
        out = out or args.results_dir / f"retrieval_{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2) + "\n")
        if not args.compare:
            print(format_report(report))
            print(f"Per-question results: {out}")
    if args.compare:
        diagnostic = {n for n in names if config.variants[n].diagnostic}
        table, recommendation, json_path, md_path = write_comparison(
            reports, list(ks), args.results_dir, diagnostic
        )
        print(table)
        print(f"Recommended default: {recommendation}")
        print(f"Comparison: {json_path} and {md_path}")
        print(f"Per-question results: {args.results_dir}/retrieval_<variant>.json")
    return 0


def cmd_llm_info(args: argparse.Namespace) -> int:
    config = load_llm_config(args.llm_config)
    provider = make_provider(
        config, args.provider, args.model, allow_paid=args.allow_paid, use_cache=False
    )
    identity = provider.identity()
    print(f"provider:        {identity.provider}")
    print(f"model:           {identity.model}")
    print(f"model digest:    {identity.model_digest}")
    print(f"runtime:         {identity.runtime} {identity.runtime_version}")
    print(f"temperature:     {config.temperature}, seed {config.seed}")
    if args.prompt:
        request = LLMRequest(
            prompt=args.prompt,
            temperature=config.temperature,
            seed=config.seed,
            max_tokens=min(config.max_tokens, 64),
        )
        response = provider.generate(request)
        print(f"reply:           {response.text.strip()[:200]}")
        print(f"tokens:          {response.metadata.completion_tokens}")
    return 0


def _candidates(args: argparse.Namespace):
    candidates = load_candidates(args.candidates_dir, args.company)
    if not candidates:
        where = f" for {', '.join(args.company)}" if args.company else ""
        raise ConfigError(
            f"no candidates{where} in {args.candidates_dir}; run scripts/generate_candidates.py"
        )
    return candidates


def cmd_label(args: argparse.Namespace) -> int:
    candidates = _candidates(args)
    store = LabelStore(args.labels)
    items = claim_items(candidates)
    done = sum(store.get(i.claim.claim_id) is not None for i in items)
    print(f"{len(items)} claim(s) in {len(candidates)} candidate(s); {done} already labeled.")
    print("Paste evidence exactly as shown; finish a paste with an empty line. q quits.")
    Labeler(store).run(items, only_unlabeled=args.only_unlabeled)
    return 0


def cmd_label_stats(args: argparse.Namespace) -> int:
    store = LabelStore(args.labels)
    candidates = load_candidates(args.candidates_dir, args.company)
    records = [r for r in store.records.values() if not args.company or r.ticker in args.company]
    print(label_stats(records, candidates))
    return 0


def cmd_label_check(args: argparse.Namespace) -> int:
    records, problems = check_labels(args.labels, load_candidates(args.candidates_dir))
    for problem in problems:
        print(f"ERROR: {problem}")
    status = "OK" if not problems else f"{len(problems)} problem(s)"
    print(f"{len(records)} valid label record(s) in {args.labels}; {status}")
    return 1 if problems else 0


def cmd_export_labels(args: argparse.Namespace) -> int:
    candidates = load_candidates(args.candidates_dir)
    records, problems = check_labels(args.labels, candidates)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        print("Nothing exported: fix these first (`groundcheck label-check`).")
        return 1
    n = export_labels(records, candidates, args.out)
    print(
        f"Exported {n} confirmed label(s) to {args.out}; "
        f"{len(records) - n} unconfirmed label(s) left out."
    )
    return 0


def add_llm_options(parser: argparse.ArgumentParser) -> None:
    """--llm-config, --provider, --model and --allow-paid, shared with the scripts."""
    parser.add_argument("--llm-config", type=Path, default=None, help="default: config/llm.yaml")
    parser.add_argument(
        "--provider", choices=sorted(PROVIDERS), default=None, help="default: from config"
    )
    parser.add_argument(
        "--model", default=None, help="model name (default: from config, llama3.2:3b)"
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="allow a provider that costs money (refused by default: $0 budget)",
    )


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

    # Options shared by the commands that embed text or retrieve.
    embed_options = argparse.ArgumentParser(add_help=False)
    embed_options.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    embed_options.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    embed_options.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    embed_options.add_argument(
        "--retrieval-config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
        help="retrieval variants (default: config/retrieval.yaml)",
    )
    embed_options.add_argument(
        "--variant", default=None, help="retrieval variant (default: `default` in the config)"
    )
    embed_options.add_argument(
        "--embedder",
        choices=list(EMBEDDERS),
        default=None,
        help="override the variant's embedder (hashing: offline fake, for smoke runs)",
    )
    embed_options.add_argument(
        "--reranker",
        choices=list(RERANKERS),
        default=None,
        help="override the variant's reranker (overlap: offline fake)",
    )
    embed_options.add_argument(
        "--device", default=None, help="cpu or mps (default: mps if available, else cpu)"
    )

    index = subparsers.add_parser(
        "index",
        parents=[embed_options],
        help="build one FAISS index per company for a retrieval variant (or --variant all)",
    )
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
        help="recall@k, section-hit@k, MRR and latency against the gold evidence "
        "(one variant, or --compare for all)",
    )
    evaluation.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    evaluation.add_argument("-k", type=int, nargs="+", default=list(DEFAULT_KS))
    evaluation.add_argument(
        "--results-dir", type=Path, default=DEFAULT_QUESTIONS_PATH.parent / "results"
    )
    evaluation.add_argument(
        "--out", type=Path, default=None, help="report JSON path (single variant only)"
    )
    evaluation.add_argument(
        "--compare",
        action="store_true",
        help="run every variant in the retrieval config and print one comparison table "
        "(also written to <results-dir>/compare.json and compare.md)",
    )
    evaluation.add_argument(
        "--check",
        action="store_true",
        help="only check that every gold string occurs in the chunks (no model needed)",
    )
    evaluation.set_defaults(func=cmd_eval_retrieval)

    llm_info = subparsers.add_parser(
        "llm-info",
        help="show the LLM provider, model digest and runtime version (and test a prompt)",
    )
    add_llm_options(llm_info)
    llm_info.add_argument(
        "--prompt", default="Reply with the single word: ready", help="'' to skip the test call"
    )
    llm_info.set_defaults(func=cmd_llm_info)

    # Options shared by the labeling commands.
    label_options = argparse.ArgumentParser(add_help=False)
    label_options.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    label_options.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)

    label = subparsers.add_parser(
        "label", parents=[label_options], help="label claims interactively (autosaves)"
    )
    label.add_argument("--company", nargs="+", metavar="TICKER", help="limit to tickers")
    label.add_argument(
        "--only-unlabeled", action="store_true", help="skip claims that already have a label"
    )
    label.set_defaults(func=cmd_label)

    stats = subparsers.add_parser(
        "label-stats",
        parents=[label_options],
        help="label counts by label, company, question type and edge case",
    )
    stats.add_argument("--company", nargs="+", metavar="TICKER", help="limit to tickers")
    stats.set_defaults(func=cmd_label_stats)

    check = subparsers.add_parser(
        "label-check", parents=[label_options], help="validate every label record"
    )
    check.set_defaults(func=cmd_label_check)

    export = subparsers.add_parser(
        "export-labels",
        parents=[label_options],
        help="write confirmed labels only, as clean JSONL under data/eval/",
    )
    export.add_argument("--out", type=Path, default=DEFAULT_EXPORT_PATH)
    export.set_defaults(func=cmd_export_labels)
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
    except (ConfigError, SecRequestError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
