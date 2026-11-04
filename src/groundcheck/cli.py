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
from groundcheck.labeling.plan import DEFAULT_PLAN_PATH, balance_table, load_plan, make_plan
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
from groundcheck.training.data import (
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_TRAINING_DIR,
    build_training_data,
    load_training_config,
)
from groundcheck.verify.base import VerifierError
from groundcheck.verify.costs import DEFAULT_PRICES_PATH, load_prices
from groundcheck.verify.errors import error_report
from groundcheck.verify.evaluate import DEFAULT_RESULTS_DIR, evaluate_verifiers, load_gold
from groundcheck.verify.factory import VERIFIER_NAMES, VerifierOptions, make_verifier


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
    if args.plan:
        ids = load_plan(args.plan)
        by_id = {c.candidate_id: c for c in candidates}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ConfigError(
                f"{len(missing)} planned candidate(s) not in {args.candidates_dir} "
                f"(first: {missing[0]}); was the plan made from other candidates?"
            )
        candidates = [by_id[i] for i in ids]
        print(f"Plan {args.plan}: {len(candidates)} candidate(s), in plan order.")
    items = claim_items(candidates)
    done = sum(store.get(i.claim.claim_id) is not None for i in items)
    print(f"{len(items)} claim(s) in {len(candidates)} candidate(s); {done} already labeled.")
    print("Paste evidence exactly as shown; finish a paste with an empty line. q quits.")
    suggester = None
    if args.suggest:
        from groundcheck.labeling.assist import verifier_suggester

        # Local models only: allow_paid is never passed through here.
        options = VerifierOptions(
            llm_config=args.llm_config,
            allow_paid=False,
            device=args.device,
            finetuned_path=args.finetuned_path,
        )
        suggester = verifier_suggester(make_verifier(args.suggest, options))
        print(
            f"Label suggestions ON ({args.suggest}). Each label records what was suggested "
            "and whether you agreed; see docs/labeling_guide.md on anchoring."
        )
    labeler = Labeler(
        store, suggest_evidence=not args.no_evidence_suggest, label_suggester=suggester
    )
    labeler.run(items, only_unlabeled=args.only_unlabeled)
    return 0


def cmd_label_plan(args: argparse.Namespace) -> int:
    candidates = _candidates(args)
    labeled = set(LabelStore(args.labels).records) if args.labels.is_file() else set()
    plan = make_plan(candidates, args.questions, args.claims, args.seed, labeled)
    by_id = {c.candidate_id: c for c in candidates}
    chosen = [by_id[i] for i in plan.candidate_ids]
    n_claims = sum(len(c.claims) for c in chosen)
    target = ", ".join(f"{k} {v}" for k, v in plan.target.items())
    print(
        f"Plan: {len(chosen)} candidate(s), {n_claims} claim(s) (target: {target}; "
        f"{len(labeled)} claim(s) already labeled count towards the balance)"
    )
    if not plan.reached:
        print("WARNING: not enough unlabeled candidates with claims to reach the target.")
    print(balance_table(chosen, "claims"))
    if args.dry_run:
        print("--dry-run: plan not written.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(plan.to_json(candidates))
    print(f"Wrote {args.out}; label it with: groundcheck label --plan {args.out}")
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
    invalid = sum(r.invalid for r in records)
    print(
        f"{len(records)} valid label record(s) in {args.labels} ({invalid} marked invalid); "
        f"{status}"
    )
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
    invalid = sum(r.invalid for r in records)
    print(
        f"Exported {n} confirmed label(s) to {args.out}; "
        f"{len(records) - n - invalid} unconfirmed label(s) and {invalid} invalid claim(s) "
        "left out."
    )
    return 0


def verifier_options(args: argparse.Namespace) -> VerifierOptions:
    return VerifierOptions(
        llm_config=args.llm_config,
        provider=args.provider,
        llm_model=args.model,
        allow_paid=args.allow_paid,
        use_cache=args.use_cache,
        model=args.hf_model,
        revision=args.revision,
        device=args.device,
        threshold=args.threshold,
        finetuned_path=args.finetuned_path,
    )


def cmd_eval_verifiers(args: argparse.Namespace) -> int:
    gold = load_gold(args.gold)
    if args.company:
        gold = [g for g in gold if g.ticker in args.company]
    if args.limit is not None:
        gold = gold[: args.limit]
    if not gold:
        raise ConfigError("no gold labels left after --company/--limit")
    if args.hf_model and len(args.verifier) > 1:
        raise ConfigError("--hf-model applies to one verifier; run them one at a time")
    options = verifier_options(args)
    prices = load_prices(args.prices)
    print(f"{len(gold)} gold claim(s) from {args.gold}")
    evaluate_verifiers(
        args.verifier,
        lambda name: make_verifier(name, options),
        gold,
        args.gold,
        args.results_dir,
        prices,
    )
    print(f"Wrote {args.results_dir / 'verifier_eval.json'} and verifier_eval.md")
    print(f"Per-claim predictions: {args.results_dir / 'predictions'}/")
    return 0


def cmd_error_report(args: argparse.Namespace) -> int:
    gold = load_gold(args.gold)
    text = error_report(args.results_dir, gold, args.gold, args.verifier)
    out = args.out or args.results_dir / "error_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out}")
    return 0


DEMO_VERIFIERS = ["prompted", "hhem", "minicheck", "fake"]


def cmd_demo(args: argparse.Namespace) -> int:
    from groundcheck.demo.app import DemoPipeline, build_app
    from groundcheck.generation.pipeline import RetrievalInfo
    from groundcheck.generation.prompts import fake_responder
    from groundcheck.retrieval.index import file_sha256

    companies = load_companies(args.config)
    tickers = args.company or [
        c.ticker for c in companies.companies if (args.index_dir / c.ticker / "meta.json").is_file()
    ]
    if not tickers:
        raise ConfigError(f"no company has an index in {args.index_dir}; run `groundcheck index`")
    verifiers = args.verifiers or (["fake"] if args.dry_run else list(DEMO_VERIFIERS))
    if args.finetuned_path and "finetuned" not in verifiers:
        verifiers.append("finetuned")
    llm_config = load_llm_config(args.llm_config)
    if args.dry_run:
        provider = make_provider(llm_config, "fake", use_cache=False, responder=fake_responder)
    else:
        provider = make_provider(llm_config, args.provider, args.model, allow_paid=args.allow_paid)
    embedder = make_embedder(args.embedder, args.device)
    indexes: dict[str, CompanyIndex] = {}

    def index(ticker: str) -> CompanyIndex:
        if ticker not in indexes:
            indexes[ticker] = CompanyIndex.load(ticker, args.chunks_dir, args.index_dir)
        return indexes[ticker]

    def retrieve(ticker: str, question: str, k: int):
        return index(ticker).search(embedder, question, k)

    def retrieval_info(ticker: str) -> RetrievalInfo:
        return RetrievalInfo(
            embedder=embedder.name,
            chunker=index(ticker).chunks[0].chunker,
            top_k=llm_config.top_k,
            chunks_sha256=file_sha256(chunks_path(ticker, args.chunks_dir)),
        )

    options = verifier_options(args)
    if args.dry_run:
        options.provider = "fake"
    pipeline = DemoPipeline(
        provider, retrieve, retrieval_info, llm_config, lambda n: make_verifier(n, options)
    )
    app = build_app(pipeline, tickers, verifiers)
    identity = provider.identity()
    print(
        f"GroundCheck demo: {identity.provider} {identity.model}, embedder {embedder.name}, "
        f"companies {', '.join(tickers)}, verifiers {', '.join(verifiers)}"
    )
    print(f"Open http://{args.host}:{args.port} (Ctrl-C to stop)")
    app.launch(server_name=args.host, server_port=args.port, share=False)
    return 0


MIN_TRAINING_EXAMPLES = 500


def cmd_build_training_data(args: argparse.Namespace) -> int:
    config = load_training_config(args.training_config)
    gold_paths = [args.gold, args.labels]
    manifest = build_training_data(
        args.chunks_dir, args.out_dir, config, gold_paths, args.candidates_dir
    )
    for problem in manifest["companies_in_config_without_chunks"]:
        print(f"WARNING: {problem} is in {args.training_config or 'config/training.yaml'} "
              "but has no chunks")  # fmt: skip
    if not (manifest["gold_claims_excluded"] or manifest["candidate_chunks_excluded"]):
        print(
            "WARNING: found no gold labels, label file or candidates to exclude. That is "
            "only right if nothing has been labeled yet; otherwise check --gold, --labels "
            "and --candidates-dir before training."
        )
    print(
        f"Excluded {manifest['gold_claims_excluded']} gold claim(s), "
        f"{manifest['gold_chunks_excluded']} gold chunk(s) and "
        f"{manifest['candidate_chunks_excluded']} other candidate chunk(s)."
    )
    for split, counts in manifest["counts_by_split_and_label"].items():
        total = sum(counts.values())
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        print(f"  {split:<5} {total:>6} example(s)  {detail}")
    print(f"By rule: {manifest['counts_by_rule']}")
    n_train = sum(manifest["counts_by_split_and_label"].get("train", {}).values())
    if n_train < MIN_TRAINING_EXAMPLES:
        print(
            f"WARNING: only {n_train} training example(s) (fewer than "
            f"{MIN_TRAINING_EXAMPLES}). A verifier fine-tuned on this few will be weak; "
            "add companies to the train split or raise max_seeds_per_company."
        )
    print(f"Wrote {args.out_dir}/{{train,dev,test}}.jsonl and manifest.json")
    return 0


def add_verifier_options(parser: argparse.ArgumentParser) -> None:
    """Options for building verifiers (and the prompted verifier's LLM)."""
    add_llm_options(parser)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="prompted: reuse cached LLM replies (off by default: latency would be meaningless)",
    )
    parser.add_argument(
        "--hf-model", default=None, help="minicheck/hhem: a different Hugging Face model ID"
    )
    parser.add_argument("--revision", default=None, help="minicheck/hhem: pin a Hub commit")
    parser.add_argument(
        "--device", default=None, help="cpu, mps or cuda (default: cuda, else mps, else cpu)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="binary verifiers: P(supported) needed to say supported (default 0.5)",
    )
    parser.add_argument(
        "--finetuned-path",
        type=Path,
        default=None,
        help="finetuned: the model directory written by scripts/train_verifier.py",
    )


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
    label.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="walk through exactly the candidates of a plan from `groundcheck label-plan`",
    )
    label.add_argument(
        "--suggest",
        choices=VERIFIER_NAMES,
        default=None,
        help="show this local verifier's proposed label before you decide (OFF by "
        "default: a suggestion can anchor you; every label records whether one was shown)",
    )
    label.add_argument(
        "--no-evidence-suggest",
        action="store_true",
        help="don't offer the best-matching sentence or table row as evidence",
    )
    label.add_argument("--llm-config", type=Path, default=None, help="--suggest prompted")
    label.add_argument("--device", default=None, help="--suggest minicheck/hhem/finetuned")
    label.add_argument("--finetuned-path", type=Path, default=None, help="--suggest finetuned")
    label.set_defaults(func=cmd_label)

    plan = subparsers.add_parser(
        "label-plan",
        parents=[label_options],
        help="propose a sample balanced across companies, question types and modes",
    )
    target = plan.add_mutually_exclusive_group(required=True)
    target.add_argument("--questions", type=int, help="number of candidate answers to plan")
    target.add_argument("--claims", type=int, help="number of claims to plan")
    plan.add_argument("--company", nargs="+", metavar="TICKER", help="limit to tickers")
    plan.add_argument("--seed", type=int, default=0, help="tie-break seed (default 0)")
    plan.add_argument("--out", type=Path, default=DEFAULT_PLAN_PATH)
    plan.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    plan.set_defaults(func=cmd_label_plan)

    stats = subparsers.add_parser(
        "label-stats",
        parents=[label_options],
        help="label counts by label, mode, company, question type and edge case",
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

    eval_verifiers = subparsers.add_parser(
        "eval-verifiers",
        help="run verifiers over the confirmed gold labels; write results/verifier_eval.*",
    )
    eval_verifiers.add_argument(
        "--verifier",
        nargs="+",
        required=True,
        choices=VERIFIER_NAMES,
        help="one or more verifiers, run one after another",
    )
    eval_verifiers.add_argument("--gold", type=Path, default=DEFAULT_EXPORT_PATH)
    eval_verifiers.add_argument("--company", nargs="+", metavar="TICKER", help="limit to tickers")
    eval_verifiers.add_argument("--limit", type=int, default=None, help="first N claims only")
    eval_verifiers.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    eval_verifiers.add_argument(
        "--prices",
        type=Path,
        default=None,
        help=f"hosted prices for the cost ESTIMATE (default {DEFAULT_PRICES_PATH.name})",
    )
    add_verifier_options(eval_verifiers)
    eval_verifiers.set_defaults(func=cmd_eval_verifiers)

    errors = subparsers.add_parser(
        "error-report",
        help="misclassified claims per verifier, by edge-case tag (results/error_report.md)",
    )
    errors.add_argument(
        "--verifier", nargs="+", default=None, help="default: every verifier with predictions"
    )
    errors.add_argument("--gold", type=Path, default=DEFAULT_EXPORT_PATH)
    errors.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    errors.add_argument(
        "--out", type=Path, default=None, help="default: <results-dir>/error_report.md"
    )
    errors.set_defaults(func=cmd_error_report)

    demo = subparsers.add_parser(
        "demo",
        help="local Gradio demo: ask, retrieve, answer, verify each claim ([demo] extra)",
    )
    demo.add_argument("--config", type=Path, default=DEFAULT_COMPANIES_PATH)
    demo.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    demo.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    demo.add_argument(
        "--embedder", choices=["bge-small", "hashing"], default="bge-small",
        help="must match the one the indexes were built with",
    )  # fmt: skip
    demo.add_argument(
        "--company", nargs="+", metavar="TICKER", help="default: every company with an index"
    )
    demo.add_argument(
        "--verifiers",
        nargs="+",
        choices=VERIFIER_NAMES,
        default=None,
        help=f"verifiers to offer (default: {' '.join(DEMO_VERIFIERS)}; fake with --dry-run)",
    )
    demo.add_argument(
        "--dry-run",
        action="store_true",
        help="fake LLM (answers with the top passage's first sentence); no Ollama needed",
    )
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=7860)
    add_verifier_options(demo)
    demo.set_defaults(func=cmd_demo)

    training = subparsers.add_parser(
        "build-training-data",
        help="synthetic verifier training data from chunks (gold set excluded, split by company)",
    )
    training.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    training.add_argument("--out-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    training.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help=f"splits and sizes (default {DEFAULT_TRAINING_CONFIG.name})",
    )
    training.add_argument(
        "--gold", type=Path, default=DEFAULT_EXPORT_PATH, help="gold labels to exclude"
    )
    training.add_argument(
        "--labels", type=Path, default=DEFAULT_LABELS_PATH, help="label file to exclude too"
    )
    training.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_DIR,
        help="every chunk retrieved for a candidate is excluded as well",
    )
    training.set_defaults(func=cmd_build_training_data)
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
    except (ConfigError, SecRequestError, LLMError, VerifierError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
