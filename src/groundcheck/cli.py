"""Command-line entry point: `groundcheck <subcommand>`.

This file only parses arguments and calls into the real modules. Keeping the
logic out of the CLI means every step can also be called from Python (tests,
notebooks, scripts) without going through argparse.

Subcommands are added as each module is built.
"""

import argparse
import logging
import sys
from pathlib import Path

from groundcheck import __version__
from groundcheck.config import (
    DEFAULT_COMPANIES_PATH,
    ConfigError,
    get_sec_user_agent,
    load_companies,
)
from groundcheck.edgar.client import SecClient, SecRequestError
from groundcheck.edgar.download import DEFAULT_RAW_DIR, download_all, resolve_latest_10k


def cmd_resolve_filings(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    client = SecClient(get_sec_user_agent())
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
        print(f'    fiscal_year_end: "{info.fiscal_year_end}"\n')
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    config = load_companies(args.config)
    companies = [config.get(t) for t in args.only] if args.only else config.companies
    client = SecClient(get_sec_user_agent())
    results = download_all(client, companies, args.raw_dir)
    fetched = sum(results.values())
    print(f"Downloaded {fetched} filing(s); {len(results) - fetched} already cached.")
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
