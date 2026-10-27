"""Command-line entry point: `groundcheck <subcommand>`.

This file only parses arguments and calls into the real modules. Keeping the
logic out of the CLI means every step can also be called from Python (tests,
notebooks, scripts) without going through argparse.

Subcommands are added as each module is built.
"""

import argparse

from groundcheck import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundcheck",
        description="Claim-level hallucination verification for RAG over SEC 10-K filings.",
    )
    parser.add_argument("--version", action="version", version=f"groundcheck {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
