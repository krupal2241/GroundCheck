"""Offline stand-ins for `groundcheck pipeline --dry-run`.

The sandbox and CI cannot reach sec.gov, so the dry run's download stage uses
`FakeSecClient`, which serves a small synthetic 10-K for each company instead of the
real filing. The synthetic filing is rebuilt from that company's committed chunks
(`data/chunks/<TICKER>.jsonl`): an Item 1, 1A, 7 and 8 heading each followed by a few
of the company's own paragraphs and tables. So the dry run parses, chunks, indexes and
generates over text that looks like the real thing, without any network access.
Everything it writes goes under the dry-run scratch directory.
"""

import html
from pathlib import Path

from groundcheck.chunking.base import chunks_path, read_chunks
from groundcheck.config import Company

ITEMS = (
    ("1", "Business"),
    ("1A", "Risk Factors"),
    ("7", "Management's Discussion and Analysis of Financial Condition and Results of Operations"),
    ("8", "Financial Statements and Supplementary Data"),
)
PER_ITEM = 6  # chunks per Item in the synthetic filing


def _table_html(text: str) -> str:
    rows = [r for r in text.splitlines() if r.strip() and r.strip() not in ("[TABLE]", "[/TABLE]")]
    cells = (
        "<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in row.split("|")) + "</tr>"
        for row in rows
    )
    return "<table>" + "".join(cells) + "</table>"


def synthetic_filing_html(company: Company, chunks_dir: Path) -> bytes:
    path = chunks_path(company.ticker, chunks_dir)
    chunks = read_chunks(path) if path.is_file() else []
    body = [
        "<html><body>",
        "<div>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</div><div>FORM 10-K</div>",
        f"<div>{html.escape(company.name)} (synthetic dry-run filing, not SEC data)</div>",
    ]
    for item, title in ITEMS:
        body.append(f"<div><b>Item {item}. {html.escape(title)}</b></div>")
        mine = [c for c in chunks if c.item == item][:PER_ITEM]
        if not mine:
            mine_text = [f"{company.name} has no committed chunks for Item {item}."]
            body += [f"<p>{html.escape(t)}</p>" for t in mine_text]
        for chunk in mine:
            if chunk.kind == "table":
                body.append(_table_html(chunk.text))
            else:
                body.append(f"<p>{html.escape(' '.join(chunk.text.split()))}</p>")
    body.append("<div><b>Item 9. Changes in and Disagreements with Accountants</b></div>")
    body.append("<p>None.</p></body></html>")
    return "\n".join(body).encode("utf-8")


class FakeSecClient:
    """Serves synthetic filings by URL; never touches the network."""

    def __init__(self, companies: list[Company], chunks_dir: Path):
        self.by_document = {c.primary_document: c for c in companies if c.primary_document}
        self.chunks_dir = chunks_dir
        self.urls: list[str] = []

    def get(self, url: str) -> bytes:
        self.urls.append(url)
        document = url.rsplit("/", 1)[-1]
        company = self.by_document.get(document)
        if company is None:
            raise KeyError(f"dry run: no synthetic filing for {url}")
        return synthetic_filing_html(company, self.chunks_dir)
