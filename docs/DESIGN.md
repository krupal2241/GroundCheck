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

## Configuration (`config.py`, `config/companies.yaml`), Day 2

**What:** `companies.yaml` lists the 8 companies. Each entry has a role and the exact
10-K filing to use: the accession number (SEC's ID for one filing, e.g.
`0000320193-24-000123`), the main document's file name, the filing date, and the
fiscal year end. `config.py` loads that file into pydantic models and reads
`SEC_USER_AGENT` from the environment.

**Why pin filings instead of always fetching "the latest":** reproducibility. If the
code fetched the latest 10-K on every run, a new filing in February would silently
change the corpus, and every chunk ID, retrieval score and label tied to the old text
would stop matching. With pinned filings, the same config always produces the same
data. Updating to a newer filing becomes a deliberate change you can see in git.

**Why pydantic:** the YAML is read into typed objects, and mistakes fail immediately
with a readable message. That covers a malformed accession number, an accession with no
document name (a "half-pinned" entry), an unknown role, or a duplicate ticker. With
plain dicts the same typo would surface later as a confusing 404 or a `KeyError`.
*Alternatives:* `dataclasses` (no validation), or JSON Schema (validates, but you still
get dicts back).

**Why roles (`labeled` / `unlabeled` / `stress_test`):** later code filters on the role
rather than hard-coding tickers. For example, question generation will only use
`labeled` companies, so BRK-B (`stress_test`) can never leak into the test set by
accident.

**Why the secret goes in `.env` and not the YAML:** the user agent contains your email
address. Config files are committed; `.env` is gitignored. `python-dotenv` loads `.env`
but never overrides a variable you already set in your shell.

## EDGAR download (`edgar/`), Day 2

**What:** three small files.
- `ratelimit.py` makes sure two requests never start less than 0.2 s apart (≤5 per
  second).
- `client.py` sends each request with your user agent, waits on the rate limiter, and
  retries temporary failures.
- `download.py` finds each company's latest 10-K (`resolve-filings`) and downloads the
  pinned file (`download`).

**SEC's rules and how we follow them:** SEC asks automated tools to (1) identify
themselves with a name and email in the User-Agent header, and (2) stay under 10
requests per second. We refuse to run without a valid user agent, and we stay at half
the allowed rate. Caching means a normal re-run makes zero requests.

**Rate limiter design:** a "minimum interval" limiter that remembers when the last
request started and sleeps until 0.2 s have passed. *Alternative:* a token bucket,
which allows short bursts. We download one file at a time, so bursts gain nothing, and
the interval version is about 10 lines. The clock and the `sleep` function are passed
in as parameters, so tests use a fake clock and check the timing without actually
waiting.

**Retries:** HTTP 429 ("too many requests") and 5xx (server trouble) are usually
temporary. So are network errors such as timeouts and dropped connections. We retry
them with exponential backoff (2 s, 4 s, 8 s, 16 s), and if SEC sends a `Retry-After`
header we wait exactly that long instead. Any other error, like 404, is a mistake on our
side (e.g. a wrong accession number), so we fail right away with a clear message.
*Alternative:* `urllib3.Retry` mounted on the session. It's less code, but the retry
behavior is then hidden inside configuration and harder to test and explain.

**Two steps, resolve then download:** `resolve-filings` reads SEC's *submissions index*
for each company (a JSON list of all its recent filings) and prints the newest 10-K.
It skips 10-K/A amendments, which are often just a few changed pages. You paste the
values into the config, and `download` fetches exactly that file. That way a human
checks what goes into the corpus. It also catches a wrong CIK, because the command
prints the name SEC has on file.

**Cache and "never re-download":** files are stored under
`data/raw/<TICKER>/<accession>/`, along with a `meta.json` that records the URL, a
sha256 checksum, the size and the download time. If the file already exists it's
skipped. If its checksum no longer matches `meta.json`, we stop with an error rather
than quietly use or replace a file that was changed. The submissions index is cached
too, and only re-fetched with `--refresh`.

**Atomic writes:** each file is written to `name.part` and then renamed. A rename is
all-or-nothing, so if a download is interrupted (Ctrl-C, Wi-Fi drop) you never end up
with a half-written file that later looks "already downloaded".

**Why `requests` and raw endpoints rather than an EDGAR library:** libraries like
`edgartools` or `sec-edgar-downloader` are convenient, but they hide exactly the parts
an interviewer might ask about: rate limiting, caching, and which filing gets picked.
The raw API is two URLs:
- `https://data.sec.gov/submissions/CIK##########.json` (CIK zero-padded to 10 digits)
- `https://www.sec.gov/Archives/edgar/data/<cik>/<accession without dashes>/<document>`

**Testing without the network:** the tests replace the HTTP session (and the whole
client, in the download tests) with fakes that return canned responses. That's why CI
never contacts SEC. The submissions fixture in `tests/fixtures/edgar/` is synthetic: a
made-up company with the same JSON structure.

## HTML to text (`parsing/html_text.py`), Day 3

**What:** `html_to_text(html)` turns a filing's HTML into plain text. Each paragraph,
heading or `<div>` becomes one line, with a blank line between blocks. Each data table
becomes a `[TABLE]`…`[/TABLE]` block with one line per row and cells joined by ` | `.
`table_rows(table)` exposes the cleaned rows, so the chunker can later split tables on
row boundaries and repeat the header row.

**Why walk the tree ourselves instead of `soup.get_text()`:** `get_text()` either glues
everything together or puts a separator between *every* tag. That splits
`<span>Net </span><span>sales</span>` wrongly, and it can't tell a paragraph break from
an inline tag. The walker knows which tags are blocks (`div`, `p`, `br`, `li`, ...)
and which are inline (`span`, `font`, `b`, `ix:*`). Source-code line breaks inside a
paragraph are just whitespace, so all whitespace inside a block is collapsed.

**Hidden content:** inline-XBRL filings put a machine-readable copy of key facts in an
`ix:header` inside `display:none`. If we kept it, retrieval would match text no reader
ever sees. Anything with `display:none` or `hidden` is dropped, along with `head`,
`script` and `style`. `ix:nonFraction` and similar tags are only unwrapped, because
they wrap the visible numbers.

**Tables:** filers split values across cells for alignment: `$` | `1,234`, or `(1,234`
| `)`, or `45.6` | `%`. Each row is cleaned in three steps:
1. *Colspan expansion:* each cell records which grid columns it covers, so rows line
   up even when some cells span several columns.
2. *Merging:* a cell holding only `$` or `(` joins the next value, and a cell holding
   only `)` or `%` joins the previous one. That gives `$(1,234)`, `45.6%`, and `(1,234)`
   kept exactly as printed. We don't convert it to `-1234`: the claim checker needs
   the filing's own text, and the parenthesis convention is part of it.
3. *Column cleanup:* a column is kept only if some ordinary (colspan=1) cell has text
   in it. A heading that spans columns (e.g. `2025` over the `$` column and the value
   column) moves to the first kept column under it, so it lines up with its numbers.
   Empty cells in kept columns stay as empty strings, so a missing value doesn't shift
   the rest of the row left.

*Alternatives:* `pandas.read_html` flattens tables too, but it doesn't merge `$`/`)`/`%`
cells. It also pulls in pandas and gives no control over colspan placement. A
per-row "drop empty cells" rule is simpler, but one blank cell then shifts every
later value into the wrong year's column.

**Layout tables:** older filings use tables for layout too. A table left with one
column after cleanup (a boxed note), or two columns where the first holds only bullets
(a bulleted list), comes out as prose, not `[TABLE]`. That way the chunker doesn't
treat a risk factor as tabular data. A table nested inside a cell is flattened into
that cell's text.

**Parser choice:** BeautifulSoup with lxml's HTML parser. It's fast, and it tolerates
the unclosed tags in older filings. Inline-XBRL files are XHTML and begin with
`<?xml ...?>`, so bs4 warns that an XML parser would be "more reliable". We silence
that warning on purpose: an XML parser rejects the first malformed tag, and one parser
for every filing means one set of rules. `bytes` input lets bs4 pick the encoding from
`<meta charset>`, since older filings are often windows-1252.

**Normalization is deliberately light:** non-breaking spaces become spaces, and
zero-width characters and soft hyphens are removed. Curly quotes, dashes and accented
letters are kept. NFKC would also rewrite things like `½` and ligatures, and
gold-evidence matching needs identical normalization on both sides. A later
`normalize_for_match` can be stricter, applied to both sides at once.

**Known gaps** (tripwires for Day 5, when the real filings arrive):
- `rowspan` is ignored. It's rare in 10-K financial tables.
- A colspan=1 heading sitting over a `$` column keeps that column, so it shows up as
  an extra empty cell.
- Page headers, footers and page numbers (e.g. `Example Corp | 2025 Form 10-K | 23`)
  are kept for now. Removing them needs patterns from the real filings.
- A literal `|` inside a cell isn't escaped. Rows are split on newlines, not pipes,
  so the chunker isn't affected.
