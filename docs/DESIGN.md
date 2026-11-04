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

**Which `.env` is read:** only `./.env` in the directory you run from, or the file
passed with `--env-file` (`get_sec_user_agent(env_file=...)` in Python). There is no
upward search. python-dotenv's default `find_dotenv()` searches up from the *calling
module's* location, so it always found the repo-root `.env`, even in tests that had
changed to an empty folder. Tests also run with the network blocked (`tests/conftest.py`),
so none of them can reach sec.gov.

**Unknown keys are rejected, and `period` is a real field:** pydantic ignores unknown
keys by default, so the hand-added `period:` lines were silently dropped, and a typo
like `acession:` would have been too (leaving the company "unpinned" with no hint why).
Both models now use `extra="forbid"`. `period` (the period of report, YYYY-MM-DD) is
validated as a date, as is `filing_date`. `resolve-filings` prints a `period:` line so
the pasted block is complete. A bad config file raises `ConfigError`, which the CLI
prints as a clean error rather than a traceback.

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
- ~~`rowspan` is ignored.~~ Handled since Day 4: see below.
- A colspan=1 heading sitting over a `$` column keeps that column, so it shows up as
  an extra empty cell.
- ~~Page headers, footers and page numbers are kept.~~ The common shapes are removed
  since Day 4 (`parsing/cleanup.py`); unusual footers still need the real filings.
- A literal `|` inside a cell isn't escaped. Rows are split on newlines, not pipes,
  so the chunker isn't affected.

## Item sections and `groundcheck parse` (`parsing/sections.py`, `cleanup.py`, `filing.py`), Day 4

**What:** `groundcheck parse` reads each pinned filing from `data/raw/`, turns it into
text (`html_to_text`), removes page furniture, splits it into Item sections, and writes
`data/parsed/<TICKER>.json` (a `ParsedFiling`: metadata, the source file's sha256,
sections, warnings) plus `<TICKER>.txt` (the full cleaned text, handy for copying gold
evidence strings). It prints each section's title, character and word counts, and any
warnings.

**Heading detection:** a block is a heading candidate if it starts with `Item 7.`,
`ITEM 1A`, `Items 1 and 2.`, `PART II, Item 5.` and so on, and is at most 200
characters. If the text after the item number starts lowercase ("Item 7 of this
report...") it's a sentence, not a heading. A one-row `[TABLE]` also counts, because
some filers lay out `Item 7. | Management's Discussion` as a two-cell table. A table
with several rows never counts, so a table of contents laid out as a table is ignored.

**Duplicate headings (the table of contents):** each Item heading usually appears
twice, once in the TOC and once where the section starts, plus sometimes again as
"(continued)". Items are picked in canonical order (1 < 1A < 1B < 2 < ... < 16), each
after the previous pick: the first candidate followed by at least 200 characters
before the next candidate, else the last one (for items whose whole body is "None.").
TOC lines are followed by almost nothing, so they lose. The ordering stops the *last*
TOC line, which is often followed by a long forward-looking-statements block, from
being picked. A later duplicate ("(continued)") stays inside the section it repeats.
*Alternatives:* "always take the last occurrence" breaks on "(continued)" headings;
"take the longest span" can pick a "(continued)" page and drop the real start.

**Sections** run from their heading (included in the text) to the next picked heading.
Text before the first one (cover page, TOC) is front matter: its length is recorded,
but it isn't a section, so it's never chunked. A trailing `PART II` line is trimmed off
the end of the section before it. `Items 1 and 2.` (Exxon's style) becomes one
section with `covers: ["1", "2"]`.

**Warnings:** each of Items 1, 1A, 7 and 8 that isn't found is logged as a WARNING
(prefixed with the ticker), stored in the JSON, and printed by `parse`. A heading that
only appears out of order is a warning too. Nothing is silently skipped.

**Page furniture** (`cleanup.py`), removed block by block with conservative patterns:
bare page numbers (`23`, `Page 7`, `- 13 -`, `ii`), "Table of Contents" back-links,
and short (≤120 characters) running footers that mention `10-K` and start or end with
a page number (`Apple Inc. | 2025 Form 10-K | 23`). Item headings and tables are never
removed. *Alternative:* drop any short line repeated on many pages. That's more
general, but it could also drop a repeated real sentence, and it needs the real
filings to tune. (Added on Day 8, tuned on what the real filings showed: see
"Parse hardening" below.)

**Rowspan** (`html_text.py`): cells are placed on a column grid. A cell with
`rowspan=N` occupies its columns in the next N-1 rows too, so those rows' cells start
further right and stay aligned. Its text appears only in its first row, so a value is
never shown as reported twice.

**Known gaps** (to check with the real filings):
- ~~JPM's 10-K is mostly its annual report plus a "Form 10-K cross-reference index"~~.
  It did parse, but into stubs (Item 7: 396 chars) and one giant Item 15. Handled on
  Day 8 by cross-reference layout recovery, without mapping page numbers: see below.
- Headings split across two blocks *before* the item number (e.g. `ITEM` and `7.` in
  separate `<div>`s) aren't recognized.
- Footers that don't mention `10-K` and aren't a bare number (e.g. only the company
  name) are kept, unless they repeat 8+ times (Day 8, repeated-line removal).

## Chunking (`chunking/`), Day 5

**What:** `groundcheck chunk` reads `data/parsed/<TICKER>.json` and writes
`data/chunks/<TICKER>.jsonl`, one `Chunk` per line: `chunk_id`, ticker, accession,
`item`, `section_title`, `kind` (`prose` or `table`), `index`, `text`, `word_count`,
and `chunker` (the chunker's name and settings, e.g. `fixed-p250-o50-t120`). These
files are committed, so retrieval results and labels can refer to chunk IDs.

**The `Chunker` interface:** a `typing.Protocol` with a `name` and
`chunk(filing) -> list[Chunk]`. The later `TableAwareChunker` only has to match that
shape. *Alternative:* an abstract base class. A Protocol needs no inheritance, and the
fake or experimental chunkers in tests don't have to import anything.

**`FixedSizeChunker` (the baseline):**
- *Prose:* all non-table blocks of a section form one word stream, cut into
  250-word windows that start every 200 words (50 words of overlap). A fact cut at one
  window's edge is whole in the next. The last window stops at the end of the
  section, so nothing is lost and there's no tiny duplicate tail. Prose before and
  after a table stays in one stream, so a table doesn't leave short fragments behind.
- *Tables:* chunked separately, split only between rows, and every piece repeats the
  header. The header is the leading rows with an empty label cell (`| 2025 | 2024`,
  `| Years ended December 31,`), up to 3, or else just the first row. Pieces stay
  within 120 words. Pipes count as words, and numbers cost more tokens per word than
  prose, which is why the limit is smaller. A single row longer than the limit becomes
  its own piece instead of being cut.
- *Never across sections:* each Item section is chunked on its own.
- *Deterministic IDs:* `<TICKER>-<item>-<p|t><nnnn>` (e.g. `AAPL-7-t0003`): readable
  in a diff or a label file, and stable as long as the parsed text and the settings
  don't change.

**Why word counts:** bge-small's limit is 512 tokens, and English prose runs about
1.3 tokens per word, so 250 words is roughly 330 tokens, comfortably under the limit.
The tokenizer isn't available in the sandbox, so `groundcheck index` counts real
tokens on your machine and reports any chunk that would be truncated. That check found
number-dense windows at 520 to 632 tokens, so since Day 8 `chunk` also enforces a
480-token cap with the real tokenizer (see "Token-safe chunks" below).

**JSONL output** is written with a fixed key order and `ensure_ascii=False` (curly
quotes stay readable), atomically via a `.part` file. Re-writing the same chunks gives
byte-identical output, so git diffs show only real changes.

*Alternatives considered:* sentence-aware splitting (needs a sentence splitter, and
10-K sentences with "Inc." and "U.S." trip simple ones), and token-based windows
(needs the tokenizer at chunk time, which the sandbox and CI don't have). Both can
plug in behind the same interface later.

## Embeddings and per-company FAISS indexes (`retrieval/embedder.py`, `index.py`), Day 6

**What:** `groundcheck index` embeds each company's chunks and writes
`data/index/<TICKER>/index.faiss` plus `meta.json`. `groundcheck retrieve TICKER
"question" -k 5` prints the top chunks with their cosine scores.

**The `Embedder` interface:** `name`, `dim`, `embed_documents(texts)` and
`embed_queries(texts)`, each returning float32, L2-normalized rows.
- `BgeEmbedder` wraps `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token limit) in
  sentence-transformers. It runs on Apple MPS when PyTorch reports it, else on CPU
  (`--device` overrides). Queries get bge's recommended instruction prefix
  ("Represent this sentence for searching relevant passages: "); passages don't, as the
  model card says. sentence-transformers lives in the `[embed]` extra because it pulls
  in PyTorch. Without it you get a clean error telling you to install the extra.
- `HashingEmbedder` is the test fake: each lowercase word (numbers like `12,345` stay
  one token) is hashed with blake2b into one of 256 buckets. Word overlap then gives
  cosine similarity, so tests can assert sensible rankings. It's deterministic across
  processes because it uses `hashlib`, not Python's per-process salted `hash()`.
  `--embedder hashing` also lets you smoke-test the whole pipeline without the model.

**Why `IndexFlatIP` on normalized vectors:** for unit vectors the inner product *is*
cosine similarity, and a flat index compares the query with every vector, so results
are exact. A filing is a few thousand chunks at most; exact search over that takes
well under a millisecond, so approximate indexes (IVF, HNSW) would only add recall
error and tuning knobs. Vectors are normalized again after the model, even though
sentence-transformers was asked to normalize, so the fake and the real model take the
same path.

**Why one index per company:** each question is about one filing, so it should get its
full `top_k` from that filing. With one shared index, a near-identical risk-factor
sentence from another company could take up slots. The test
`test_one_index_per_company` checks this.

**Staleness guards** (`meta.json`): the index records the embedder name and dimension,
the chunker name, the chunk IDs in vector order, and the sha256 of the chunks file.
Loading fails if the chunks file has changed since the index was built. Searching
fails if the query embedder differs from the build embedder. Both mistakes would
otherwise return confident-looking, wrong results.

**Token check:** with the real model, `index` counts each chunk's tokens with the bge
tokenizer and prints a WARNING for any chunk over 512 tokens, which the model would
silently truncate. This is the check the chunker's word limits rely on.

**Tests** never load the real model: the bge wrapper is tested against a stand-in
`sentence_transformers` module (device choice, `max_seq_length`, query prefix,
normalization), and FAISS behavior with the hashing embedder. `faiss-cpu` and `numpy`
are normal dependencies (small wheels, no GPU), so CI tests the real index code.

## Retrieval evaluation (`retrieval/evaluate.py`, `data/eval/retrieval_questions.yaml`), Day 7

**What:** `groundcheck eval-retrieval` runs every filled-in question through its
company's index and reports recall@{1,3,5,10} and section-hit@{1,3,5,10}: overall,
per company and per `qtype`. Per-question details (retrieved chunk IDs, rank of the
first chunk containing gold, scores per k) are written to
`data/eval/results/retrieval_<embedder>.json`. `--check` only verifies that every gold
string occurs in the chunks. It needs no model, so run it right after writing
questions.

**Question file:** YAML (comments allowed, easy to hand-edit). Each entry has `id`,
`ticker`, `qtype`, `question`, `gold_evidence` (a list of exact strings copied from
`data/parsed/<TICKER>.txt`), an optional `gold_items`, and `notes`. Unknown keys and
unknown qtypes are rejected, so `gold_evidnce:` fails loudly. The committed template
has 30 empty entries (5 per labeled company, one each of numeric_lookup, table,
multi_step, yes_no and temporal). No gold text is pre-filled: gold labels only come
from a human.

**Metrics:**
- *recall@k*: the fraction of a question's gold strings found in at least one top-k
  chunk, averaged over questions. Fractional, so a multi_step question that needs two
  values and retrieves one scores 0.5. *Alternative:* "hit@k" (any gold string found)
  would give that question full credit for half the evidence.
- *section-hit@k*: 1 if any top-k chunk is from a gold Item section. That's `gold_items`
  if given, otherwise the sections whose chunks contain the gold. This tells "found
  the right part of the filing but the wrong chunk" apart from "looked in the wrong
  place entirely".

**Matching:** a chunk contains a gold string if it's a substring after collapsing all
whitespace to single spaces on both sides (`normalize_for_match`). Case, punctuation,
curly quotes and `|` cell separators must match exactly. That's why gold strings
should be copied from the parsed `.txt`, which comes from the same `html_to_text`
output the chunks do. A table row boundary counts as whitespace.

**What's skipped, and why it's reported rather than scored:**
- *unfilled* template entries (no question or no gold);
- *unanswerable* questions (nothing to retrieve; they matter for generation, later);
- *gold_not_in_corpus*: a gold string that occurs in none of the company's chunks.
  That's a typo, a normalization mismatch, or a string straddling two chunks, not a
  retrieval failure, so scoring it as 0 would understate retrieval. It's printed as a
  WARNING and counted in `skipped`, and `--check` exits non-zero.

**Testability:** `evaluate()` takes a `retrieve(ticker, question, k)` function rather
than an index. Tests pass a fake retriever that returns chunks in a fixed order, so
each metric can be checked by hand. The CLI test runs the whole path (parse → chunk →
index → eval) with the hashing embedder.

## Parse hardening and token-safe chunks (Day 8)

The first real run on the Mac showed four problems the synthetic fixtures couldn't:
two filings (JPM, XOM) put MD&A and the financial statements outside Items 7 and 8,
running page headers survived cleanup, and some 250-word windows were over the token
limit. Each fix below is covered by synthetic fixtures that copy the real patterns
(`tests/fixtures/parsing/build_xref_fixtures.py` writes `xref_bank_style.htm`, JPM's
layout, and `xref_energy_style.htm`, XOM's). They still need checking on the real
filings (see the PR for the commands).

### Section-size warnings (`sections.py`, `check_section_sizes`)

**What:** besides missing Items, `parse` now warns when Item 1, 1A, 7 or 8 is under
1,500 characters, and when any *other* Item is over 60,000. The thresholds sit in one
commented block (`KEY_ITEMS`, `MIN_KEY_SECTION_CHARS`, `MAX_OTHER_SECTION_CHARS`). The
warnings are stored in the JSON and counted in the final "N warning(s)" line.

**Why these numbers:** they sit in wide gaps in the real data. The stubs that must
warn are 265 to 710 characters (JPM Items 7 and 8, XOM Items 7 and 8); the smallest
real key section is far above 1,500. The oversized sections that must warn are
342,034 (XOM Item 16) and 1,050,977 (JPM Item 15); the largest legitimate non-key
section seen is XOM Item 2 at 40,860. Key Items are exempt from the upper limit
because they really are long: JNJ Item 8 is 227,361 and BRK-B Item 1 is 126,647. An
"Items 1 and 2" section counts as key.

**Why warn on the split as found, even after a recovery:** the warning is the evidence
that the layout was unusual, and the recovery is a heuristic you should check. So the
size warnings describe the split before recovery, and after a recovery the sizes are
checked again and any *new* problem is added too. Nothing that fired is silently
dropped. *Alternative:* only report sizes after recovery. That hides the fact that
JPM's sections were rebuilt.

### Repeated running headers (`cleanup.py`, `remove_repeated_lines`)

**What:** after the pattern-based furniture removal and before the Item split, any
line of at most 120 characters that occurs at least 8 times in the filing (compared
case-insensitively, ignoring surrounding whitespace) is removed. JPM prints
"Management's discussion and analysis" at the top of every MD&A page; XOM prints
"MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF
OPERATIONS" every 20 to 40 lines. Left in, these lines end up in dozens of chunks and
pull MD&A chunks toward any question that mentions MD&A.

**The heading is kept:** for each repeated line, the first occurrence that is
*followed by prose* stays, because that one is the real section heading (and Part C
needs it). "Followed by prose" means a paragraph (at least 100 characters and 15 words,
not a heading) within the next 4 blocks, passing over subheadings and data tables (XOM's
MD&A heading is followed by "FUNCTIONAL EARNINGS SUMMARY" and a table), but not over a
table of contents. *Decision:* the spec said "keep the first occurrence if it is a
heading". Taking the first occurrence *that is* a heading is a small generalization: if
the first one is a table-of-contents line, the real heading further on survives
instead of every copy being dropped.

**What is never removed:** tables (a repeated `Total | ...` row is data), Item headings
(the splitter needs every copy to tell TOC from section), and lines over 120
characters. Each removed line and its count is logged at INFO (`groundcheck -v parse`)
and the total is printed as "repeated header lines removed". *Risk:* a short real line
that genuinely repeats 8+ times (e.g. a footnote like "(a) Refer to Note 2.") is also
removed. The INFO log lists every removed line so this can be checked per filing.

**Why a table of contents needs page numbers in order:** a table whose rows mostly end
in a 1 to 3 digit number is a TOC only if there are at least two such rows and the
numbers never go down. A data table's last column (`Upstream | 412`) jumps around.

### Cross-reference layout recovery (`recovery.py`)

**What:** some filers print Items 7 and 8 as pointers ("see pages 50-160", "Reference
is made to ... the Financial Section"), and the real MD&A and financial statements sit
later, inside whatever Item heading precedes them: JPM's annual report lands in Item
15, XOM's Financial Section in Item 16 (after the signatures). Recovery relabels those
spans as Items 7 and 8.

**Trigger:** Item 7 or 8 is a stub (under 1,500) *and* a later non-key section is over
60,000. Inside the largest such section:
1. *MD&A start:* the first line with no `|` that contains "Management's discussion and
   analysis" (JPM) or "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND
   RESULTS OF OPERATIONS" (XOM), either apostrophe, and is followed by prose. The line
   is either the heading itself or a paragraph that opens with the phrase (within the
   first 40 characters, as in JPM's "The following is Management's discussion and
   analysis ..."). A paragraph that cites a page ("see Management's discussion and
   analysis on pages 50-160") never counts.
2. *Financial statements start:* the first line with no `|`, after the MD&A start, equal
   to "Report of Independent Registered Public Accounting Firm".

Table-of-contents rows can never be starts: every row of a `[TABLE]` (and any line
with a `|`) is skipped, which covers JPM's cross-reference index and XOM's
"Report of Independent Registered Public Accounting Firm | 69".

**Result:** MD&A start → financial statements start is Item 7; financial statements
start → the end of the oversized section is Item 8 ("Financial Statements and
Supplementary Data", so the notes and supplementary schedules belong there); the
oversized section keeps only what came before the MD&A start. The stubs' own text is
kept at the top of the recovered sections, so no text is dropped and chunk IDs stay
unique (one Item 7, one Item 8). Recovered sections have `recovered: true` in the JSON
and `recovered=true` in `parse` output, and a NOTE line says where each starts (line
number in `<TICKER>.txt` and the first 80 characters), which is what to check on the
real filings.

**When it gives up:** if either start isn't found, or Item 7 or 8 is already a full
section (it would be overwritten), nothing changes and a WARNING naming the filing is
logged, stored and counted. *Alternative:* map the cross-reference index's page numbers
to the text. That needs page boundaries, which `html_to_text` doesn't keep, and every
filer formats the index differently; two anchor lines are easier to check and fail
loudly.

### Token-safe chunks (`chunking/fixed.py`)

**What:** no chunk may exceed 480 tokens as counted by bge-small's tokenizer (special
tokens included). On the Mac, JNJ-1-p0001, JNJ-1-p0003, JNJ-8-p0097, WMT-3-p0001 and
WMT-3-p0002 were 520 to 632 tokens at 250 words: number-dense text runs well over 1.3
tokens per word. `FixedSizeChunker` takes an optional `count_tokens` function and
`max_tokens` (default 480, leaving headroom under the model's 512). `groundcheck chunk`
loads the real tokenizer (only the tokenizer, via `transformers`, from the `[embed]`
extra); `--tokenizer none` skips the cap (no extra needed), and tests pass a fake
counter, so CI never downloads anything.

**How a chunk is split:** only when its count is over the cap. The 250-word default
and the 200-word step are unchanged.
- *Prose* is cut at the sentence end nearest the middle (searching the middle third;
  "U.S.", "Inc.", "No." and initials don't count as sentence ends), else at the middle
  word. The second part starts 50 words (the configured overlap) before the cut, so a
  fact at the cut is whole in one part. Repeat until every part fits. The window's
  first and last words don't move, so the overlap with neighbouring windows is kept.
- *Tables* are cut between rows (each part repeats the header), and a single row that
  is still too long is cut between words. No overlap, as before.
- A single "word" longer than the cap can't be split; `chunk` prints a WARNING for it.

**IDs stay stable and unique:** a split window keeps its ID with a letter appended
(`JNJ-1-p0001a`, `JNJ-1-p0001b`) and its parts share the window's `index`. Every window
that wasn't split keeps exactly the ID and text it had before, so labels and results
that cite those IDs stay valid. The chunker name gains `-k480` when the cap is on
(`fixed-p250-o50-t120-k480`), so an index built from capped chunks says so. *Alternative:*
renumber all windows after a split. Simpler IDs, but every later chunk ID in the
section shifts. *Alternative:* token-based windows everywhere. That changes every
chunk, not only the five that were too long.

