# Test fixtures

Everything in this folder is **synthetic**: hand-written to copy the *shape* of real
SEC data (field names, structure, formatting quirks). It is not real SEC data. Company
names, CIKs, accession numbers and figures are made up and refer to no real company.

- `edgar/submissions_synthetic.json`: the structure of a
  `data.sec.gov/submissions/CIK##########.json` response, with a 10-K/A newer than
  the latest 10-K and an older 10-K, to test that the right filing is picked.
- `parsing/workiva_style.htm`: the shape of a Workiva-generated inline-XBRL 10-K:
  an XML declaration, a hidden `ix:header`, text split across `<span>`s, `ix:` tags
  around numbers, and a financial table with separate `$`, `)` and `%` cells, padding
  columns and colspan headers.
- `parsing/donnelley_style.htm`: the shape of an older Donnelley-style 10-K: uppercase
  tags, `<font>`, `<br>` inside paragraphs, a bullet list laid out as a table, a
  single-cell "box" table, `<th>` headers and a nested table.
