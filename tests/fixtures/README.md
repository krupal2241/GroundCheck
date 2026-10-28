# Test fixtures

Everything in this folder is **synthetic**: hand-written to copy the *shape* of real
SEC data (field names, structure, formatting quirks). It is not real SEC data. Company
names, CIKs, accession numbers and figures are made up and refer to no real company.

- `edgar/submissions_synthetic.json`: the structure of a
  `data.sec.gov/submissions/CIK##########.json` response, with a 10-K/A newer than
  the latest 10-K and an older 10-K, to test that the right filing is picked.
