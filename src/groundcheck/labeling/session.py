"""The interactive labeler behind `groundcheck label`.

For each claim it shows the question, the answer, the retrieved chunks (with their
IDs) and the claim, then asks for:

1. the label: `s`upported, `c`ontradicted, `u`nsupported; `k` skips the claim,
   `q` quits (Enter keeps an existing label);
2. for supported/contradicted, one or more evidence spans pasted from the chunks
   above. Paste, then press Enter on an empty line. The span must occur in a
   retrieved chunk after whitespace normalization; if it occurs in several, you pick
   which one you are citing;
3. an optional edge-case tag (by number), and an optional note;
4. whether to confirm the label (only confirmed labels are exported).

Each label is saved immediately (atomically), so quitting, Ctrl-D or a crash loses at
most the claim on screen. Plain `input()`/`print()`, injected as functions so tests
can script a whole session.
"""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from groundcheck.labeling.schema import EDGE_CASES, EvidenceSpan, LabelRecord, make_span
from groundcheck.labeling.store import ClaimItem, LabelStore
from groundcheck.retrieval.evaluate import normalize_for_match

LABEL_KEYS = {"s": "supported", "c": "contradicted", "u": "unsupported"}
RULE = "-" * 78


class Quit(Exception):
    """The user typed q (or the input ended)."""


@dataclass
class SessionResult:
    labeled: int = 0
    kept: int = 0
    skipped: int = 0
    remaining: int = 0  # claims not shown because the user quit
    quit: bool = False


class Labeler:
    def __init__(
        self,
        store: LabelStore,
        input_fn: Callable[[str], str] | None = None,
        output: Callable[[str], None] | None = None,
        now: Callable[[], str] | None = None,
    ):
        self.store = store
        # Looked up at call time, so tests can patch builtins.input / print.
        self.input_fn = input_fn or (lambda prompt: input(prompt))
        self.out = output or (lambda text: print(text))
        self.now = now or (lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))

    def ask(self, prompt: str) -> str:
        try:
            return self.input_fn(prompt).strip()
        except EOFError as exc:
            raise Quit from exc

    def ask_block(self, prompt: str) -> str:
        """Lines until an empty line, joined with spaces (a pasted table row breaks
        across lines). 'q' alone on the first line quits."""
        self.out(prompt)
        lines = []
        while True:
            try:
                line = self.input_fn("")
            except EOFError as exc:
                if lines:
                    break
                raise Quit from exc
            if not line.strip():
                break
            if not lines and line.strip().lower() == "q":
                raise Quit
            lines.append(line)
        return normalize_for_match(" ".join(lines))

    # --- display ----------------------------------------------------------------------

    def show(self, item: ClaimItem, position: int, total: int) -> None:
        c = item.candidate
        n = c.claims.index(item.claim) + 1
        self.out(RULE)
        self.out(
            f"[{position}/{total}] {c.ticker} {c.question_id} ({c.qtype})  claim {n} of "
            f"{len(c.claims)}  {item.claim.claim_id}"
        )
        self.out(f"QUESTION: {c.question}")
        self.out(f"ANSWER:   {c.answer}")
        self.out("RETRIEVED CHUNKS:")
        for chunk in c.retrieved:
            self.out(
                f"  [{chunk.rank}] {chunk.chunk_id} (Item {chunk.item}, {chunk.kind}, "
                f"score {chunk.score:.3f})"
            )
            for line in chunk.text.splitlines():
                self.out(f"      {line}")
        self.out(f"CLAIM:    {item.claim.text}")
        existing = self.store.get(item.claim.claim_id)
        if existing:
            status = "confirmed" if existing.confirmed else "NOT confirmed"
            self.out(
                f"CURRENT LABEL: {existing.label} ({status})"
                + (f", edge case {existing.edge_case}" if existing.edge_case else "")
            )

    # --- questions --------------------------------------------------------------------

    def ask_evidence(self, item: ClaimItem) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        retrieved = item.candidate.retrieved
        while True:
            prompt = (
                "Paste an evidence span from the chunks, then Enter on an empty line"
                if not spans
                else "Paste another span, or just Enter to finish"
            )
            text = self.ask_block(prompt + ":")
            if not text:
                if spans:
                    return spans
                self.out("  At least one evidence span is required for this label.")
                continue
            found = [(c, s) for c in retrieved if (s := make_span(c.chunk_id, c.text, text))]
            if not found:
                self.out(
                    "  Not found in any retrieved chunk (it must match exactly, apart from "
                    "whitespace). Copy it again from the chunk text."
                )
                continue
            if len(found) > 1:
                choices = ", ".join(f"[{c.rank}] {c.chunk_id}" for c, _ in found)
                ranks = {str(c.rank): s for c, s in found}
                answer = ""
                while answer not in ranks:
                    answer = self.ask(f"  Found in {choices}. Which one? [{found[0][0].rank}]: ")
                    answer = answer or str(found[0][0].rank)
                span = ranks[answer]
            else:
                span = found[0][1]
            if any(s.chunk_id == span.chunk_id and s.start == span.start for s in spans):
                self.out("  That span is already cited.")
                continue
            spans.append(span)
            self.out(f"  OK: {span.chunk_id} [{span.start}:{span.end}]")

    def ask_edge_case(self) -> str | None:
        menu = "  ".join(f"{i}={name}" for i, name in enumerate(EDGE_CASES, start=1))
        while True:
            answer = self.ask(f"Edge case? Enter=none  {menu}: ")
            if not answer:
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(EDGE_CASES):
                return EDGE_CASES[int(answer) - 1]
            if answer in EDGE_CASES:
                return answer
            self.out("  Type a number from the list, or Enter for none.")

    def label_one(self, item: ClaimItem) -> str:
        """Returns 'labeled', 'kept' or 'skipped'; raises Quit."""
        existing = self.store.get(item.claim.claim_id)
        keep = "  Enter=keep" if existing else ""
        while True:
            key = self.ask(
                f"Label? s=supported c=contradicted u=unsupported  k=skip q=quit{keep}: "
            )
            key = key.lower()
            if key == "q":
                raise Quit
            if key == "k":
                return "skipped"
            if not key and existing:
                return "kept"
            if key in LABEL_KEYS:
                break
            self.out("  Type s, c, u, k or q.")
        label = LABEL_KEYS[key]
        evidence = self.ask_evidence(item) if label != "unsupported" else []
        while True:
            edge_case = self.ask_edge_case()
            if edge_case == "absent_evidence" and label != "unsupported":
                self.out("  absent_evidence goes with unsupported; pick another tag.")
                continue
            break
        note = self.ask("Note (optional): ")
        confirmed = self.ask("Confirm this label? [Y/n]: ").lower() not in ("n", "no")
        c = item.candidate
        try:
            record = LabelRecord.model_validate(
                {
                    "claim_id": item.claim.claim_id,
                    "candidate_id": c.candidate_id,
                    "question_id": c.question_id,
                    "ticker": c.ticker,
                    "qtype": c.qtype,
                    "claim": item.claim.text,
                    "label": label,
                    "evidence": [s.model_dump() for s in evidence],
                    "edge_case": edge_case,
                    "confirmed": confirmed,
                    "note": note,
                    "labeled_at": self.now(),
                },
                context={"chunks": item.chunks},
            )
        except ValidationError as exc:  # shouldn't happen: every answer was checked
            self.out(f"  Not saved: {exc.errors()[0]['msg']}")
            return "skipped"
        self.store.put(record)
        self.out(f"  Saved: {label}{' (confirmed)' if confirmed else ' (NOT confirmed)'}")
        return "labeled"

    def run(self, items: list[ClaimItem], only_unlabeled: bool = False) -> SessionResult:
        if only_unlabeled:
            items = [i for i in items if self.store.get(i.claim.claim_id) is None]
        result = SessionResult()
        if not items:
            self.out("Nothing to label.")
            return result
        for position, item in enumerate(items, start=1):
            self.show(item, position, len(items))
            try:
                outcome = self.label_one(item)
            except Quit:
                result.quit = True
                result.remaining = len(items) - position + 1
                break
            if outcome == "labeled":
                result.labeled += 1
            elif outcome == "kept":
                result.kept += 1
            else:
                result.skipped += 1
        self.out(RULE)
        self.out(
            f"Session: {result.labeled} labeled, {result.kept} kept, {result.skipped} skipped"
            + (f", quit with {result.remaining} left" if result.quit else "")
            + f". Saved to {self.store.path}"
        )
        return result
