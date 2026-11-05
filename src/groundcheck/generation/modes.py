"""Generation modes: where the passages the model answers from come from.

- `rag`: the top-k chunks retrieved for the question (the normal pipeline).
- `no-context`: no passages at all; the model answers closed-book, from memory.
- `wrong-context`: passages deliberately mismatched to the question: the top-k chunks
  for the same question from a *different company's* index, or the best-ranked chunks
  of the same company that come from a *different Item* than the correct passages.

A 3B model answering from good passages mostly produces supported claims, so the
hand-labeled set would be badly imbalanced. The two extra modes produce natural
hallucinations (a closed-book guess, an answer built from the wrong filing or section)
instead of synthetic ones.

In every mode the record's `retrieved` passages are the ones retrieved for the correct
question: that is the evidence shown for labeling, so each claim is judged against the
right evidence. What the model actually saw is stored separately (`context`).
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from groundcheck.config import ConfigError
from groundcheck.retrieval.index import Hit

Mode = Literal["rag", "no-context", "wrong-context"]
MODES: tuple[str, ...] = ("rag", "no-context", "wrong-context")
WrongSource = Literal["mixed", "other-company", "other-item"]
WRONG_SOURCES: tuple[str, ...] = ("mixed", "other-company", "other-item")

Retriever = Callable[[str, str, int], list[Hit]]


def stable_int(text: str) -> int:
    """A deterministic integer from a string (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


@dataclass
class WrongContextPicker:
    """Chooses mismatched passages for a question, deterministically.

    `source="mixed"` alternates per question (by a hash of the question ID) between
    another company and another Item; if the chosen kind is impossible (no other
    company indexed, or no chunk from another Item), the other kind is used.
    """

    retrieve: Retriever
    other_tickers: list[str]  # every company with an index (the question's own is skipped)
    source: WrongSource = "mixed"
    depth: int = 60  # how far down the same company's ranking to look for other Items

    def kind_for(self, question_id: str) -> str:
        if self.source != "mixed":
            return self.source
        return "other-item" if stable_int(question_id) % 2 == 0 else "other-company"

    def other_company(self, ticker: str, question: str, question_id: str, k: int):
        others = sorted(t for t in self.other_tickers if t != ticker)
        if not others:
            return None
        other = others[stable_int(question_id + "#company") % len(others)]
        hits = self.retrieve(other, question, k)
        return (hits, f"other-company:{other}") if hits else None

    def other_item(self, ticker: str, question: str, correct: list[Hit], k: int):
        wrong_items = {h.chunk.item for h in correct}
        deep = self.retrieve(ticker, question, max(self.depth, k))
        hits = [h for h in deep if h.chunk.item not in wrong_items][:k]
        if not hits:
            return None
        items = ",".join(sorted({h.chunk.item for h in hits}))
        return hits, f"other-item:{ticker} Item {items}"

    def pick(
        self, ticker: str, question: str, question_id: str, correct: list[Hit], k: int
    ) -> tuple[list[Hit], str]:
        """(mismatched hits re-ranked from 1, a description like 'other-company:MSFT')."""
        first = self.kind_for(question_id)
        order = [first, "other-company" if first == "other-item" else "other-item"]
        for kind in order:
            if kind == "other-company":
                found = self.other_company(ticker, question, question_id, k)
            else:
                found = self.other_item(ticker, question, correct, k)
            if found:
                hits, description = found
                reranked = [Hit(h.chunk, h.score, rank) for rank, h in enumerate(hits, 1)]
                return reranked, description
        raise ConfigError(
            f"{question_id}: no mismatched passages possible (no other company indexed and "
            "no chunk from another Item)"
        )
