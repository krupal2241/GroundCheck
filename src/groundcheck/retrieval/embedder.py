"""Turn text into unit-length vectors, behind one `Embedder` interface.

- `BgeEmbedder`: `BAAI/bge-small-en-v1.5` (384-dim) or `BAAI/bge-base-en-v1.5`
  (768-dim) through sentence-transformers, on Apple MPS if available, else CPU. Needs
  the `[embed]` extra and downloads the model from Hugging Face on first use (so it
  only runs on your machine, not in CI).
- `HashingEmbedder`: a deterministic fake for tests. Each lowercase word is hashed
  into one of `dim` buckets, so texts that share words get similar vectors. It needs no
  model and no network.

All vectors are float32 and L2-normalized, so the inner product FAISS computes
(`IndexFlatIP`) is exactly cosine similarity.
"""

import hashlib
import re
from collections.abc import Callable
from typing import Protocol

import numpy as np

from groundcheck.config import ConfigError

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_BASE_MODEL = "BAAI/bge-base-en-v1.5"
# Embedder kinds selectable in config/retrieval.yaml and on the command line.
EMBEDDERS = {"bge-small": BGE_MODEL, "bge-base": BGE_BASE_MODEL, "hashing": None}
# The query instruction recommended for bge v1.5 retrieval. Passages get no prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_MAX_TOKENS = 512

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows. All-zero rows (e.g. an empty text) stay zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    """Bag-of-words hashed into `dim` buckets. Deterministic across runs and machines."""

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                # hashlib, not hash(): Python salts str hashes per process.
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                out[row, int.from_bytes(digest, "little") % self._dim] += 1.0
        return normalize(out)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


def pick_device() -> str:
    """'mps' on Apple silicon when PyTorch supports it, else 'cpu'."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class BgeEmbedder:
    """A bge v1.5 model via sentence-transformers (small: 384-dim, base: 768-dim). Both
    use the same query instruction and 512-token limit."""

    def __init__(self, model_name: str = BGE_MODEL, device: str | None = None, batch_size=32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                'the embedding model needs the [embed] extra: pip install -e ".[embed]"'
            ) from exc
        self.model_name = model_name
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = BGE_MAX_TOKENS

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return normalize(vectors)  # already unit length; also forces float32

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([BGE_QUERY_PREFIX + t for t in texts])

    def token_counts(self, texts: list[str]) -> list[int]:
        """Real token counts (with special tokens), to find chunks that get truncated."""
        tokenizer = self.model.tokenizer
        return [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]


def load_bge_token_counter(model_name: str = BGE_MODEL) -> Callable[[str], int]:
    """A function counting a text's bge tokens (special tokens included), as the model
    sees them. Only the tokenizer files are loaded (a few hundred KB, cached by Hugging
    Face after the first download), not the model."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ConfigError(
            'counting tokens needs the [embed] extra: pip install -e ".[embed]" '
            "(or pass --tokenizer none to skip the token limit)"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        # verbose=False: no "sequence length is longer than ..." warning per long text.
        return len(tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"])

    return count


def make_embedder(kind: str, device: str | None = None) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if EMBEDDERS.get(kind):
        return BgeEmbedder(EMBEDDERS[kind], device=device)
    raise ConfigError(f"unknown embedder {kind!r}; use one of {', '.join(EMBEDDERS)}")
