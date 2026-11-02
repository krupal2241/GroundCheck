import sys
import types

import numpy as np
import pytest

from groundcheck.chunking.base import Chunk, chunks_path, write_chunks
from groundcheck.config import ConfigError
from groundcheck.retrieval import embedder as embedder_module
from groundcheck.retrieval.embedder import (
    BGE_QUERY_PREFIX,
    BgeEmbedder,
    HashingEmbedder,
    make_embedder,
    normalize,
)
from groundcheck.retrieval.index import CompanyIndex, build_company_index, build_index

TEXTS = [
    "Total net sales increased 12% to $12,345 million driven by iPhone.",
    "Competition in the smartphone market could reduce our gross margins.",
    "The company repurchased $90 billion of common stock during the year.",
    "Our headquarters campus in Cupertino, California is owned.",
]


def make_chunks(ticker="EXCO", texts=TEXTS):
    return [
        Chunk(
            chunk_id=f"{ticker}-7-p{i:04d}",
            ticker=ticker,
            accession="0000000001-25-000001",
            item="7",
            section_title="MD&A",
            kind="prose",
            index=i,
            text=t,
            word_count=len(t.split()),
            chunker="fixed-p250-o50-t120",
        )
        for i, t in enumerate(texts)
    ]


# --- embedders --------------------------------------------------------------------


def test_hashing_embedder_is_normalized_and_deterministic():
    e = HashingEmbedder(dim=64)
    a, b = e.embed_documents(TEXTS), HashingEmbedder(dim=64).embed_documents(TEXTS)
    assert a.shape == (4, 64) and a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(a, b)


def test_hashing_embedder_ranks_word_overlap_first():
    e = HashingEmbedder()
    q = e.embed_queries(["How much common stock was repurchased?"])
    scores = e.embed_documents(TEXTS) @ q[0]
    assert int(np.argmax(scores)) == 2


def test_normalize_leaves_zero_rows_alone():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer: no model download."""

    instances: list = []

    def __init__(self, name, device=None):
        self.name, self.device, self.max_seq_length = name, device, None
        self.calls = []
        self.tokenizer = lambda text, add_special_tokens=True: {"input_ids": text.split() + [0, 0]}
        FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.ones((len(texts), 8), dtype=np.float64)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embedder_module, "pick_device", lambda: "mps")
    FakeSentenceTransformer.instances.clear()
    return FakeSentenceTransformer


def test_bge_embedder_settings(fake_sentence_transformers):
    e = BgeEmbedder()
    model = fake_sentence_transformers.instances[0]
    assert model.name == "BAAI/bge-small-en-v1.5"
    assert model.device == "mps" and model.max_seq_length == 512
    assert e.name == "BAAI/bge-small-en-v1.5" and e.dim == 8
    docs = e.embed_documents(["a passage"])
    assert docs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-6)
    e.embed_queries(["a question"])
    (doc_texts, doc_kwargs), (query_texts, _) = model.calls
    assert doc_texts == ["a passage"]  # passages get no instruction prefix
    assert query_texts == [BGE_QUERY_PREFIX + "a question"]
    assert doc_kwargs["normalize_embeddings"] is True
    assert e.token_counts(["one two three"]) == [5]


def test_bge_embedder_explicit_cpu(fake_sentence_transformers):
    BgeEmbedder(device="cpu")
    assert fake_sentence_transformers.instances[0].device == "cpu"


def test_bge_without_extra_is_a_clean_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails
    with pytest.raises(ConfigError, match=r"\[embed\] extra"):
        BgeEmbedder()


def test_make_embedder():
    assert make_embedder("hashing").name == "hashing-256"
    with pytest.raises(ConfigError, match="unknown embedder"):
        make_embedder("word2vec")


# --- FAISS --------------------------------------------------------------------------


def test_index_flat_ip_is_exact_cosine():
    rng = np.random.default_rng(0)
    vectors = normalize(rng.normal(size=(50, 16)))
    index = build_index(vectors)
    query = normalize(rng.normal(size=(1, 16)))
    scores, ids = index.search(query, 5)
    expected = np.argsort(-(vectors @ query[0]))[:5]
    np.testing.assert_array_equal(ids[0], expected)
    np.testing.assert_allclose(scores[0], (vectors @ query[0])[expected], rtol=1e-5)


def test_build_load_and_search(tmp_path):
    chunks_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    write_chunks(make_chunks(), chunks_path("EXCO", chunks_dir))
    e = HashingEmbedder()
    assert build_company_index("EXCO", e, chunks_dir, index_dir) == 4
    index = CompanyIndex.load("EXCO", chunks_dir, index_dir)
    hits = index.search(e, "What were total net sales?", k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.chunk_id == "EXCO-7-p0000"
    assert hits[0].score >= hits[1].score
    assert len(index.search(e, "anything", k=50)) == 4  # k capped at index size


def test_one_index_per_company(tmp_path):
    chunks_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    write_chunks(make_chunks("AAA", TEXTS[:2]), chunks_path("AAA", chunks_dir))
    write_chunks(make_chunks("BBB", TEXTS[2:]), chunks_path("BBB", chunks_dir))
    e = HashingEmbedder()
    for t in ["AAA", "BBB"]:
        build_company_index(t, e, chunks_dir, index_dir)
    hits = CompanyIndex.load("AAA", chunks_dir, index_dir).search(e, "repurchased stock", k=10)
    assert {h.chunk.ticker for h in hits} == {"AAA"}  # never another company's chunks


def test_stale_index_is_refused(tmp_path):
    chunks_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    write_chunks(make_chunks(), chunks_path("EXCO", chunks_dir))
    build_company_index("EXCO", HashingEmbedder(), chunks_dir, index_dir)
    write_chunks(make_chunks(texts=TEXTS[:3]), chunks_path("EXCO", chunks_dir))
    with pytest.raises(ConfigError, match="changed since the index was built"):
        CompanyIndex.load("EXCO", chunks_dir, index_dir)


def test_embedder_mismatch_is_refused(tmp_path):
    chunks_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    write_chunks(make_chunks(), chunks_path("EXCO", chunks_dir))
    build_company_index("EXCO", HashingEmbedder(), chunks_dir, index_dir)
    index = CompanyIndex.load("EXCO", chunks_dir, index_dir)
    with pytest.raises(ConfigError, match="built with"):
        index.search(HashingEmbedder(dim=128), "net sales")


def test_missing_index_is_a_clean_error(tmp_path):
    with pytest.raises(ConfigError, match="run `groundcheck index`"):
        CompanyIndex.load("EXCO", tmp_path, tmp_path)
