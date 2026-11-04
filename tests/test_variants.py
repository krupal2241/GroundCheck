"""Retrieval variants: BM25, RRF, contextual headers, reranking, bge-base, the variant
config and index/variant guards. Every model here is a fake: no network, no torch."""

import subprocess
import sys
import types

import numpy as np
import pytest

from groundcheck.chunking.base import Chunk, chunks_path, read_chunks, write_chunks
from groundcheck.config import ConfigError
from groundcheck.retrieval import embedder as embedder_module
from groundcheck.retrieval.bm25 import BM25, tokenize
from groundcheck.retrieval.embedder import BGE_QUERY_PREFIX, HashingEmbedder, make_embedder
from groundcheck.retrieval.rerank import BgeReranker, OverlapReranker, make_reranker
from groundcheck.retrieval.retriever import (
    build_variant_index,
    open_retriever,
    rrf_merge,
    variant_index_dir,
)
from groundcheck.retrieval.variants import (
    RetrievalConfig,
    VariantSpec,
    contextual_header,
    embedding_key,
    fiscal_years,
    load_retrieval_config,
    model_text,
    recommend_default,
)

CHUNKS = [
    ("1", "Business", "prose", "Example Corp designs and sells widgets and gadgets worldwide."),
    ("1A", "Risk Factors", "prose", "Competition could reduce our gross margin and market share."),
    ("7", "MD&A", "prose", "Net sales increased 6% in 2025, driven by higher widget volumes."),
    (
        "7",
        "MD&A",
        "table",
        "[TABLE]\n | 2025 | 2024\nGreater Region | $64,377 | $66,952\n"
        "Total net sales | $416,161 | $391,035\n[/TABLE]",
    ),
    ("8", "Financial Statements", "prose", "The Company repurchased $89.3 billion of stock."),
]


def make_chunks(ticker="EXCO"):
    return [
        Chunk(
            chunk_id=f"{ticker}-{item}-{kind[0]}{i:04d}",
            ticker=ticker,
            accession="0000000001-25-000001",
            item=item,
            section_title=title,
            kind=kind,
            index=i,
            text=text,
            word_count=len(text.split()),
            chunker="fixed-p250-o50-t120",
        )
        for i, (item, title, kind, text) in enumerate(CHUNKS)
    ]


CONFIG = RetrievalConfig.model_validate(
    {
        "default": "dense",
        "pool": 4,
        "variants": {
            "dense": {"embedder": "hashing"},
            "dense-headers": {"embedder": "hashing", "headers": True},
            "hybrid": {"embedder": "hashing", "bm25": True},
            "hybrid-headers": {"embedder": "hashing", "headers": True, "bm25": True},
            "hybrid-headers-rerank": {
                "embedder": "hashing",
                "headers": True,
                "bm25": True,
                "rerank": "overlap",
            },
            "bm25": {"embedder": None, "bm25": True},
        },
    }
)
YEARS = {"EXCO": "FY2025"}


class RecordingEmbedder(HashingEmbedder):
    """The hashing fake, remembering every text it was asked to embed."""

    def __init__(self):
        super().__init__()
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts):
        self.documents += texts
        return super().embed_documents(texts)

    def embed_queries(self, texts):
        self.queries += texts
        return HashingEmbedder.embed_documents(self, texts)  # not recorded as a document


class ScriptedReranker:
    """Scores passages by a fixed preference list and records what it was shown."""

    name = "scripted"

    def __init__(self, prefer: str):
        self.prefer = prefer
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [1.0 if self.prefer in p else 0.0 for p in passages]


@pytest.fixture
def corpus(tmp_path):
    chunks_dir, index_dir = tmp_path / "chunks", tmp_path / "index"
    write_chunks(make_chunks(), chunks_path("EXCO", chunks_dir))
    return chunks_dir, index_dir


def build(name, corpus, embedder=None, years=YEARS):
    chunks_dir, index_dir = corpus
    spec = CONFIG.variants[name]
    return build_variant_index(
        name, spec, "EXCO", embedder or HashingEmbedder(), chunks_dir, index_dir, years["EXCO"]
    )


def open_variant(name, corpus, **kwargs):
    chunks_dir, index_dir = corpus
    return open_retriever(
        name, CONFIG.variants[name], CONFIG, ["EXCO"], chunks_dir, index_dir, YEARS, **kwargs
    )


# --- BM25 and RRF -------------------------------------------------------------------------


def test_tokenize_keeps_numbers_whole_and_drops_stopwords():
    assert tokenize("What were Greater Region sales of $64,377 in 2025?") == [
        "greater",
        "region",
        "sales",
        "64,377",
        "2025",
    ]


def test_bm25_ranks_rare_exact_terms_first():
    texts = [t for *_, t in CHUNKS]
    bm25 = BM25(texts)
    assert bm25.top("Greater Region 64,377", 3)[0][0] == 3
    assert bm25.top("repurchased stock", 1)[0][0] == 4
    assert bm25.idf("sales") < bm25.idf("gadgets")  # in two chunks vs one
    assert bm25.top("zebra", 5) == []  # nothing scores above zero
    assert BM25([]).top("anything", 3) == []


def test_bm25_length_normalization():
    bm25 = BM25(["net sales", "net sales " + "filler " * 50])
    scores = bm25.scores("net sales")
    assert scores[0] > scores[1] > 0


def test_rrf_merge_by_hand():
    merged = rrf_merge([[0, 1, 2], [2, 0, 3]], rrf_k=60)
    expected = {0: 1 / 61 + 1 / 62, 1: 1 / 62, 2: 1 / 63 + 1 / 61, 3: 1 / 63}
    assert [p for p, _ in merged] == [0, 2, 1, 3]
    for pos, score in merged:
        assert score == pytest.approx(expected[pos])
    # A tie goes to the better single rank: 5 (rank 1) before 6 (rank 2 and 2).
    assert [p for p, _ in rrf_merge([[5, 6], [7, 6]], rrf_k=1)][:1] == [6]


# --- contextual headers ---------------------------------------------------------------------


def test_header_format():
    chunk = make_chunks()[3]
    assert contextual_header(chunk, "FY2025") == "EXCO FY2025 10-K | Item 7 MD&A | table"
    assert contextual_header(chunk, None) == "EXCO 10-K | Item 7 MD&A | table"
    assert model_text(chunk, False, "FY2025") == chunk.text
    assert model_text(chunk, True, "FY2025") == (
        "EXCO FY2025 10-K | Item 7 MD&A | table\n" + chunk.text
    )


def test_fiscal_year_comes_from_the_period_of_report():
    from groundcheck.config import CompaniesConfig

    companies = CompaniesConfig.model_validate(
        {
            "companies": [
                {"ticker": "WMT", "name": "W", "cik": 1, "role": "labeled", "period": "2026-01-31"},
                {"ticker": "NOPE", "name": "N", "cik": 2, "role": "labeled"},
            ]
        }
    )
    assert fiscal_years(companies) == {"WMT": "FY2026"}


def test_headers_never_modify_stored_chunk_text(corpus):
    chunks_dir, _ = corpus
    path = chunks_path("EXCO", chunks_dir)
    before = path.read_bytes()
    originals = {c.chunk_id: c.text for c in read_chunks(path)}
    embedder = RecordingEmbedder()
    build("hybrid-headers-rerank", corpus, embedder)
    reranker = ScriptedReranker(prefer="64,377")
    retriever = open_variant("hybrid-headers-rerank", corpus, embedder=embedder, reranker=reranker)
    hits = retriever.search("EXCO", "Greater Region net sales", 3)

    # The model saw header + text ...
    assert all(t.startswith("EXCO FY2025 10-K | Item ") for t in embedder.documents)
    assert all(p.startswith("EXCO FY2025 10-K | Item ") for p in reranker.calls[0][1])
    assert embedder.queries == ["Greater Region net sales"]  # queries get no header
    # ... but the chunks, on disk and in every hit, are byte-for-byte unchanged.
    assert path.read_bytes() == before
    assert all(h.chunk.text == originals[h.chunk.chunk_id] for h in hits)
    assert not any("10-K |" in h.chunk.text for h in hits)
    for company in retriever.companies.values():
        assert all(c.text == originals[c.chunk_id] for c in company.chunks)


# --- every variant, with fakes ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["dense", "dense-headers", "hybrid", "hybrid-headers"])
def test_dense_and_hybrid_variants_find_the_table_row(corpus, name):
    build(name, corpus)
    hits = open_variant(name, corpus).search("EXCO", "Greater Region 64,377 net sales", 3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].chunk.chunk_id == "EXCO-7-t0003"
    assert len({h.chunk.chunk_id for h in hits}) == 3


def test_hybrid_merges_both_rankings_with_rrf(corpus):
    build("hybrid", corpus)
    hybrid = open_variant("hybrid", corpus)
    dense = open_variant("dense", corpus)
    query = "Did the company buy back stock?"
    hits = hybrid.search("EXCO", query, 3)  # pool = max(k, 4) = 4 from each ranking
    company = hybrid.companies["EXCO"]
    dense_order = [h.chunk.chunk_id for h in dense.search("EXCO", query, 4)]
    lex_order = [company.chunks[p].chunk_id for p, _ in company.bm25.top(query, 4)]
    ids = [c.chunk_id for c in company.chunks]
    expected = rrf_merge(
        [[ids.index(i) for i in dense_order], [ids.index(i) for i in lex_order]], 60
    )
    assert [h.chunk.chunk_id for h in hits] == [ids[p] for p, _ in expected][:3]
    assert hits[0].score == pytest.approx(expected[0][1])


def test_bm25_only_variant_needs_no_model_and_no_index(corpus):
    chunks_dir, index_dir = corpus
    retriever = open_variant("bm25", corpus)
    assert retriever.embedder is None and not index_dir.exists()
    hits = retriever.search("EXCO", "repurchased stock billion", 2)
    assert hits[0].chunk.chunk_id == "EXCO-8-p0004"
    assert retriever.description == "bm25"
    with pytest.raises(ConfigError, match="no embedder"):
        build("bm25", corpus)


def test_rerank_reorders_the_top_pool(corpus):
    build("hybrid-headers-rerank", corpus)
    reranker = ScriptedReranker(prefer="Competition")
    retriever = open_variant("hybrid-headers-rerank", corpus, reranker=reranker)
    hits = retriever.search("EXCO", "net sales widgets gross margin competition", 2)
    assert hits[0].chunk.chunk_id == "EXCO-1A-p0001"  # the reranker's pick, moved to 1
    assert hits[0].score == 1.0
    ((_, passages),) = reranker.calls
    assert len(passages) == CONFIG.pool  # reranks the pool (4), returns k (2)
    assert len(hits) == 2


def test_overlap_reranker_variant_from_config(corpus):
    build("hybrid-headers-rerank", corpus)
    retriever = open_variant("hybrid-headers-rerank", corpus)  # makes the overlap fake
    assert isinstance(retriever.reranker, OverlapReranker)
    assert "rerank=overlap" in retriever.description and "pool=4" in retriever.description
    assert retriever.search("EXCO", "repurchased stock", 1)[0].chunk.chunk_id == "EXCO-8-p0004"


def test_overlap_reranker_scores():
    scores = OverlapReranker().score("net sales 2025", ["Net sales rose in 2025.", "Other."])
    assert scores == [1.0, 0.0]
    assert OverlapReranker().score("the", ["x"]) == [0.0]


def test_variants_with_the_same_embedding_share_one_index(corpus):
    _, index_dir = corpus
    _, folder = build("dense", corpus)
    e = HashingEmbedder()
    assert variant_index_dir(index_dir, e, CONFIG.variants["hybrid"]) == folder.parent
    assert variant_index_dir(index_dir, e, CONFIG.variants["dense-headers"]) != folder.parent
    open_variant("hybrid", corpus)  # loads the index "dense" built
    assert embedding_key("BAAI/bge-small-en-v1.5", True) == "BAAI/bge-small-en-v1.5+headers-h1"


# --- guards ---------------------------------------------------------------------------------


def test_missing_variant_index_names_the_variant(corpus):
    build("dense", corpus)
    with pytest.raises(ConfigError, match="groundcheck index --variant dense-headers"):
        open_variant("dense-headers", corpus)


def test_index_from_a_mismatching_variant_is_refused(corpus):
    _, index_dir = corpus
    _, plain = build("dense", corpus)
    # Put the plain-text index where the header variant looks for its own.
    target = variant_index_dir(index_dir, HashingEmbedder(), CONFIG.variants["dense-headers"])
    target.mkdir(parents=True)
    plain.rename(target / "EXCO")
    with pytest.raises(ConfigError, match="built by variant 'dense'.*needs hashing-256"):
        open_variant("dense-headers", corpus)


def test_index_with_other_header_texts_is_refused(corpus):
    build("dense-headers", corpus, years={"EXCO": "FY2024"})  # e.g. companies.yaml changed
    with pytest.raises(ConfigError, match="from different texts"):
        open_variant("dense-headers", corpus)


def test_query_embedder_must_match_the_index(corpus):
    build("dense", corpus)
    retriever = open_variant("dense", corpus)
    retriever.embedder = HashingEmbedder(dim=128)
    with pytest.raises(ConfigError, match="built with"):
        retriever.search("EXCO", "net sales", 2)


def test_unknown_ticker_is_a_clean_error(corpus):
    with pytest.raises(ConfigError, match="no index loaded for OTHR"):
        open_variant("bm25", corpus).search("OTHR", "x", 1)


# --- config ---------------------------------------------------------------------------------


def test_committed_retrieval_config_is_valid():
    config = load_retrieval_config()
    names = set(config.variants)
    assert config.default in names
    assert {"dense", "dense-headers", "hybrid", "hybrid-headers-rerank", "dense-base"} <= names
    assert config.variants["dense"] == VariantSpec(
        description=config.variants["dense"].description, embedder="bge-small"
    )
    assert config.variants["dense-base"].embedder == "bge-base"
    assert config.pool == 30
    assert any(v.rerank == "bge-reranker-base" for v in config.variants.values())
    # The default must be a real candidate and run with the default install ([embed]).
    default = config.variants[config.default]
    assert not default.diagnostic and default.rerank is None
    assert default.embedder == "bge-small"
    assert config.variants["bm25"].diagnostic


def test_config_validation(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("default: nope\nvariants: {a: {embedder: hashing}}\n")
    with pytest.raises(ConfigError, match="not one of the variants"):
        load_retrieval_config(path)
    path.write_text("default: a\nvariants: {a: {embedder: null}}\n")
    with pytest.raises(ConfigError, match="needs an embedder, bm25"):
        load_retrieval_config(path)
    path.write_text("default: a\nvariants: {a: {embedder: hashing, bm52: true}}\n")
    with pytest.raises(ConfigError, match="bm52"):
        load_retrieval_config(path)
    with pytest.raises(ConfigError, match="not found"):
        load_retrieval_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="unknown retrieval variant 'x'"):
        CONFIG.get("x")


def test_overrides_swap_models_but_keep_stages():
    spec = CONFIG.variants["hybrid-headers-rerank"].with_overrides("hashing", "overlap")
    assert (spec.embedder, spec.headers, spec.bm25, spec.rerank) == (
        "hashing",
        True,
        True,
        "overlap",
    )
    # A BM25-only variant stays model-free; a variant without rerank gets none.
    assert CONFIG.variants["bm25"].with_overrides("hashing").embedder is None
    assert CONFIG.variants["dense"].with_overrides(reranker="overlap").rerank is None


def test_recommend_default_prefers_the_cheaper_variant_within_the_tolerance():
    rows = [
        {"variant": "dense", "recall": {5: 0.50}, "latency_ms": 10.0},
        {"variant": "hybrid", "recall": {5: 0.60}, "latency_ms": 12.0},
        {"variant": "rerank", "recall": {5: 0.64}, "latency_ms": 900.0},
        {"variant": "base", "recall": {5: 0.66}, "latency_ms": 40.0},
    ]
    name, reason = recommend_default(rows)
    assert name == "base"  # hybrid (0.60) is more than 0.05 below the best (0.66)
    assert "best recall@5 is 0.66" in reason
    rows[3]["recall"][5] = 0.63
    assert recommend_default(rows)[0] == "hybrid"  # 0.60 is within 0.05 of 0.64, cheaper
    assert recommend_default([]) is None


# --- real-model wrappers, against stand-in modules -------------------------------------------


class FakeSentenceTransformer:
    def __init__(self, name, device=None):
        self.name, self.device, self.max_seq_length = name, device, None
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 6

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.ones((len(texts), 6))


class FakeCrossEncoder:
    instances: list = []

    def __init__(self, name, max_length=None, device=None):
        self.name, self.max_length, self.device = name, max_length, device
        self.pairs = []
        FakeCrossEncoder.instances.append(self)

    def predict(self, pairs, **kwargs):
        self.pairs.append(list(pairs))
        return np.array([float(len(p)) for _, p in pairs])


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embedder_module, "pick_device", lambda: "cpu")
    FakeCrossEncoder.instances.clear()


def test_bge_base_variant_loads_the_base_model(fake_st):
    e = make_embedder("bge-base")
    assert e.name == "BAAI/bge-base-en-v1.5" and e.model.max_seq_length == 512
    e.embed_queries(["q"])
    assert e.model.calls == [[BGE_QUERY_PREFIX + "q"]]  # same instruction as bge-small
    assert make_embedder("bge-small").name == "BAAI/bge-small-en-v1.5"


def test_bge_reranker_wrapper(fake_st):
    reranker = make_reranker("bge-reranker-base")
    assert isinstance(reranker, BgeReranker)
    model = FakeCrossEncoder.instances[0]
    assert model.name == "BAAI/bge-reranker-base" and model.max_length == 512
    assert model.device == "cpu"
    assert reranker.score("q", ["ab", "abcd"]) == [2.0, 4.0]
    assert model.pairs == [[("q", "ab"), ("q", "abcd")]]
    assert reranker.score("q", []) == []


def test_reranker_without_extra_is_a_clean_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ConfigError, match=r"\[rerank\] extra"):
        BgeReranker()
    with pytest.raises(ConfigError, match="unknown reranker"):
        make_reranker("colbert")


def test_importing_the_package_loads_no_model_library():
    code = (
        "import sys\n"
        "import groundcheck.cli, groundcheck.generation.runner\n"
        "import groundcheck.retrieval.retriever, groundcheck.retrieval.rerank\n"
        "bad = [m for m in ('torch', 'sentence_transformers', 'transformers') "
        "if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
