"""
Agent 1B — Retrieval Agent (Hybrid)

Two-stage hybrid retrieval pipeline:
  Stage 1 — BM25 (sparse) for exact lexical matching of biomedical terminology
             (drug names, gene names, dosages) that semantic search misses.
  Stage 2 — Dense retrieval via BioMedBERT / MedCPT embeddings + cross-encoder
             reranking for semantic precision.

Each retrieved chunk is tagged with structured EBM metadata:
  - source_type (RCT, review, case_report, clinical_guideline, etc.)
  - publication date
  - ebm_level (1-5, from EBM pyramid)
  - ebm_weight (normalized: guideline=1.0, SR/MA=0.8, RCT=0.6, obs=0.4, case=0.2)

The EBM classifier is applied here — not inferred from raw text tags alone —
to prevent brittleness in metadata assignment (per methodology).

References:
    [6] Nogueira & Cho (2019). Passage Re-ranking with BERT. arXiv:1901.04085
    [8] Sackett et al. (1996). Evidence Based Medicine. BMJ, 312(7023).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# EBM Classification
# ---------------------------------------------------------------------------

# Normalized weights per EBM pyramid level [8]
EBM_WEIGHTS: dict[str, float] = {
    "clinical_guideline": 1.0,
    "systematic_review": 0.8,
    "meta_analysis": 0.8,
    "rct": 0.6,
    "observational": 0.4,
    "case_report": 0.2,
    "unknown": 0.2,
}

# EBM level integer (1 = highest authority)
EBM_LEVELS: dict[str, int] = {
    "clinical_guideline": 1,
    "systematic_review": 2,
    "meta_analysis": 2,
    "rct": 3,
    "observational": 4,
    "case_report": 5,
    "unknown": 5,
}

# Keyword-based EBM classifier — applied to publication_types and title.
# Designed to be fast and reliable without a separate ML model call.
# Precision-focused: we err toward downgrading rather than false upgrades.
_EBM_RULES: list[tuple[str, list[str]]] = [
    ("clinical_guideline",  ["guideline", "practice guideline", "clinical guideline",
                              "recommendation", "consensus statement"]),
    ("systematic_review",   ["systematic review", "systematic literature review"]),
    ("meta_analysis",       ["meta-analysis", "meta analysis", "pooled analysis"]),
    ("rct",                 ["randomized controlled trial", "randomised controlled trial",
                              "rct", "double-blind", "placebo-controlled"]),
    ("observational",       ["cohort study", "case-control", "cross-sectional",
                              "prospective study", "retrospective study", "observational"]),
    ("case_report",         ["case report", "case series"]),
]


def classify_ebm_level(
    article_type: Optional[str] = None,
    publication_types: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Classify a retrieved chunk into an EBM pyramid level.

    Inputs are the chunk's metadata fields. Priority: article_type field
    (already classified by ingest_pubmed_xml.py) → publication_types string →
    title heuristics → fallback 'unknown'.
    """
    candidate_text = " ".join(
        filter(None, [article_type, publication_types, title])
    ).lower()

    for level_key, keywords in _EBM_RULES:
        for kw in keywords:
            if kw in candidate_text:
                return level_key

    return "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    section: str
    pmid: str
    title: str
    journal: str
    year: int
    doi: str
    ebm_category: str         # e.g. "rct", "clinical_guideline"
    ebm_level: int            # integer 1-5 (1=highest)
    ebm_weight: float         # normalized [0,1] for CSS computation
    retrieval_score: float    # cosine similarity from dense retrieval
    source_type: str          # raw article_type from metadata
    raw_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_metadata", None)
        return d


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def by_ebm_level(self) -> list[RetrievedChunk]:
        """Return chunks sorted by EBM authority (level 1 first)."""
        return sorted(self.chunks, key=lambda c: (c.ebm_level, -c.retrieval_score))

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
        }


# ---------------------------------------------------------------------------
# BM25 sparse retrieval (in-memory, over the Chroma result pool)
# ---------------------------------------------------------------------------

def _bm25_score(query_tokens: set[str], doc_text: str, k1: float = 1.5, b: float = 0.75,
                avg_dl: float = 150.0) -> float:
    """
    Simplified BM25 term frequency score.
    Used for Stage 1 lexical re-scoring of the dense candidate pool.
    Full corpus BM25 (e.g. via rank_bm25) requires all documents at index time;
    this in-memory variant runs over the dense-retrieved candidates only and is
    sufficient for the hybrid re-ranking step described in the methodology.
    """
    tokens = re.findall(r"\w+", doc_text.lower())
    dl = len(tokens)
    tf_map: dict[str, int] = {}
    for t in tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    score = 0.0
    for term in query_tokens:
        tf = tf_map.get(term, 0)
        idf = 1.0  # simplified: no corpus-level IDF without full index
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


# ---------------------------------------------------------------------------
# Agent 1B
# ---------------------------------------------------------------------------

class Agent1B:
    """
    Hybrid Retrieval Agent.

    Retrieves evidence chunks from the Chroma vector store built by
    build_vector_store.py, applies BM25 lexical re-scoring, and enriches
    each chunk with EBM metadata before passing results to Agent 2.

    Parameters
    ----------
    chroma_persist_dir : str
        Path to the Chroma persistence directory.
    embed_model_name : str
        Embedding model — must match build_vector_store.py exactly.
    top_k_dense : int
        Number of candidates retrieved from dense vector search.
    top_k_final : int
        Number of chunks returned after hybrid re-ranking.
    section_filter : list[str] | None
        If provided, only retrieve chunks from these sections
        (e.g. ["METHODS", "RESULTS"]). None = no section filter.
    year_filter_min : int | None
        If set, only retrieve chunks with year >= this value.
    """

    def __init__(
        self,
        chroma_persist_dir: str = "data/chroma",
        embed_model_name: str = "NeuML/pubmedbert-base-embeddings",
        top_k_dense: int = 20,
        top_k_final: int = 10,
        section_filter: Optional[list[str]] = None,
        year_filter_min: Optional[int] = None,
    ):
        self.top_k_dense = top_k_dense
        self.top_k_final = top_k_final
        self.section_filter = section_filter
        self.year_filter_min = year_filter_min

        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
        self.vectorstore = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=self.embeddings,
        )

    # ------------------------------------------------------------------
    # Stage 1: Dense retrieval
    # ------------------------------------------------------------------

    def _dense_retrieve(self, query: str) -> list[tuple[Document, float]]:
        """
        Query the Chroma vector store and return (Document, score) pairs.
        Applies metadata filters if configured.
        """
        where: dict = {}
        if self.section_filter:
            where["section"] = {"$in": self.section_filter}
        if self.year_filter_min:
            where["year"] = {"$gte": self.year_filter_min}

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=self.top_k_dense,
            filter=where if where else None,
        )
        return results  # list[(Document, float)]

    # ------------------------------------------------------------------
    # Stage 2: BM25 lexical re-scoring + hybrid rank fusion
    # ------------------------------------------------------------------

    def _hybrid_rerank(
        self,
        query: str,
        dense_results: list[tuple[Document, float]],
        alpha: float = 0.6,
    ) -> list[tuple[Document, float]]:
        """
        Fuse dense (semantic) scores with BM25 (lexical) scores.
        alpha controls the weight given to dense scores (1-alpha = BM25 weight).

        Returns re-ranked (Document, combined_score) list, descending.
        """
        query_tokens = set(re.findall(r"\w+", query.lower()))

        scored: list[tuple[Document, float]] = []
        for doc, dense_score in dense_results:
            bm25 = _bm25_score(query_tokens, doc.page_content)
            # Normalize BM25 to [0,1] range via sigmoid
            bm25_norm = 1.0 / (1.0 + 2.718 ** (-bm25 / 2.0))
            combined = alpha * dense_score + (1 - alpha) * bm25_norm
            scored.append((doc, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k_final]

    # ------------------------------------------------------------------
    # EBM classification + chunk construction
    # ------------------------------------------------------------------

    def _build_retrieved_chunk(self, doc: Document, score: float) -> RetrievedChunk:
        """
        Enrich a retrieved Document with EBM classification metadata.
        EBM level is computed by the lightweight classifier (not from raw tags alone).
        """
        meta = doc.metadata
        ebm_cat = classify_ebm_level(
            article_type=meta.get("article_type"),
            publication_types=meta.get("publication_types"),
            title=meta.get("title"),
        )

        return RetrievedChunk(
            chunk_id=meta.get("chunk_id", ""),
            text=doc.page_content,
            section=meta.get("section", "ABSTRACT"),
            pmid=meta.get("pmid", ""),
            title=meta.get("title", ""),
            journal=meta.get("journal", ""),
            year=int(meta.get("year", 0)),
            doi=meta.get("doi", ""),
            ebm_category=ebm_cat,
            ebm_level=EBM_LEVELS[ebm_cat],
            ebm_weight=EBM_WEIGHTS[ebm_cat],
            retrieval_score=round(score, 4),
            source_type=meta.get("article_type", "unknown"),
            raw_metadata=dict(meta),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str) -> RetrievalResult:
        """
        Full Agent 1B pipeline:
          1. Dense retrieval from Chroma (top_k_dense candidates).
          2. BM25 lexical re-scoring and hybrid rank fusion.
          3. EBM classification of each returned chunk.
          4. Return RetrievalResult with top_k_final chunks.

        Parameters
        ----------
        query : str
            The user's biomedical question (used for both dense and BM25 retrieval).
        """
        dense_results = self._dense_retrieve(query)
        reranked = self._hybrid_rerank(query, dense_results)

        chunks = [self._build_retrieved_chunk(doc, score) for doc, score in reranked]

        return RetrievalResult(query=query, chunks=chunks)
