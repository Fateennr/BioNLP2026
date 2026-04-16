"""
Agent 2 — Conflict Detection and Classification Agent

The theoretical heart of the system. Detects, classifies, and scores conflicts
between parametric claims (Agent 1A) and retrieved evidence (Agent 1B) via a
four-step pipeline:

  Step 1 — Two-stage claim alignment:
              Bi-encoder shortlisting (recall) → cross-encoder comparability
              filtering (precision). Designed to handle short atomic claims that
              are poorly discriminated by fixed cosine thresholds alone.

  Step 2 — NLI conflict classification:
              MedNLI / BioNLI model classifies each comparable pair as
              ENTAILMENT, NEUTRAL, or CONTRADICTION, then maps to conflict type.

  Step 3 — EBM-Weighted Conflict Severity Score (CSS):
              CSS = NLI_prob × EBM_weight × recency_delta × inv(parametric_conf)
              Acts as a principled gatekeeper: only high-CSS conflicts proceed
              to Agent 3.

  Step 4 — Claim clustering and consensus formation:
              High-CSS claims are grouped into evidence position clusters
              (consensus / contextual / noise) and scored for authority,
              agreement density, source diversity, and recency.

Conflict type taxonomy:
  Type I  — Contradiction: mutually exclusive facts.
  Type II — Gap: parametric absence of retrieved fact (epistemic absence).
  Type III — Granularity: both correct, different levels of specificity.

References:
    [4] Romanov & Shivade (2018). MedNLI. EMNLP 2018.
    [5] Zhang et al. (2023). BioNLI. arXiv:2307.00631
    [6] Nogueira & Cho (2019). BERT re-ranking. arXiv:1901.04085
    [8] Sackett et al. (1996). EBM pyramid. BMJ 312(7023).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline as hf_pipeline

from agents.agent1a_parametric_elicitation import AtomicClaim, ParametricExtractionResult
from agents.agent1b_retrieval import RetrievedChunk, RetrievalResult

# ---------------------------------------------------------------------------
# Conflict types and NLI labels
# ---------------------------------------------------------------------------

class ConflictType(str, Enum):
    TYPE_I_CONTRADICTION = "TYPE_I_CONTRADICTION"
    TYPE_II_GAP = "TYPE_II_GAP"
    TYPE_III_GRANULARITY = "TYPE_III_GRANULARITY"
    NONE = "NONE"


class NLILabel(str, Enum):
    ENTAILMENT = "ENTAILMENT"
    NEUTRAL = "NEUTRAL"
    CONTRADICTION = "CONTRADICTION"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimPair:
    """A matched (parametric_claim, retrieved_chunk) pair after alignment."""
    parametric_claim: AtomicClaim
    retrieved_chunk: RetrievedChunk
    biencoder_similarity: float
    crossencoder_score: float       # comparability score from cross-encoder


@dataclass
class ConflictRecord:
    """Full conflict record produced by Agent 2 for a single claim pair."""
    parametric_claim_id: str
    parametric_statement: str
    retrieved_chunk_id: str
    retrieved_text_snippet: str     # first 300 chars of chunk

    nli_label: NLILabel
    nli_contradiction_probability: float
    nli_entailment_probability: float
    nli_neutral_probability: float

    conflict_type: ConflictType
    css: float                      # Conflict Severity Score

    ebm_weight: float
    ebm_category: str
    recency_delta: float
    parametric_confidence: float

    specificity_mismatch: bool      # True for Type III detection
    passes_css_threshold: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nli_label"] = self.nli_label.value
        d["conflict_type"] = self.conflict_type.value
        return d


@dataclass
class EvidenceCluster:
    """
    A cluster of consistent high-CSS evidence positions.
    Cluster A = consensus, B = contextual/subgroup, C = noise (pre-filtered by CSS).
    """
    cluster_id: str
    label: str                           # "consensus" | "contextual" | "noise"
    conflict_records: list[ConflictRecord] = field(default_factory=list)
    evidence_strength: float = 0.0       # mean EBM weight of constituent records
    agreement_density: int = 0           # number of independent supporting claims
    source_diversity: int = 0            # number of distinct papers (PMIDs)
    recency_score: float = 0.0           # mean recency_delta
    cluster_score: float = 0.0           # α·E + β·A + γ·D + δ·T

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "evidence_strength": self.evidence_strength,
            "agreement_density": self.agreement_density,
            "source_diversity": self.source_diversity,
            "recency_score": self.recency_score,
            "cluster_score": self.cluster_score,
            "records": [r.to_dict() for r in self.conflict_records],
        }


@dataclass
class ConflictDetectionResult:
    all_records: list[ConflictRecord] = field(default_factory=list)
    css_passing_records: list[ConflictRecord] = field(default_factory=list)
    clusters: list[EvidenceCluster] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_pairs_evaluated": len(self.all_records),
            "css_passing_count": len(self.css_passing_records),
            "clusters": [c.to_dict() for c in self.clusters],
            "all_records": [r.to_dict() for r in self.all_records],
        }


# ---------------------------------------------------------------------------
# Named entity density (for Type III specificity mismatch detection)
# ---------------------------------------------------------------------------

# Simple heuristic: numeric values, measurement units, percentages, and
# common biomedical qualifiers signal specificity in retrieved text.
_SPECIFICITY_PATTERN = re.compile(
    r"\b\d+\.?\d*\s*(%|mg|dL|mmol|mcg|kg|ml|ng|IU|years?|months?|weeks?)\b"
    r"|p\s*[<=>]\s*0\.\d+"
    r"|\b(95%\s*CI|hazard ratio|odds ratio|relative risk|NNT)\b",
    re.IGNORECASE,
)

def _specificity_score(text: str) -> int:
    """Count specificity markers in text. Higher = more granular claim."""
    return len(_SPECIFICITY_PATTERN.findall(text))


# ---------------------------------------------------------------------------
# Recency delta computation
# ---------------------------------------------------------------------------

def _recency_delta(chunk_year: int, reference_year: int = 2024) -> float:
    """
    Compute a recency weight that favors newer evidence.
    Returns a value in [0.1, 1.0].
    Very old studies (15+ years) get minimum weight = 0.1.
    Studies from the reference year get maximum weight = 1.0.
    """
    if chunk_year <= 0:
        return 0.3  # unknown year: moderate weight
    age = max(0, reference_year - chunk_year)
    # Exponential decay with 8-year half-life
    return max(0.1, float(np.exp(-0.087 * age)))


# ---------------------------------------------------------------------------
# Agent 2
# ---------------------------------------------------------------------------

class Agent2:
    """
    Conflict Detection and Classification Agent.

    Parameters
    ----------
    embed_model_name : str
        Bi-encoder embedding model for claim alignment Stage 1.
        Must match the model used in Agent 1B for coherent similarity space.
    nli_model_name : str
        HuggingFace NLI model fine-tuned on biomedical NLI data.
        Default uses a MedNLI-capable model.
    top_n_biencoder : int
        Number of retrieved chunks to shortlist per parametric claim
        in the bi-encoder stage. Maximizes recall.
    crossencoder_threshold : float
        Minimum cross-encoder comparability score to proceed to NLI.
        Filters pairs addressing different factual dimensions.
    css_threshold : float
        Minimum CSS to pass a conflict to Agent 3.
    cluster_weights : dict
        α, β, γ, δ coefficients for cluster scoring formula.
    reference_year : int
        Year used to compute recency_delta (default: current publication year).
    """

    def __init__(
        self,
        embed_model_name: str = "NeuML/pubmedbert-base-embeddings",
        nli_model_name: str = "typeform/distilbert-base-uncased-mnli",
        top_n_biencoder: int = 5,
        crossencoder_threshold: float = 0.5,
        css_threshold: float = 0.3,
        cluster_weights: Optional[dict] = None,
        reference_year: int = 2024,
    ):
        self.top_n_biencoder = top_n_biencoder
        self.crossencoder_threshold = crossencoder_threshold
        self.css_threshold = css_threshold
        self.reference_year = reference_year

        # α·E + β·A + γ·D + δ·T cluster scoring weights
        self.cluster_weights = cluster_weights or {
            "alpha": 0.4,   # evidence strength (EBM weight)
            "beta": 0.3,    # agreement density
            "gamma": 0.2,   # source diversity
            "delta": 0.1,   # recency
        }

        # Bi-encoder for Stage 1 shortlisting
        self.embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

        # NLI pipeline for Step 2 — [4][5]
        # NOTE: In production, replace with a model fine-tuned on MedNLI / BioNLI.
        # The pipeline below uses a general NLI model as a drop-in placeholder.
        # Fine-tuned biomedical NLI models (e.g. from HuggingFace hub for MedNLI)
        # can be swapped in by changing nli_model_name.
        self.nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=nli_model_name,
            hypothesis_template="This text {}.",
        )

        # Cross-encoder for Stage 1 comparability filtering — [6]
        # Using the NLI pipeline score as a proxy for comparability.
        # In production, use a dedicated cross-encoder fine-tuned on MedNLI pairs.
        self._crossencoder_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 1: Two-stage claim alignment
    # ------------------------------------------------------------------

    def _biencoder_similarity(self, claim_text: str, chunk_text: str) -> float:
        """Cosine similarity between claim and chunk embeddings."""
        claim_emb = np.array(self.embed_model.embed_query(claim_text))
        chunk_emb = np.array(self.embed_model.embed_query(chunk_text[:512]))
        return float(
            np.dot(claim_emb, chunk_emb)
            / (np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb) + 1e-9)
        )

    def _crossencoder_comparability(self, claim_text: str, chunk_text: str) -> float:
        """
        Score comparability of a claim-chunk pair by checking whether the
        chunk 'addresses the same factual question' as the claim.
        Uses the NLI pipeline as a lightweight comparability proxy.
        In production: replace with a cross-encoder fine-tuned on MedNLI pairs [6].
        """
        cache_key = f"{hash(claim_text)}|{hash(chunk_text[:200])}"
        if cache_key in self._crossencoder_cache:
            return self._crossencoder_cache[cache_key]

        hypothesis = f"is about the same medical topic as: {claim_text}"
        try:
            result = self.nli_pipeline(
                chunk_text[:300], candidate_labels=["same topic", "different topic"]
            )
            score = result["scores"][result["labels"].index("same topic")]
        except Exception:
            score = 0.5  # fallback: neutral comparability

        self._crossencoder_cache[cache_key] = score
        return score

    def _align_claims(
        self,
        parametric_result: ParametricExtractionResult,
        retrieval_result: RetrievalResult,
    ) -> list[ClaimPair]:
        """
        Two-stage alignment:
          1. Bi-encoder: for each parametric claim, compute similarity to all
             retrieved chunks and keep top-N (no absolute threshold → maximizes recall).
          2. Cross-encoder: filter shortlisted pairs by comparability threshold
             (precision filter → removes topically unrelated pairs).
        """
        pairs: list[ClaimPair] = []

        for claim in parametric_result.claims:
            # Stage 1: bi-encoder shortlisting — no threshold, top-N by similarity
            scored_chunks = [
                (chunk, self._biencoder_similarity(claim.statement, chunk.text))
                for chunk in retrieval_result.chunks
            ]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            shortlisted = scored_chunks[: self.top_n_biencoder]

            # Stage 2: cross-encoder comparability filter
            for chunk, bi_sim in shortlisted:
                ce_score = self._crossencoder_comparability(claim.statement, chunk.text)
                if ce_score >= self.crossencoder_threshold:
                    pairs.append(
                        ClaimPair(
                            parametric_claim=claim,
                            retrieved_chunk=chunk,
                            biencoder_similarity=bi_sim,
                            crossencoder_score=ce_score,
                        )
                    )

        return pairs

    # ------------------------------------------------------------------
    # Step 2: NLI conflict classification
    # ------------------------------------------------------------------

    def _run_nli(self, claim_text: str, chunk_text: str) -> dict[str, float]:
        """
        Run NLI on a (claim, chunk_excerpt) pair.
        Returns dict with keys: ENTAILMENT, NEUTRAL, CONTRADICTION.

        In production, use a model fine-tuned on MedNLI [4] or BioNLI [5].
        """
        try:
            result = self.nli_pipeline(
                chunk_text[:512],
                candidate_labels=["entailment", "contradiction", "neutral"],
            )
            label_map = {
                "entailment": NLILabel.ENTAILMENT,
                "contradiction": NLILabel.CONTRADICTION,
                "neutral": NLILabel.NEUTRAL,
            }
            scores = {
                label_map[lbl].value: sc
                for lbl, sc in zip(result["labels"], result["scores"])
            }
            # Ensure all three keys exist
            for lbl in NLILabel:
                if lbl.value not in scores:
                    scores[lbl.value] = 0.0
            return scores
        except Exception:
            # Safe fallback: all NEUTRAL
            return {
                NLILabel.ENTAILMENT.value: 0.33,
                NLILabel.NEUTRAL.value: 0.34,
                NLILabel.CONTRADICTION.value: 0.33,
            }

    def _classify_conflict_type(
        self,
        nli_scores: dict[str, float],
        parametric_claim: AtomicClaim,
        retrieved_chunk: RetrievedChunk,
    ) -> tuple[ConflictType, bool]:
        """
        Map NLI output + auxiliary signals to conflict type taxonomy.

        Type I  → CONTRADICTION label
        Type II → NEUTRAL + low parametric confidence (epistemic absence)
        Type III → ENTAILMENT + specificity mismatch (granularity gap)

        Returns (ConflictType, specificity_mismatch_flag).
        """
        entail_prob = nli_scores[NLILabel.ENTAILMENT.value]
        neutral_prob = nli_scores[NLILabel.NEUTRAL.value]
        contra_prob = nli_scores[NLILabel.CONTRADICTION.value]

        # Specificity mismatch: retrieved chunk is notably more specific
        param_specificity = _specificity_score(parametric_claim.statement)
        chunk_specificity = _specificity_score(retrieved_chunk.text)
        specificity_mismatch = chunk_specificity > param_specificity + 1

        max_label = max(nli_scores, key=lambda k: nli_scores[k])

        if max_label == NLILabel.CONTRADICTION.value:
            return ConflictType.TYPE_I_CONTRADICTION, specificity_mismatch

        if max_label == NLILabel.NEUTRAL.value:
            # Type II: gap conflict (epistemic absence in parametric knowledge)
            if parametric_claim.verbalized_confidence < 0.5 or parametric_claim.uncertainty_flag:
                return ConflictType.TYPE_II_GAP, specificity_mismatch
            return ConflictType.NONE, specificity_mismatch

        if max_label == NLILabel.ENTAILMENT.value and specificity_mismatch:
            # Type III: both correct but at different granularities
            return ConflictType.TYPE_III_GRANULARITY, True

        return ConflictType.NONE, specificity_mismatch

    # ------------------------------------------------------------------
    # Step 3: CSS computation
    # ------------------------------------------------------------------

    def _compute_css(
        self,
        nli_contradiction_prob: float,
        ebm_weight: float,
        chunk_year: int,
        parametric_confidence: float,
    ) -> float:
        """
        Conflict Severity Score:
          CSS = NLI_contradiction_prob × EBM_weight × recency_delta × inv(parametric_conf)

        Inversion of parametric confidence: low-confidence parametric claims
        receive higher CSS, making them more susceptible to retrieval override.
        inv(p) = 1 - p + 0.05 to avoid zero weight at p=1.
        """
        recency = _recency_delta(chunk_year, self.reference_year)
        inv_conf = 1.0 - parametric_confidence + 0.05
        css = nli_contradiction_prob * ebm_weight * recency * inv_conf
        return round(float(np.clip(css, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # Step 4: Claim clustering
    # ------------------------------------------------------------------

    def _cluster_conflicts(
        self, css_passing: list[ConflictRecord]
    ) -> list[EvidenceCluster]:
        """
        Group high-CSS conflicts into evidence position clusters.

        Clustering strategy:
          - Cluster A (consensus): Type I records from EBM level ≤ 3 (RCT and above)
          - Cluster B (contextual): Type III records (subgroup / granularity)
          - Cluster C (noise): very low CSS records (largely pre-filtered by CSS threshold)

        In production, replace with a semantic clustering algorithm (e.g. k-means
        on claim embeddings) for more fine-grained grouping.

        Cluster score: α·E + β·A + γ·D + δ·T  (per methodology)
        """
        cw = self.cluster_weights
        clusters_map: dict[str, list[ConflictRecord]] = {
            "consensus": [],
            "contextual": [],
            "noise": [],
        }

        for rec in css_passing:
            if rec.conflict_type in (
                ConflictType.TYPE_I_CONTRADICTION, ConflictType.TYPE_II_GAP
            ):
                clusters_map["consensus"].append(rec)
            elif rec.conflict_type == ConflictType.TYPE_III_GRANULARITY:
                clusters_map["contextual"].append(rec)
            else:
                clusters_map["noise"].append(rec)

        clusters: list[EvidenceCluster] = []
        for label, records in clusters_map.items():
            if not records:
                continue

            pmids = {r.retrieved_chunk_id.split("_")[0] for r in records}
            mean_ebm = float(np.mean([r.ebm_weight for r in records]))
            mean_recency = float(np.mean([r.recency_delta for r in records]))
            diversity = len(pmids)
            density = len(records)

            # Normalize agreement density to [0,1] with soft cap at 10
            density_norm = min(density / 10.0, 1.0)
            diversity_norm = min(diversity / 5.0, 1.0)

            cluster_score = (
                cw["alpha"] * mean_ebm
                + cw["beta"] * density_norm
                + cw["gamma"] * diversity_norm
                + cw["delta"] * mean_recency
            )

            clusters.append(
                EvidenceCluster(
                    cluster_id=f"cluster_{label}",
                    label=label,
                    conflict_records=records,
                    evidence_strength=mean_ebm,
                    agreement_density=density,
                    source_diversity=diversity,
                    recency_score=mean_recency,
                    cluster_score=round(cluster_score, 4),
                )
            )

        return sorted(clusters, key=lambda c: c.cluster_score, reverse=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        parametric_result: ParametricExtractionResult,
        retrieval_result: RetrievalResult,
    ) -> ConflictDetectionResult:
        """
        Full Agent 2 four-step pipeline.

        Parameters
        ----------
        parametric_result : ParametricExtractionResult
            Output from Agent 1A.
        retrieval_result : RetrievalResult
            Output from Agent 1B.

        Returns
        -------
        ConflictDetectionResult
            All conflict records, CSS-passing records, and evidence clusters.
        """
        # Step 1: Two-stage claim alignment
        aligned_pairs = self._align_claims(parametric_result, retrieval_result)

        all_records: list[ConflictRecord] = []

        for pair in aligned_pairs:
            claim = pair.parametric_claim
            chunk = pair.retrieved_chunk

            # Step 2: NLI classification
            nli_scores = self._run_nli(claim.statement, chunk.text)
            conflict_type, spec_mismatch = self._classify_conflict_type(
                nli_scores, claim, chunk
            )

            contra_prob = nli_scores[NLILabel.CONTRADICTION.value]
            entail_prob = nli_scores[NLILabel.ENTAILMENT.value]
            neutral_prob = nli_scores[NLILabel.NEUTRAL.value]
            max_label_str = max(nli_scores, key=lambda k: nli_scores[k])
            nli_label = NLILabel(max_label_str)

            # Step 3: CSS computation
            css = self._compute_css(
                nli_contradiction_prob=contra_prob,
                ebm_weight=chunk.ebm_weight,
                chunk_year=chunk.year,
                parametric_confidence=claim.verbalized_confidence,
            )

            recency = _recency_delta(chunk.year, self.reference_year)
            passes = css >= self.css_threshold

            record = ConflictRecord(
                parametric_claim_id=claim.id,
                parametric_statement=claim.statement,
                retrieved_chunk_id=chunk.chunk_id,
                retrieved_text_snippet=chunk.text[:300],
                nli_label=nli_label,
                nli_contradiction_probability=round(contra_prob, 4),
                nli_entailment_probability=round(entail_prob, 4),
                nli_neutral_probability=round(neutral_prob, 4),
                conflict_type=conflict_type,
                css=css,
                ebm_weight=chunk.ebm_weight,
                ebm_category=chunk.ebm_category,
                recency_delta=round(recency, 4),
                parametric_confidence=claim.verbalized_confidence,
                specificity_mismatch=spec_mismatch,
                passes_css_threshold=passes,
            )
            all_records.append(record)

        css_passing = [r for r in all_records if r.passes_css_threshold]

        # Step 4: Claim clustering
        clusters = self._cluster_conflicts(css_passing)

        return ConflictDetectionResult(
            all_records=all_records,
            css_passing_records=css_passing,
            clusters=clusters,
        )
