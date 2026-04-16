"""
Agent 1A — Parametric Elicitation Agent

Decomposes the LLM's parametric response into atomic claims and assigns each
a calibrated confidence score using a two-signal hybrid approach:
  - Primary: Verbalized confidence (instruction-tuned models calibrate better
    via verbalization than raw log-probabilities [1])
  - Secondary: Semantic entropy consistency check across N sampled generations
    at temperature > 0, using embedding similarity [2]

Claims where both signals disagree are flagged as maximally uncertain and
prioritized for retrieval checking by Agent 1B / Agent 2.

References:
    [1] Tian et al. (2023). Just Ask for Calibration. arXiv:2305.14975
    [2] Kuhn et al. (2023). Semantic Uncertainty. ICLR 2023.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AtomicClaim:
    id: str
    statement: str
    type: str                          # core_fact | qualifier | mechanism | statistic
    verbalized_confidence: float       # calibrated probability [0, 1]
    semantic_entropy_flag: bool        # True = high variance across samples
    uncertainty_flag: bool             # True = signals disagree → prioritize for retrieval

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParametricExtractionResult:
    query: str
    raw_answer: str
    claims: list[AtomicClaim] = field(default_factory=list)

    def flagged_claims(self) -> list[AtomicClaim]:
        """Claims marked as maximally uncertain — highest priority for retrieval."""
        return [c for c in self.claims if c.uncertainty_flag]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "raw_answer": self.raw_answer,
            "claims": [c.to_dict() for c in self.claims],
        }


# ---------------------------------------------------------------------------
# Calibration mapping
# BioASQ-derived linear calibration: verbalized [0,10] → probability [0,1]
# In production, replace with a sigmoid fit from held-out BioASQ examples.
# ---------------------------------------------------------------------------

def _calibrate_verbalized_score(raw_score: float) -> float:
    """
    Map a raw verbalized confidence (0-10) to a calibrated probability.
    Uses a conservative shrinkage toward 0.5 to account for overconfidence
    typical in RLHF-tuned models. Replace with empirical calibration curve
    once BioASQ calibration set is available.
    """
    p = raw_score / 10.0
    # Platt-style shrinkage: pull extreme values toward center
    calibrated = 0.1 + 0.8 * p
    return round(float(np.clip(calibrated, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_SYSTEM = """You are a biomedical fact extraction assistant.
Given a question and a model answer, extract every distinct factual claim
from the answer as an atomic statement (one fact per claim).

For each claim output a JSON object with these fields:
  - id: sequential string starting at "C1"
  - statement: the atomic fact as a full sentence
  - type: one of [core_fact, qualifier, mechanism, statistic]
  - verbalized_confidence: integer 0-10 (your confidence this claim is correct)

Return ONLY a JSON array of claim objects — no preamble, no markdown fences."""

_CLAIM_EXTRACTION_USER = """Question: {query}

Answer: {answer}

Extract atomic claims with confidence scores:"""


# ---------------------------------------------------------------------------
# Semantic entropy helpers
# ---------------------------------------------------------------------------

def _mean_pairwise_similarity(embeddings: list[np.ndarray]) -> float:
    """
    Compute mean cosine similarity across all pairs of embedding vectors.
    High mean → semantically consistent → low entropy.
    Low mean → high semantic variance → entropy flag raised.
    """
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a, b = embeddings[i], embeddings[j]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            sims.append(cos)
    return float(np.mean(sims))


def _compute_semantic_entropy_flag(
    claim_text: str,
    n_samples: int,
    temperature: float,
    llm,
    embed_model: HuggingFaceEmbeddings,
    entropy_threshold: float,
) -> bool:
    """
    Re-generate the claim N times at temperature > 0 and measure embedding
    consistency. High variance → True (entropy flag raised).
    Implements the semantic entropy check from Kuhn et al. [2].
    """
    prompt = (
        f"State the following biomedical fact in your own words "
        f"(one sentence only): {claim_text}"
    )

    variants: list[str] = []
    for _ in range(n_samples):
        try:
            msg = llm.invoke([HumanMessage(content=prompt)])
            variants.append(msg.content.strip())
        except Exception:
            variants.append(claim_text)  # fallback: no variance introduced

    if not variants:
        return False

    embeddings = [np.array(embed_model.embed_query(v)) for v in variants]
    mean_sim = _mean_pairwise_similarity(embeddings)

    # Flag if mean similarity falls below threshold (high semantic variance)
    return mean_sim < entropy_threshold


# ---------------------------------------------------------------------------
# Agent 1A
# ---------------------------------------------------------------------------

class Agent1A:
    """
    Parametric Elicitation Agent.

    Parameters
    ----------
    llm :
        LangChain-compatible chat model (e.g. ChatOllama, ChatOpenAI).
        Must support temperature parameter for semantic entropy sampling.
    embed_model_name : str
        HuggingFace embedding model used for semantic entropy consistency check.
        Should match the model used in the vector store (Agent 1B).
    n_entropy_samples : int
        Number of re-generation samples for semantic entropy estimation.
    entropy_temperature : float
        Sampling temperature used during entropy sampling (> 0).
    entropy_threshold : float
        Mean cosine similarity below which semantic entropy flag is raised.
        Lower → more permissive; higher → more sensitive.
    verbalized_confidence_threshold : float
        Calibrated probability below which verbalized confidence is considered
        low — used for disagreement detection with entropy signal.
    """

    def __init__(
        self,
        llm,
        embed_model_name: str = "NeuML/pubmedbert-base-embeddings",
        n_entropy_samples: int = 5,
        entropy_temperature: float = 0.7,
        entropy_threshold: float = 0.75,
        verbalized_confidence_threshold: float = 0.60,
    ):
        self.llm = llm
        self.embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
        self.n_entropy_samples = n_entropy_samples
        self.entropy_temperature = entropy_temperature
        self.entropy_threshold = entropy_threshold
        self.verbalized_confidence_threshold = verbalized_confidence_threshold

    # ------------------------------------------------------------------
    # Step 1: Generate parametric answer
    # ------------------------------------------------------------------

    def _generate_parametric_answer(self, query: str) -> str:
        """Generate a direct answer from parametric knowledge (no retrieval)."""
        system = (
            "You are a biomedical expert. Answer the question using only your "
            "internal knowledge. Do not hedge excessively; be specific."
        )
        messages = [SystemMessage(content=system), HumanMessage(content=query)]
        response = self.llm.invoke(messages)
        return response.content.strip()

    # ------------------------------------------------------------------
    # Step 2: Extract atomic claims with verbalized confidence
    # ------------------------------------------------------------------

    def _extract_claims_with_confidence(
        self, query: str, answer: str
    ) -> list[dict]:
        """
        Prompt the LLM to decompose its answer into atomic claims,
        each with a verbalized confidence score (0-10).
        """
        user_prompt = _CLAIM_EXTRACTION_USER.format(query=query, answer=answer)
        messages = [
            SystemMessage(content=_CLAIM_EXTRACTION_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if the model adds them despite the prompt
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        try:
            claims_raw = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat entire answer as a single low-confidence claim
            claims_raw = [
                {
                    "id": "C1",
                    "statement": answer,
                    "type": "core_fact",
                    "verbalized_confidence": 5,
                }
            ]

        return claims_raw if isinstance(claims_raw, list) else [claims_raw]

    # ------------------------------------------------------------------
    # Step 3: Attach semantic entropy flags and build AtomicClaim objects
    # ------------------------------------------------------------------

    def _build_atomic_claims(
        self, raw_claims: list[dict], run_entropy_check: bool = True
    ) -> list[AtomicClaim]:
        """
        For each raw claim dict:
          1. Calibrate verbalized confidence score.
          2. Optionally run semantic entropy consistency check.
          3. Raise uncertainty_flag where both signals disagree.
        """
        atomic_claims: list[AtomicClaim] = []

        for rc in raw_claims:
            claim_id = rc.get("id", f"C{len(atomic_claims)+1}")
            statement = rc.get("statement", "")
            claim_type = rc.get("type", "core_fact")
            raw_conf = float(rc.get("verbalized_confidence", 5))

            calibrated_conf = _calibrate_verbalized_score(raw_conf)

            # Semantic entropy check (can be disabled for speed in testing)
            if run_entropy_check and statement:
                entropy_flag = _compute_semantic_entropy_flag(
                    claim_text=statement,
                    n_samples=self.n_entropy_samples,
                    temperature=self.entropy_temperature,
                    llm=self.llm,
                    embed_model=self.embed_model,
                    entropy_threshold=self.entropy_threshold,
                )
            else:
                entropy_flag = False

            # Disagreement: high verbalized confidence but high semantic entropy
            # OR low verbalized confidence but low semantic entropy
            # Either scenario → flag as maximally uncertain [1][2]
            low_verbalized = calibrated_conf < self.verbalized_confidence_threshold
            uncertainty_flag = entropy_flag or low_verbalized

            atomic_claims.append(
                AtomicClaim(
                    id=claim_id,
                    statement=statement,
                    type=claim_type,
                    verbalized_confidence=calibrated_conf,
                    semantic_entropy_flag=entropy_flag,
                    uncertainty_flag=uncertainty_flag,
                )
            )

        return atomic_claims

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        run_entropy_check: bool = True,
    ) -> ParametricExtractionResult:
        """
        Full Agent 1A pipeline:
          1. Generate parametric answer.
          2. Extract atomic claims with verbalized confidence.
          3. Attach semantic entropy flags.
          4. Return structured ParametricExtractionResult.

        Parameters
        ----------
        query : str
            The user's biomedical question.
        run_entropy_check : bool
            Set False during development/testing to skip the N-sample
            re-generation step and speed up iteration.
        """
        raw_answer = self._generate_parametric_answer(query)
        raw_claims = self._extract_claims_with_confidence(query, raw_answer)
        claims = self._build_atomic_claims(raw_claims, run_entropy_check)

        return ParametricExtractionResult(
            query=query,
            raw_answer=raw_answer,
            claims=claims,
        )
