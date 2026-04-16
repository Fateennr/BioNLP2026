"""
Agent 3 — Resolution Engine

Receives only high-CSS, high-authority conflicts from Agent 2 and executes
the appropriate resolution action based on conflict type. Because EBM weighting
has already pre-qualified each conflict, Agent 3 does NOT re-evaluate evidence
quality — it only resolves.

Resolution protocols:

  Type I — Override Protocol:
      The retrieved claim replaces the parametric claim. Justified because the
      CSS already encodes that (a) the retrieved source outranks the parametric
      position on the EBM hierarchy, and (b) the contradiction is confident
      (high NLI probability) and authoritative (high EBM weight).

  Type II — Knowledge Injection Protocol:
      The retrieved claim is injected into the answer context with a salience
      weight. Parametric answer is augmented, NOT overridden — Type II is about
      addition, not correction.

  Type III — Claim Merging Protocol:
      A merged claim is constructed that preserves the specificity of the
      retrieved document within the fluency framing of the parametric answer.
      The parametric claim serves as a template; specific retrieved entities,
      numbers, and qualifiers are slotted in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from agents.agent2_conflict_detection import (
    ConflictDetectionResult,
    ConflictRecord,
    ConflictType,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResolvedClaim:
    """
    A claim that has been through the resolution protocol.
    Passed to Agent 4 as a hard constraint for constrained generation.
    """
    origin_parametric_id: str
    original_parametric_statement: str
    resolution_type: str            # "override" | "inject" | "merge" | "retain"
    conflict_type: str              # from ConflictType enum
    resolved_statement: str         # the final resolved claim text
    source_chunk_id: str
    css: float
    ebm_category: str
    salience_weight: float          # injection weight for Type II (0-1)
    provenance_note: str            # human-readable justification

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetainedClaim:
    """
    A parametric claim that was not challenged by any high-CSS conflict.
    Passed to Agent 4 unchanged.
    """
    claim_id: str
    statement: str
    verbalized_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolutionResult:
    resolved_claims: list[ResolvedClaim] = field(default_factory=list)
    retained_claims: list[RetainedClaim] = field(default_factory=list)

    def all_claims_for_synthesis(self) -> list[dict]:
        """
        Return resolved + retained claims as dicts for Agent 4.
        Resolved claims are marked as hard constraints; retained are soft.
        """
        output = []
        for rc in self.resolved_claims:
            d = rc.to_dict()
            d["constraint_type"] = "hard"
            output.append(d)
        for rt in self.retained_claims:
            d = rt.to_dict()
            d["constraint_type"] = "soft"
            output.append(d)
        return output

    def to_dict(self) -> dict:
        return {
            "resolved_count": len(self.resolved_claims),
            "retained_count": len(self.retained_claims),
            "resolved_claims": [r.to_dict() for r in self.resolved_claims],
            "retained_claims": [r.to_dict() for r in self.retained_claims],
        }


# ---------------------------------------------------------------------------
# LLM-assisted merge prompt (Type III)
# ---------------------------------------------------------------------------

_MERGE_SYSTEM = """You are a biomedical claim editor.
You will receive a PARAMETRIC CLAIM (general, fluent) and a RETRIEVED CLAIM
(more specific, with exact figures/qualifiers). Your task is to produce a single
merged claim that:
  1. Preserves the general framing and fluency of the parametric claim.
  2. Slots in the specific figures, drug names, subgroup qualifiers, and
     quantitative details from the retrieved claim.
  3. Does NOT add any information not present in either input.
  4. Is one sentence or two concise sentences maximum.

Output ONLY the merged claim — no explanation, no preamble."""

_MERGE_USER = """PARAMETRIC CLAIM: {parametric}

RETRIEVED CLAIM EXCERPT: {retrieved}

Merged claim:"""

# ---------------------------------------------------------------------------
# Agent 3
# ---------------------------------------------------------------------------

class Agent3:
    """
    Resolution Engine.

    Parameters
    ----------
    llm :
        LangChain-compatible chat model used for Type III claim merging.
        Must support deterministic or low-temperature generation for consistency.
    type_ii_salience_weight : float
        Default salience weight for Type II (injected) claims.
        Higher = injected claims will receive more prominence in Agent 4.
    """

    def __init__(
        self,
        llm,
        type_ii_salience_weight: float = 0.85,
    ):
        self.llm = llm
        self.type_ii_salience_weight = type_ii_salience_weight

    # ------------------------------------------------------------------
    # Type I — Override Protocol
    # ------------------------------------------------------------------

    def _resolve_type_i(self, record: ConflictRecord) -> ResolvedClaim:
        """
        The retrieved claim replaces the parametric claim.

        Justified by CSS: high CSS encodes both high NLI contradiction confidence
        AND high EBM authority (EBM weight already baked into CSS).
        The parametric position is discarded.
        """
        # Extract the most informative sentence from the retrieved chunk
        retrieved_snippet = self._extract_key_sentence(record.retrieved_text_snippet)

        provenance = (
            f"OVERRIDE: Parametric claim contradicted by {record.ebm_category} "
            f"(CSS={record.css:.3f}, EBM_weight={record.ebm_weight:.2f}). "
            f"Retrieved source takes precedence per EBM hierarchy."
        )

        return ResolvedClaim(
            origin_parametric_id=record.parametric_claim_id,
            original_parametric_statement=record.parametric_statement,
            resolution_type="override",
            conflict_type=ConflictType.TYPE_I_CONTRADICTION.value,
            resolved_statement=retrieved_snippet,
            source_chunk_id=record.retrieved_chunk_id,
            css=record.css,
            ebm_category=record.ebm_category,
            salience_weight=1.0,
            provenance_note=provenance,
        )

    # ------------------------------------------------------------------
    # Type II — Knowledge Injection Protocol
    # ------------------------------------------------------------------

    def _resolve_type_ii(self, record: ConflictRecord) -> ResolvedClaim:
        """
        Retrieved claim is injected as new knowledge.
        Parametric answer is AUGMENTED, not overridden.
        Type II is about addition (epistemic absence), not correction.
        """
        retrieved_snippet = self._extract_key_sentence(record.retrieved_text_snippet)

        provenance = (
            f"INJECT: Gap conflict — parametric knowledge absent. "
            f"Retrieved fact injected from {record.ebm_category} "
            f"(CSS={record.css:.3f}, salience={self.type_ii_salience_weight:.2f})."
        )

        return ResolvedClaim(
            origin_parametric_id=record.parametric_claim_id,
            original_parametric_statement=record.parametric_statement,
            resolution_type="inject",
            conflict_type=ConflictType.TYPE_II_GAP.value,
            resolved_statement=retrieved_snippet,
            source_chunk_id=record.retrieved_chunk_id,
            css=record.css,
            ebm_category=record.ebm_category,
            salience_weight=self.type_ii_salience_weight,
            provenance_note=provenance,
        )

    # ------------------------------------------------------------------
    # Type III — Claim Merging Protocol
    # ------------------------------------------------------------------

    def _resolve_type_iii(self, record: ConflictRecord) -> ResolvedClaim:
        """
        Both claims are correct but at different granularities.
        Merge: parametric claim as template, retrieved specifics slotted in.
        Uses LLM to perform the merge with a constrained prompt.
        """
        try:
            messages = [
                SystemMessage(content=_MERGE_SYSTEM),
                HumanMessage(
                    content=_MERGE_USER.format(
                        parametric=record.parametric_statement,
                        retrieved=record.retrieved_text_snippet[:400],
                    )
                ),
            ]
            response = self.llm.invoke(messages)
            merged = response.content.strip()
        except Exception:
            # Fallback: append retrieved specifics verbatim
            merged = (
                f"{record.parametric_statement} "
                f"(Specifically: {self._extract_key_sentence(record.retrieved_text_snippet)})"
            )

        provenance = (
            f"MERGE: Granularity conflict — both sources correct at different "
            f"specificity levels. Parametric template enriched with retrieved "
            f"quantitative details from {record.ebm_category} "
            f"(CSS={record.css:.3f})."
        )

        return ResolvedClaim(
            origin_parametric_id=record.parametric_claim_id,
            original_parametric_statement=record.parametric_statement,
            resolution_type="merge",
            conflict_type=ConflictType.TYPE_III_GRANULARITY.value,
            resolved_statement=merged,
            source_chunk_id=record.retrieved_chunk_id,
            css=record.css,
            ebm_category=record.ebm_category,
            salience_weight=0.9,
            provenance_note=provenance,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_key_sentence(text: str, max_chars: int = 250) -> str:
        """
        Extract the most informative sentence from a text snippet.
        Heuristic: prefer sentences containing numeric data or key biomedical terms.
        Falls back to the first sentence if none qualify.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:max_chars]

        # Prefer sentences with quantitative specificity
        _num_pattern = re.compile(r'\d')
        specific = [s for s in sentences if _num_pattern.search(s)]
        best = specific[0] if specific else sentences[0]
        return best[:max_chars]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        conflict_result: ConflictDetectionResult,
        parametric_claims_all: list,    # list[AtomicClaim] from Agent 1A
    ) -> ResolutionResult:
        """
        Full Agent 3 resolution pipeline.

        Only high-CSS conflicts from Agent 2 are processed.
        Each conflict is routed to the appropriate protocol based on conflict type.
        Parametric claims with no high-CSS conflict are retained unchanged.

        Parameters
        ----------
        conflict_result : ConflictDetectionResult
            Output from Agent 2.
        parametric_claims_all : list[AtomicClaim]
            Complete claim list from Agent 1A (needed to identify un-challenged claims).
        """
        resolved: list[ResolvedClaim] = []
        resolved_claim_ids: set[str] = set()

        # Process each high-CSS conflict record
        for record in conflict_result.css_passing_records:
            if record.conflict_type == ConflictType.TYPE_I_CONTRADICTION:
                rc = self._resolve_type_i(record)
            elif record.conflict_type == ConflictType.TYPE_II_GAP:
                rc = self._resolve_type_ii(record)
            elif record.conflict_type == ConflictType.TYPE_III_GRANULARITY:
                rc = self._resolve_type_iii(record)
            else:
                # ConflictType.NONE passed CSS threshold — retain parametric
                continue

            resolved.append(rc)
            resolved_claim_ids.add(record.parametric_claim_id)

        # Retain parametric claims that were not challenged by any high-CSS conflict
        retained: list[RetainedClaim] = []
        for claim in parametric_claims_all:
            if claim.id not in resolved_claim_ids:
                retained.append(
                    RetainedClaim(
                        claim_id=claim.id,
                        statement=claim.statement,
                        verbalized_confidence=claim.verbalized_confidence,
                    )
                )

        return ResolutionResult(
            resolved_claims=resolved,
            retained_claims=retained,
        )
