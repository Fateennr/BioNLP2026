"""
Agent 4 — Constrained Answer Synthesizer

Takes the resolved claim set from Agent 3 and generates the final answer.

Key constraint: the synthesizer MUST use resolved claims as its grounding
context and is NOT free to paraphrase them away. This is enforced via a
constrained generation prompt that lists each resolved claim as a hard
constraint the answer must be consistent with.

Hard constraints (from resolved Type I/II/III claims) take precedence over
soft constraints (retained parametric claims). The synthesizer is explicitly
instructed not to drop or contradict hard constraints, and not to hallucinate
facts not present in the claim set.

Design rationale: Agent 4 is intentionally "thin" — it performs only
linguistic synthesis, not reasoning or evidence evaluation. All epistemic
work has already been done by Agents 1A, 1B, 2, and 3.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from agents.agent3_resolution import ResolutionResult

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """You are a biomedical answer synthesizer.
Your task is to write a clear, accurate, and clinically appropriate answer to
the user's question using ONLY the claims provided below.

Rules you must follow without exception:
  1. HARD CONSTRAINTS must appear in your answer — do not paraphrase them away,
     do not omit them, and do not contradict them. These claims have been
     verified against retrieved biomedical evidence and represent the most
     authoritative position available.
  2. SOFT CONSTRAINTS are parametric beliefs that were not contradicted by
     evidence. Use them to fill in context, but they may be omitted if space
     is limited.
  3. Do NOT add any factual claims not present in the provided claim set.
  4. Do NOT hallucinate drug names, dosages, statistics, or citations.
  5. Write in clear, professional biomedical prose. Use one paragraph unless
     the answer genuinely requires structure.
  6. If the provided claims are insufficient to answer the question fully,
     say so explicitly — do not fabricate a complete answer."""

_SYNTHESIS_USER = """QUESTION: {query}

HARD CONSTRAINTS (must be included, do not contradict):
{hard_constraints}

SOFT CONSTRAINTS (use for context if relevant):
{soft_constraints}

Write the final answer now:"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    query: str
    final_answer: str
    hard_constraint_count: int
    soft_constraint_count: int
    provenance_summary: list[str]   # list of provenance notes from resolved claims

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent 4
# ---------------------------------------------------------------------------

class Agent4:
    """
    Constrained Answer Synthesizer.

    Parameters
    ----------
    llm :
        LangChain-compatible chat model for final answer generation.
        Should use low temperature (0.0-0.2) for factual consistency.
    max_hard_constraints : int
        Maximum hard constraints to include. If more exist, the highest-CSS
        constraints are selected to fit the context window.
    max_soft_constraints : int
        Maximum soft constraints to include.
    """

    def __init__(
        self,
        llm,
        max_hard_constraints: int = 10,
        max_soft_constraints: int = 5,
    ):
        self.llm = llm
        self.max_hard_constraints = max_hard_constraints
        self.max_soft_constraints = max_soft_constraints

    # ------------------------------------------------------------------
    # Constraint formatting
    # ------------------------------------------------------------------

    def _format_hard_constraints(self, resolution_result: ResolutionResult) -> str:
        """
        Format resolved claims as numbered hard constraints for the prompt.
        Sorted by CSS (descending) so highest-authority claims appear first.
        Includes provenance note to help the model understand why each claim
        has been included (traceability).
        """
        claims = sorted(
            resolution_result.resolved_claims,
            key=lambda r: r.css,
            reverse=True,
        )[: self.max_hard_constraints]

        if not claims:
            return "(none)"

        lines = []
        for i, claim in enumerate(claims, start=1):
            lines.append(
                f"{i}. [{claim.resolution_type.upper()} | {claim.ebm_category} | "
                f"CSS={claim.css:.3f}]\n"
                f"   {claim.resolved_statement}\n"
                f"   → {claim.provenance_note}"
            )
        return "\n\n".join(lines)

    def _format_soft_constraints(self, resolution_result: ResolutionResult) -> str:
        """
        Format retained parametric claims as soft constraints.
        Sorted by verbalized confidence (descending).
        """
        claims = sorted(
            resolution_result.retained_claims,
            key=lambda r: r.verbalized_confidence,
            reverse=True,
        )[: self.max_soft_constraints]

        if not claims:
            return "(none)"

        lines = []
        for i, claim in enumerate(claims, start=1):
            lines.append(
                f"{i}. [conf={claim.verbalized_confidence:.2f}] {claim.statement}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Synthesis Fidelity Check
    # ------------------------------------------------------------------

    def _check_hard_constraint_presence(
        self,
        answer: str,
        resolution_result: ResolutionResult,
    ) -> list[str]:
        """
        Lightweight Synthesis Fidelity Score (SFS) pre-check.

        For each hard constraint, verify that at least one key term from the
        resolved statement appears in the final answer. Returns a list of
        constraint statements that appear to have been dropped.

        NOTE: This is a heuristic pre-check only. Full SFS evaluation (including
        NLI-based entailment checking) is implemented in the evaluation pipeline.
        """
        dropped = []
        answer_lower = answer.lower()

        for claim in resolution_result.resolved_claims:
            # Extract key noun phrases (simple heuristic: words > 4 chars)
            key_terms = [
                w.lower()
                for w in claim.resolved_statement.split()
                if len(w) > 4 and w.isalpha()
            ]
            if key_terms:
                # At least 30% of key terms should appear in the answer
                hit_rate = sum(1 for t in key_terms if t in answer_lower) / len(key_terms)
                if hit_rate < 0.3:
                    dropped.append(claim.resolved_statement)

        return dropped

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        resolution_result: ResolutionResult,
        retry_on_dropped_constraints: bool = True,
    ) -> SynthesisResult:
        """
        Full Agent 4 pipeline:
          1. Format hard and soft constraints.
          2. Generate constrained answer via LLM.
          3. Run heuristic fidelity check.
          4. Optionally retry once if hard constraints were dropped.

        Parameters
        ----------
        query : str
            Original user question.
        resolution_result : ResolutionResult
            Output from Agent 3.
        retry_on_dropped_constraints : bool
            If True and the fidelity check finds dropped hard constraints,
            re-run synthesis with an explicit reminder to include them.
        """
        hard_str = self._format_hard_constraints(resolution_result)
        soft_str = self._format_soft_constraints(resolution_result)

        user_prompt = _SYNTHESIS_USER.format(
            query=query,
            hard_constraints=hard_str,
            soft_constraints=soft_str,
        )

        messages = [
            SystemMessage(content=_SYNTHESIS_SYSTEM),
            HumanMessage(content=user_prompt),
        ]

        answer = self.llm.invoke(messages).content.strip()

        # Fidelity check
        dropped = self._check_hard_constraint_presence(answer, resolution_result)

        if dropped and retry_on_dropped_constraints:
            dropped_list = "\n".join(f"- {d}" for d in dropped)
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"WARNING: Your previous draft omitted these hard constraints. "
                f"You MUST include them:\n{dropped_list}\n\n"
                f"Revised answer:"
            )
            messages_retry = [
                SystemMessage(content=_SYNTHESIS_SYSTEM),
                HumanMessage(content=retry_prompt),
            ]
            answer = self.llm.invoke(messages_retry).content.strip()

        provenance_notes = [
            rc.provenance_note for rc in resolution_result.resolved_claims
        ]

        return SynthesisResult(
            query=query,
            final_answer=answer,
            hard_constraint_count=len(resolution_result.resolved_claims),
            soft_constraint_count=len(resolution_result.retained_claims),
            provenance_summary=provenance_notes,
        )
