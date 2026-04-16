"""
RAG Pipeline Orchestrator — Retrieval-Parametric Conflict Resolution

Ties all four agents together into a single end-to-end pipeline:

  User Query
       ↓
  Agent 1A: Parametric Claim Extraction
       ↓
  Agent 1B: Hybrid Retrieval (BM25 + Dense, Cross-encoder rerank)
       ↓
  Agent 2:  Conflict Detection (Bi-encoder → Cross-encoder → NLI → CSS → Clustering)
       ↓
  Agent 3:  Resolution (CSS-gated: Override / Inject / Merge)
       ↓
  Agent 4:  Constrained Answer Synthesis
       ↓
  Final Answer

Usage
-----
  from pipeline import BiomedRAGPipeline

  pipeline = BiomedRAGPipeline(
      chroma_persist_dir="data/chroma",
      llm=your_langchain_chat_model,
  )
  result = pipeline.run("What is the recommended first-line treatment for type 2 diabetes?")
  print(result.final_answer)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

from agents.agent1a_parametric_elicitation import Agent1A, ParametricExtractionResult
from agents.agent1b_retrieval import Agent1B, RetrievalResult
from agents.agent2_conflict_detection import Agent2, ConflictDetectionResult
from agents.agent3_resolution import Agent3, ResolutionResult
from agents.agent4_synthesizer import Agent4, SynthesisResult


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    query: str
    final_answer: str

    # Intermediate outputs for inspection / evaluation
    parametric_result: ParametricExtractionResult
    retrieval_result: RetrievalResult
    conflict_result: ConflictDetectionResult
    resolution_result: ResolutionResult
    synthesis_result: SynthesisResult

    # Timing
    elapsed_seconds: float

    def summary(self) -> dict:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "parametric_claims": len(self.parametric_result.claims),
            "flagged_claims": len(self.parametric_result.flagged_claims()),
            "retrieved_chunks": len(self.retrieval_result.chunks),
            "total_conflict_pairs": len(self.conflict_result.all_records),
            "css_passing_conflicts": len(self.conflict_result.css_passing_records),
            "clusters": len(self.conflict_result.clusters),
            "resolved_claims": len(self.resolution_result.resolved_claims),
            "retained_claims": len(self.resolution_result.retained_claims),
            "hard_constraints_in_answer": self.synthesis_result.hard_constraint_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.summary(), indent=indent)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BiomedRAGPipeline:
    """
    End-to-end Retrieval-Parametric Conflict Resolution pipeline.

    Parameters
    ----------
    llm :
        LangChain-compatible chat model. Used by Agent 1A (claim extraction +
        semantic entropy sampling), Agent 3 (Type III merging), and Agent 4
        (constrained synthesis).
    chroma_persist_dir : str
        Path to the Chroma vector store built by build_vector_store.py.
    embed_model_name : str
        Embedding model. Must match build_vector_store.py.

    Agent-specific kwargs can be passed via agent1a_kwargs, agent1b_kwargs,
    agent2_kwargs, agent3_kwargs, agent4_kwargs dictionaries.
    """

    def __init__(
        self,
        llm,
        chroma_persist_dir: str = "data/chroma",
        embed_model_name: str = "NeuML/pubmedbert-base-embeddings",
        agent1a_kwargs: Optional[dict] = None,
        agent1b_kwargs: Optional[dict] = None,
        agent2_kwargs: Optional[dict] = None,
        agent3_kwargs: Optional[dict] = None,
        agent4_kwargs: Optional[dict] = None,
    ):
        self.agent1a = Agent1A(
            llm=llm,
            embed_model_name=embed_model_name,
            **(agent1a_kwargs or {}),
        )
        self.agent1b = Agent1B(
            chroma_persist_dir=chroma_persist_dir,
            embed_model_name=embed_model_name,
            **(agent1b_kwargs or {}),
        )
        self.agent2 = Agent2(
            embed_model_name=embed_model_name,
            **(agent2_kwargs or {}),
        )
        self.agent3 = Agent3(
            llm=llm,
            **(agent3_kwargs or {}),
        )
        self.agent4 = Agent4(
            llm=llm,
            **(agent4_kwargs or {}),
        )

    def run(
        self,
        query: str,
        run_entropy_check: bool = True,
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Execute the full pipeline for a single query.

        Parameters
        ----------
        query : str
            Biomedical question from the user.
        run_entropy_check : bool
            Set False to skip semantic entropy sampling (faster for dev/testing).
        verbose : bool
            Print intermediate stage summaries to stdout.
        """
        t0 = time.time()

        # ── Agent 1A: Parametric claim extraction ──────────────────────
        if verbose:
            print(f"\n[Agent 1A] Extracting parametric claims...")
        parametric_result = self.agent1a.run(query, run_entropy_check=run_entropy_check)
        if verbose:
            print(f"  → {len(parametric_result.claims)} claims extracted, "
                  f"{len(parametric_result.flagged_claims())} flagged as uncertain")

        # ── Agent 1B: Hybrid retrieval ─────────────────────────────────
        if verbose:
            print(f"[Agent 1B] Retrieving evidence chunks...")
        retrieval_result = self.agent1b.run(query)
        if verbose:
            print(f"  → {len(retrieval_result.chunks)} chunks retrieved")
            ebm_summary = {}
            for c in retrieval_result.chunks:
                ebm_summary[c.ebm_category] = ebm_summary.get(c.ebm_category, 0) + 1
            for cat, count in sorted(ebm_summary.items()):
                print(f"     {cat}: {count}")

        # ── Agent 2: Conflict detection ────────────────────────────────
        if verbose:
            print(f"[Agent 2] Detecting and classifying conflicts...")
        conflict_result = self.agent2.run(parametric_result, retrieval_result)
        if verbose:
            print(f"  → {len(conflict_result.all_records)} pairs evaluated")
            print(f"  → {len(conflict_result.css_passing_records)} passed CSS threshold")
            if conflict_result.css_passing_records:
                from agents.agent2_conflict_detection import ConflictType
                type_counts = {}
                for r in conflict_result.css_passing_records:
                    type_counts[r.conflict_type.value] = (
                        type_counts.get(r.conflict_type.value, 0) + 1
                    )
                for t, n in sorted(type_counts.items()):
                    print(f"     {t}: {n}")

        # ── Agent 3: Resolution ────────────────────────────────────────
        if verbose:
            print(f"[Agent 3] Resolving conflicts...")
        resolution_result = self.agent3.run(conflict_result, parametric_result.claims)
        if verbose:
            print(f"  → {len(resolution_result.resolved_claims)} claims resolved")
            print(f"  → {len(resolution_result.retained_claims)} claims retained")
            for rc in resolution_result.resolved_claims:
                print(f"     [{rc.resolution_type.upper()}] {rc.resolved_statement[:80]}...")

        # ── Agent 4: Constrained synthesis ────────────────────────────
        if verbose:
            print(f"[Agent 4] Synthesizing final answer...")
        synthesis_result = self.agent4.run(query, resolution_result)

        elapsed = time.time() - t0

        if verbose:
            print(f"\n{'='*60}")
            print(f"FINAL ANSWER:\n{synthesis_result.final_answer}")
            print(f"{'='*60}")
            print(f"Pipeline completed in {elapsed:.2f}s")

        return PipelineResult(
            query=query,
            final_answer=synthesis_result.final_answer,
            parametric_result=parametric_result,
            retrieval_result=retrieval_result,
            conflict_result=conflict_result,
            resolution_result=resolution_result,
            synthesis_result=synthesis_result,
            elapsed_seconds=elapsed,
        )
