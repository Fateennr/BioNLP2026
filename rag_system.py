from typing import List, Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings

from app.ebm import choose_best_doc_by_ebm


class ConflictDecision(BaseModel):
    relation: Literal["agree", "partial_conflict", "conflict"] = Field(
        description="Relationship between parametric answer and retrieved answer."
    )
    reason: str = Field(description="Short reason for the decision.")


class MedicalRAG:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        llm_model: str = "gpt-4o-mini",
        retrieval_k: int = 5,
    ):
        load_dotenv()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
        self.llm = ChatOllama(model="mistral", temperature=0)
        self.structured_llm = self.llm.with_structured_output(ConflictDecision)

    def get_parametric_answer(self, question: str) -> str:
        prompt = f"""
You are a medical assistant.
Answer the following question from your own internal knowledge only.
Be concise and factual.

Question: {question}
"""
        return self.llm.invoke(prompt).content.strip()

    def retrieve_docs(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def get_retrieved_answer(self, question: str, docs: List[Document]) -> str:
        context = "\n\n".join(
            [
                (
                    f"TITLE: {d.metadata.get('title')}\n"
                    f"ARTICLE_TYPE: {d.metadata.get('article_type')}\n"
                    f"YEAR: {d.metadata.get('year')}\n"
                    f"PMID: {d.metadata.get('pmid')}\n"
                    f"TEXT: {d.page_content}"
                )
                for d in docs
            ]
        )

        prompt = f"""
You are a medical evidence assistant.
Answer the question using only the retrieved evidence below.
If the evidence is insufficient, say so.
Do not invent facts.

Question:
{question}

Retrieved evidence:
{context}
"""
        return self.llm.invoke(prompt).content.strip()

    def detect_conflict(self, question, parametric_answer, retrieved_answer):
        prompt = f"""
    Compare the answers.

    Question: {question}

    Parametric answer:
    {parametric_answer}

    Retrieved answer:
    {retrieved_answer}

    Return ONLY one word:
    - agree
    - partial_conflict
    - conflict
    """

        result = self.llm.invoke(prompt).content.lower()

        if "partial" in result:
            return "partial_conflict"
        elif "conflict" in result:
            return "conflict"
        else:
            return "agree"

    def resolve_conflict(
        self,
        question: str,
        parametric_answer: str,
        retrieved_answer: str,
        docs: List[Document],
        conflict: ConflictDecision,
    ) -> str:
        best_doc = choose_best_doc_by_ebm(docs)

        if best_doc is None:
            return retrieved_answer

        best_type = best_doc.metadata.get("article_type", "unknown")
        best_title = best_doc.metadata.get("title", "Unknown title")
        best_text = best_doc.page_content
        best_year = best_doc.metadata.get("year")
        best_pmid = best_doc.metadata.get("pmid")

        prompt = f"""
You are resolving a conflict between a parametric medical answer and retrieved biomedical evidence.

Question:
{question}

Parametric answer:
{parametric_answer}

Retrieved answer:
{retrieved_answer}

Conflict decision:
{conflict.relation}
Reason:
{conflict.reason}

Highest-ranked retrieved evidence by EBM hierarchy:
Title: {best_title}
Article type: {best_type}
Year: {best_year}
PMID: {best_pmid}
Text: {best_text}

Rules:
1. If there is conflict and the retrieved evidence comes from a stronger EBM category, prefer the retrieved evidence.
2. If evidence is weak or unclear, answer cautiously.
3. Mention that the answer is based on retrieved evidence when relevant.
4. Keep the answer concise.

Return the final answer only.
"""
        return self.llm.invoke(prompt).content.strip()

    def ask(self, question: str) -> dict:
        parametric_answer = self.get_parametric_answer(question)
        docs = self.retrieve_docs(question)
        retrieved_answer = self.get_retrieved_answer(question, docs)

        conflict_relation = self.detect_conflict(
            question, parametric_answer, retrieved_answer
        )

        if conflict_relation == "conflict":
            final_answer = self.resolve_conflict(
                question=question,
                parametric_answer=parametric_answer,
                retrieved_answer=retrieved_answer,
                docs=docs,
                conflict=conflict_relation,
            )
            conflict_reason = "Parametric and retrieved answers conflict."
        elif conflict_relation == "partial_conflict":
            final_answer = retrieved_answer
            conflict_reason = "Parametric and retrieved answers partially conflict."
        else:
            final_answer = retrieved_answer
            conflict_reason = "Parametric and retrieved answers agree."

        sources = [
            {
                "title": d.metadata.get("title"),
                "article_type": d.metadata.get("article_type"),
                "year": d.metadata.get("year"),
                "pmid": d.metadata.get("pmid"),
            }
            for d in docs
        ]

        return {
            "question": question,
            "parametric_answer": parametric_answer,
            "retrieved_answer": retrieved_answer,
            "conflict_relation": conflict_relation,
            "conflict_reason": conflict_reason,
            "final_answer": final_answer,
            "sources": sources,
        }