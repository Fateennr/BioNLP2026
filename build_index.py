import argparse
import os

# from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.ingest_pubmed_xml import load_pubmed_path

def build_chunks_from_pubmed(docs):
    chunks = []

    for doc in docs:
        title = doc.metadata.get("title", "")
        pmid = doc.metadata.get("pmid", "")
        sections = doc.metadata.get("abstract_sections", [])

        if sections:
            for i, sec in enumerate(sections):
                label = (sec.get("label") or "UNLABELED").upper()
                text = sec.get("text", "").strip()

                if not text:
                    continue

                chunk_text = f"TITLE: {title}\n\nSECTION: {label}\n\n{text}"

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "section": label,
                            "chunk_id": f"{pmid}_{label}_{i}",
                        },
                    )
                )

        else:
            # fallback if no sections
            chunk_text = f"TITLE: {title}\n\n{doc.page_content}"

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "section": "ABSTRACT",
                        "chunk_id": f"{pmid}_abstract_0",
                    },
                )
            )

    return chunks


def main():
    # load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="data/raw/articles.xml")
    parser.add_argument("--persist_dir", type=str, default="data/chroma")
    args = parser.parse_args()

    # 1. Load raw PubMed documents
    docs = load_pubmed_path(args.xml_path)

    if not docs:
        raise RuntimeError("No documents parsed from XML.")

    print(f"Loaded {len(docs)} documents")

    # 2. Embedding model (same model must be used later)
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings"
    )

    # 3. Directly store documents (NO chunking)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=args.persist_dir,
    )

    print(f"Chroma index stored at: {args.persist_dir}")
    print(f"Indexed docs: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()