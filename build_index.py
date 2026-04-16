import argparse
import math
import os
from collections import Counter

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ingest_pubmed_xml import load_pubmed_path


SECTION_LABEL_MAP = {
    "BACKGROUND": "BACKGROUND",
    "INTRODUCTION": "BACKGROUND",
    "OBJECTIVE": "BACKGROUND",
    "OBJECTIVES": "BACKGROUND",
    "AIM": "BACKGROUND",
    "AIMS": "BACKGROUND",
    "PURPOSE": "BACKGROUND",
    "METHODS": "METHODS",
    "METHOD": "METHODS",
    "MATERIALS AND METHODS": "METHODS",
    "STUDY DESIGN": "METHODS",
    "DESIGN": "METHODS",
    "RESULTS": "RESULTS",
    "FINDINGS": "RESULTS",
    "OUTCOMES": "RESULTS",
    "CONCLUSIONS": "CONCLUSIONS",
    "CONCLUSION": "CONCLUSIONS",
    "SUMMARY": "CONCLUSIONS",
    "INTERPRETATION": "CONCLUSIONS",
    "CLINICAL IMPLICATIONS": "CONCLUSIONS",
}

INDEXABLE_SECTIONS = {"BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"}


def build_chunks_from_pubmed(docs: list[Document]) -> list[Document]:
    """
    Convert parsed PubMed documents into section-level chunks for indexing.

    Strategy:
    - Structured abstracts are split into one chunk per INDEXABLE_SECTIONS
      label. Each chunk prepends the article title and section label so the
      embedding captures full context when chunks are retrieved in isolation.
    - When two raw labels normalise to the same category (e.g. "CONCLUSION"
      and "CONCLUSIONS"), only the first occurrence is indexed. The duplicate
      is dropped entirely — mutating page_content after a chunk is created
      would cause the stored text and its embedding to diverge.
    - Unstructured abstracts are indexed as a single ABSTRACT chunk.
    - abstract_sections (list of dicts) is stripped from metadata before
      Chroma insertion because Chroma requires scalar metadata values.
      It is read here from doc.metadata where ingest_pubmed_xml.py attached
      it after clean_metadata() ran (see parse_document() there).
    """
    chunks = []

    for doc in docs:
        title = doc.metadata.get("title", "")
        pmid = doc.metadata.get("pmid", "")
        sections = doc.metadata.get("abstract_sections", [])

        # Base metadata without the non-scalar abstract_sections field
        base_metadata = {
            k: v for k, v in doc.metadata.items()
            if k != "abstract_sections"
        }

        if sections:
            seen_normalised = set()
            for sec in sections:
                raw_label = (sec.get("label") or "").strip().upper()
                normalised = SECTION_LABEL_MAP.get(raw_label, raw_label or "ABSTRACT")

                if normalised not in INDEXABLE_SECTIONS:
                    normalised = "ABSTRACT"

                text = sec.get("text", "").strip()
                if not text:
                    continue

                # If two raw labels normalise to the same category, keep only
                # the first. Merging by mutating page_content is not safe here
                # because the chunk has already been appended with its text
                # embedded in the formatted string — mutating after the fact
                # would not update the embedding, causing stored text and
                # vector to diverge.
                if normalised in seen_normalised:
                    continue
                seen_normalised.add(normalised)

                chunk_text = (
                    f"TITLE: {title}\n\n"
                    f"SECTION: {normalised}\n\n"
                    f"{text}"
                )

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **base_metadata,
                            "section": normalised,
                            # Use normalised label (not raw loop index) so
                            # chunk_id is stable across runs regardless of
                            # which raw labels were skipped during dedup.
                            "chunk_id": f"{pmid}_{normalised}",
                        },
                    )
                )

        else:
            chunk_text = (
                f"TITLE: {title}\n\n"
                f"SECTION: ABSTRACT\n\n"
                f"{doc.page_content}"
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **base_metadata,
                        "section": "ABSTRACT",
                        "chunk_id": f"{pmid}_ABSTRACT",
                    },
                )
            )

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="data/raw/articles")
    parser.add_argument("--persist_dir", type=str, default="data/chroma")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Chunks per Chroma insert batch (default: 500)")
    parser.add_argument("--file_batch", type=int, default=20,
                        help="XML files to load per iteration (default: 20)")
    args = parser.parse_args()

    # Embedding model must match the model used at query time exactly.
    # A mismatch produces a valid-looking but semantically meaningless index.
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings"
    )

    vectorstore = Chroma(
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
    )

    xml_files = sorted(os.listdir(args.xml_path))
    total_chunks_indexed = 0

    for i in range(0, len(xml_files), args.file_batch):
        file_batch = xml_files[i : i + args.file_batch]
        batch_chunks = []

        for file in file_batch:
            path = os.path.join(args.xml_path, file)
            docs = load_pubmed_path(path)
            batch_chunks.extend(build_chunks_from_pubmed(docs))

        if not batch_chunks:
            continue

        section_counts = Counter(c.metadata["section"] for c in batch_chunks)
        print(f"\nFile batch {i // args.file_batch + 1} "
              f"({min(i + args.file_batch, len(xml_files))}/{len(xml_files)} files) "
              f"— {len(batch_chunks)} chunks")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count}")

        total_batches = math.ceil(len(batch_chunks) / args.batch_size)
        for j in range(0, len(batch_chunks), args.batch_size):
            chunk_batch = batch_chunks[j : j + args.batch_size]
            vectorstore.add_documents(chunk_batch)
            print(f"  Indexed chunk batch {j // args.batch_size + 1}/{total_batches} "
                  f"({min(j + args.batch_size, len(batch_chunks))}/{len(batch_chunks)})")

        total_chunks_indexed += len(batch_chunks)
        del batch_chunks

    print(f"\nChroma index stored at: {args.persist_dir}")
    print(f"Total chunks indexed this run: {total_chunks_indexed}")
    print(f"Total chunks in collection: {vectorstore._collection.count()}")