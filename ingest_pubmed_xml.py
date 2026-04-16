import argparse
import gzip
import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Optional, Iterable

from langchain_core.documents import Document
from tqdm import tqdm

from article_type import infer_article_type

def iter_pubmed_articles_stream(file_obj) -> Iterable[Document]:
    """
    Stream PubMedArticle elements one by one without loading the full XML tree.
    This is the memory-safe way to parse large PubMed dumps.
    """
    context = ET.iterparse(file_obj, events=("end",))
    for event, elem in context:
        if elem.tag == "PubmedArticle":
            doc = parse_document(elem)
            if doc is not None:
                yield doc

            # critical: free memory for this article
            elem.clear()

def iter_pubmed_articles_from_xml_file(xml_file: str) -> Iterable[Document]:
    with open(xml_file, "rb") as f:
        yield from iter_pubmed_articles_stream(f)


def iter_pubmed_articles_from_gz_file(gz_file: str) -> Iterable[Document]:
    with gzip.open(gz_file, "rb") as f:
        yield from iter_pubmed_articles_stream(f)


def get_text(elem, path: str, default: str = "") -> str:
    found = elem.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def get_all_texts(elem, path: str) -> List[str]:
    values = []
    for item in elem.findall(path):
        if item is not None and item.text:
            values.append(item.text.strip())
    return values


def parse_year(article) -> Optional[int]:
    candidates = [
        ".//JournalIssue/PubDate/Year",
        ".//PubDate/Year",
        ".//DateCompleted/Year",
    ]
    for path in candidates:
        val = get_text(article, path)
        if val.isdigit():
            return int(val)
    return None


def parse_doi(article) -> Optional[str]:
    for aid in article.findall(".//ArticleId"):
        if aid.attrib.get("IdType") == "doi" and aid.text:
            return aid.text.strip()
    return None


def parse_authors(article) -> List[str]:
    authors = []
    for author in article.findall(".//Author"):
        last = get_text(author, "LastName")
        fore = get_text(author, "ForeName")
        collective = get_text(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        full = f"{fore} {last}".strip()
        if full:
            authors.append(full)
    return authors


def parse_abstract_sections(article) -> List[dict]:
    """
    Parse structured abstract into labelled sections.

    Returns a list of dicts with keys 'label' and 'text'.
    Unstructured abstracts are returned as a single section with label None.
    Structured abstracts typically carry labels like BACKGROUND, METHODS,
    RESULTS, CONCLUSIONS — preserving these is critical for section-aware
    retrieval in biomedical RAG.
    """
    sections = []
    for node in article.findall(".//Abstract/AbstractText"):
        label = node.attrib.get("Label") or None
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        sections.append({"label": label, "text": text})
    return sections


def parse_abstract(article) -> str:
    """
    Flat string representation of the abstract (used in page_content
    so the document is self-contained for embedding).
    """
    parts = []
    for node in article.findall(".//Abstract/AbstractText"):
        label = node.attrib.get("Label")
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return "\n".join(parts).strip()


def clean_metadata(metadata: dict) -> dict:
    """
    Chroma requires all metadata values to be str, int, float, or bool.
    Lists are joined into semicolon-separated strings.
    Nones and empty values are dropped.
    abstract_sections (list of dicts) is intentionally excluded here —
    it is carried separately and consumed during chunking.
    """
    cleaned = {}
    for k, v in metadata.items():
        if k == "abstract_sections":
            # Handled separately; not a valid Chroma metadata type
            continue
        if v is None:
            continue
        if isinstance(v, list):
            if not v:
                continue
            joined = "; ".join(str(x) for x in v if x is not None and str(x).strip())
            if joined:
                cleaned[k] = joined
        else:
            cleaned[k] = v
    return cleaned


def parse_document(article) -> Optional[Document]:
    pmid = get_text(article, ".//PMID")
    title = "".join(article.findtext(".//ArticleTitle", default="")).strip()
    abstract_sections = parse_abstract_sections(article)
    abstract = parse_abstract(article)
    publication_types = get_all_texts(article, ".//PublicationType")
    mesh_terms = get_all_texts(article, ".//MeshHeading/DescriptorName")
    journal = get_text(article, ".//Journal/Title")
    year = parse_year(article)
    doi = parse_doi(article)
    authors = parse_authors(article)

    if not title and not abstract:
        return None

    article_type = infer_article_type(title=title, publication_types=publication_types)

    # page_content is the full flat abstract — used as fallback and for
    # document-level retrieval. Section-level chunks are built in build_vector_store.
    page_content = f"Title: {title}\n\n{abstract}".strip() if abstract else f"Title: {title}"

    flat_metadata = clean_metadata({
        "pmid": pmid or "",
        "title": title or "",
        "article_type": article_type,
        "journal": journal or "",
        "year": year if year is not None else 0,
        "doi": doi or "",
        "authors": authors,
        "publication_types": publication_types,
        "mesh_terms": mesh_terms,
        "source": "pubmed_xml",
    })

    # abstract_sections is stored outside flat_metadata so build_vector_store
    # can iterate sections without re-parsing the XML.
    doc = Document(page_content=page_content, metadata=flat_metadata)
    doc.metadata["abstract_sections"] = abstract_sections  # list[dict], stripped before Chroma insert
    return doc


def load_pubmed_path(path: str) -> Iterable[Document]:
    if os.path.isdir(path):
        xml_files = sorted(glob.glob(os.path.join(path, "*.xml")))
        gz_files = sorted(glob.glob(os.path.join(path, "*.xml.gz")))
        all_files = xml_files + gz_files

        if not all_files:
            raise RuntimeError(f"No .xml or .xml.gz files found in folder: {path}")

        for file_path in tqdm(all_files, desc="Parsing PubMed files"):
            if file_path.endswith(".xml.gz"):
                yield from iter_pubmed_articles_from_gz_file(file_path)
            elif file_path.endswith(".xml"):
                yield from iter_pubmed_articles_from_xml_file(file_path)
        return

    if path.endswith(".xml.gz"):
        yield from iter_pubmed_articles_from_gz_file(path)
        return

    if path.endswith(".xml"):
        yield from iter_pubmed_articles_from_xml_file(path)
        return

    raise ValueError(f"Unsupported input path: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml_path",
        type=str,
        default="data/raw/pubmed",
        help="Path to a .xml file, .xml.gz file, or folder containing PubMed files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"Input path not found: {args.xml_path}")

    count = 0
    first_doc = None

    for doc in load_pubmed_path(args.xml_path):
        if first_doc is None:
            first_doc = doc
        count += 1

    print(f"Parsed {count} documents")

    if first_doc:
        print("\nSample document:")
        print("Title:", first_doc.metadata.get("title"))
        print("Type:", first_doc.metadata.get("article_type"))
        print("PMID:", first_doc.metadata.get("pmid"))
        print("Sections:", [s["label"] for s in first_doc.metadata.get("abstract_sections", [])])
        print("Text preview:", first_doc.page_content[:300])

if __name__ == "__main__":
    main()