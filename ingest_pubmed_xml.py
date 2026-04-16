import argparse
import gzip
import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Optional, Iterable

from langchain_core.documents import Document
from tqdm import tqdm

from app.article_type import infer_article_type


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


def parse_abstract(article) -> str:
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
    cleaned = {}
    for k, v in metadata.items():
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

    # Keep title in content too, not only metadata
    page_content = f"Title: {title}\n\n{abstract}".strip() if abstract else f"Title: {title}"

    metadata = clean_metadata({
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

    return Document(page_content=page_content, metadata=metadata)


def iter_pubmed_articles_from_xml_file(xml_file: str) -> Iterable[Document]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for article in root.findall(".//PubmedArticle"):
        doc = parse_document(article)
        if doc is not None:
            yield doc


def iter_pubmed_articles_from_gz_file(gz_file: str) -> Iterable[Document]:
    with gzip.open(gz_file, "rb") as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for article in root.findall(".//PubmedArticle"):
            doc = parse_document(article)
            if doc is not None:
                yield doc


def load_pubmed_path(path: str) -> List[Document]:
    docs: List[Document] = []

    if os.path.isdir(path):
        xml_files = sorted(glob.glob(os.path.join(path, "*.xml")))
        gz_files = sorted(glob.glob(os.path.join(path, "*.xml.gz")))
        all_files = xml_files + gz_files

        if not all_files:
            raise RuntimeError(f"No .xml or .xml.gz files found in folder: {path}")

        for file_path in tqdm(all_files, desc="Parsing PubMed files"):
            if file_path.endswith(".xml.gz"):
                docs.extend(iter_pubmed_articles_from_gz_file(file_path))
            elif file_path.endswith(".xml"):
                docs.extend(iter_pubmed_articles_from_xml_file(file_path))

        return docs

    if path.endswith(".xml.gz"):
        return list(iter_pubmed_articles_from_gz_file(path))

    if path.endswith(".xml"):
        return list(iter_pubmed_articles_from_xml_file(path))

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

    docs = load_pubmed_path(args.xml_path)
    print(f"Parsed {len(docs)} documents")

    if docs:
        print("\nSample document:")
        print("Title:", docs[0].metadata.get("title"))
        print("Type:", docs[0].metadata.get("article_type"))
        print("PMID:", docs[0].metadata.get("pmid"))
        print("Text preview:", docs[0].page_content[:300])


if __name__ == "__main__":
    main()