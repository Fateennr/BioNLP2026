import argparse
import os
import xml.etree.ElementTree as ET
from typing import List, Optional

from dotenv import load_dotenv
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

    page_content = abstract if abstract else title

    metadata = {
        "pmid": pmid or "",
        "title": title or "",
        "article_type": article_type,
        "journal": journal or "",
        "year": year if year is not None else 0,
        "doi": doi or "",
        "authors": "; ".join(authors) if authors else "",
        "publication_types": "; ".join(publication_types) if publication_types else "",
        "mesh_terms": "; ".join(mesh_terms) if mesh_terms else "",
        "source": "pubmed_xml",
    }

    return Document(page_content=page_content, metadata=metadata)


def load_pubmed_xml(xml_path: str) -> List[Document]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    docs: List[Document] = []

    articles = root.findall(".//PubmedArticle")
    for article in tqdm(articles, desc="Parsing XML"):
        doc = parse_document(article)
        if doc is not None:
            docs.append(doc)

    return docs


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="data/raw/articles.xml")
    args = parser.parse_args()

    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"XML file not found: {args.xml_path}")

    docs = load_pubmed_xml(args.xml_path)
    print(f"Parsed {len(docs)} documents")

    if docs:
        print("\nSample document:")
        print("Title:", docs[0].metadata.get("title"))
        print("Type:", docs[0].metadata.get("article_type"))
        print("PMID:", docs[0].metadata.get("pmid"))
        print("Text preview:", docs[0].page_content[:300])


if __name__ == "__main__":
    main()