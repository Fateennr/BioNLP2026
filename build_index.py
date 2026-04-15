import argparse
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.ingest_pubmed_xml import load_pubmed_xml


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="data/raw/articles.xml")
    parser.add_argument("--persist_dir", type=str, default="data/chroma")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    docs = load_pubmed_xml(args.xml_path)
    if not docs:
        raise RuntimeError("No documents parsed from XML.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    split_docs = splitter.split_documents(docs)
    print(f"Built {len(split_docs)} chunks from {len(docs)} documents")



    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=args.persist_dir,
    )

    print(f"Chroma index stored at: {args.persist_dir}")
    print(f"Indexed chunks: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()