import argparse
import json

from app.rag_system import MedicalRAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", type=str, default="data/chroma")
    parser.add_argument("--question", type=str, default=None)
    args = parser.parse_args()

    rag = MedicalRAG(persist_directory=args.persist_dir)

    if args.question:
        result = rag.ask(args.question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Medical RAG ready. Type 'exit' to quit.\n")

    while True:
        q = input("Ask: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        result = rag.ask(q)

        print("\n=== FINAL ANSWER ===")
        print(result["final_answer"])

        print("\n=== CONFLICT ===")
        print(result["conflict_relation"])
        print(result["conflict_reason"])

        print("\n=== SOURCES ===")
        for i, src in enumerate(result["sources"], start=1):
            print(
                f"{i}. {src.get('title')} | "
                f"type={src.get('article_type')} | "
                f"year={src.get('year')} | "
                f"pmid={src.get('pmid')}"
            )
        print()


if __name__ == "__main__":
    main()