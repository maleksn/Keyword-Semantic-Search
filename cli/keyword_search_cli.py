import argparse
import math
import sys
from inverted_index import (
    InvertedIndex,
    tokenize_text,
    tokenize_single_term,
    build_command,
    bm25_idf_command,
    bm25_tf_command,
    BM25_K1,
    BM25_B,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # build
    subparsers.add_parser("build", help="Build the inverted index")

    # tf (term frequency)
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Single term")

    # idf (inverse document frequency)
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a term")
    idf_parser.add_argument("term", type=str, help="Single term")

    # tfidf (TF-IDF score)
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Single term")

    # bm25idf (BM25 IDF score)
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # bm25tf (BM25 TF score)
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    # bm25search (full BM25 search)
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            query_tokens = tokenize_text(args.query)
            result_ids = set()
            for token in query_tokens:
                doc_ids = idx.get_documents(token)
                for doc_id in doc_ids:
                    if len(result_ids) >= 5:
                        break
                    result_ids.add(doc_id)
                if len(result_ids) >= 5:
                    break

            if not result_ids:
                print("No results found.")
            else:
                for doc_id in list(result_ids)[:5]:
                    movie = idx.docmap.get(doc_id)
                    if movie:
                        print(f"ID: {doc_id}  Title: {movie['title']}")

        case "build":
            build_command()

        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            try:
                doc_id = int(args.doc_id)
            except ValueError:
                doc_id = args.doc_id

            try:
                token = tokenize_single_term(args.term)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

            tf = idx.get_tf(doc_id, token)
            print(tf)

        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            try:
                token = tokenize_single_term(args.term)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

            total_docs = len(idx.docmap)
            matching_docs = len(idx.get_documents(token))

            idf = math.log((total_docs + 1) / (matching_docs + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            try:
                doc_id = int(args.doc_id)
            except ValueError:
                doc_id = args.doc_id

            try:
                token = tokenize_single_term(args.term)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

            tf = idx.get_tf(doc_id, token)

            total_docs = len(idx.docmap)
            matching_docs = len(idx.get_documents(token))
            idf = math.log((total_docs + 1) / (matching_docs + 1))

            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            results = idx.bm25_search(args.query, limit=args.limit)
            if not results:
                print("No results found.")
            else:
                for rank, (doc_id, score) in enumerate(results, start=1):
                    movie = idx.docmap.get(doc_id)
                    title = movie["title"] if movie else "Unknown"
                    print(f"{rank}. ({doc_id}) {title} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()