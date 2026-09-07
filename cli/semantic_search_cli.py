import argparse
import json
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    chunk_text,
    chunk_text_semantic,
    SemanticSearch,
    ChunkedSemanticSearch,
)
from lib.search_utils import load_movies, format_search_result, SCORE_PRECISION


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify
    subparsers.add_parser("verify", help="Load model and display info")

    # embed
    embed_parser = subparsers.add_parser("embed", help="Generate embedding for a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    # verify_embeddings
    subparsers.add_parser("verify_embeddings", help="Build/load movie embeddings and show shape")

    # embed_query
    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query string")

    # search
    search_parser = subparsers.add_parser("search", help="Semantic search for movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results (default: 5)"
    )

    # chunk (word-based)
    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words per chunk (default: 200)"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)"
    )

    # semantic_chunk (sentence-based)
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into chunks based on sentence boundaries"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4,
        help="Maximum number of sentences per chunk (default: 4)"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0,
        help="Number of sentences to overlap between chunks (default: 0)"
    )

    # embed_chunks
    subparsers.add_parser("embed_chunks", help="Build/load chunked movie embeddings")

    # search_chunked
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            searcher = SemanticSearch()
            with open("data/movies.json", "r") as f:
                data = json.load(f)
            documents = data.get("movies", [])
            searcher.load_or_create_embeddings(documents)

            results = searcher.search(args.query, limit=args.limit)

            if not results:
                print("No results found.")
            else:
                for rank, movie in enumerate(results, start=1):
                    desc = movie["description"]
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    print(f"{rank}. {movie['title']} (score: {movie['score']:.4f})")
                    print(f"  {desc}")

        case "chunk":
            chunks = chunk_text(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            chunks = chunk_text_semantic(
                args.text,
                max_chunk_size=args.max_chunk_size,
                overlap=args.overlap
            )
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            documents = load_movies()
            searcher = ChunkedSemanticSearch()
            embeddings = searcher.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            documents = load_movies()
            searcher = ChunkedSemanticSearch()
            searcher.load_or_create_chunk_embeddings(documents)
            results = searcher.search_chunks(args.query, limit=args.limit)

            if not results:
                print("No results found.")
            else:
                for i, res in enumerate(results, start=1):
                    print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                    print(f"   {res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()