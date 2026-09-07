# File: ./cli/multimodal_search_cli.py
import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Search CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # verify_image_embedding command
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify CLIP model loads and generates image embeddings",
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to embed",
    )

    # image_search command
    search_parser = subparsers.add_parser(
        "image_search",
        help="Search movies using an image instead of text",
    )
    search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to search with",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)

            if not results:
                print("No results found.")
            else:
                for rank, res in enumerate(results, start=1):
                    desc = res["description"]
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    print(
                        f"{rank}. {res['title']} "
                        f"(similarity: {res['similarity']:.3f})"
                    )
                    print(f"   {desc}")
                    print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()