# File: ./cli/augmented_generation_cli.py
import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import HybridSearch
from lib.search_utils import load_movies


def get_llm_client() -> OpenAI:
    """Create an OpenAI client pointed at OpenRouter."""
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def format_documents_for_prompt(results: list[dict]) -> str:
    """Format search results into a numbered string block for the LLM prompt."""
    lines = []
    for i, res in enumerate(results, start=1):
        title = res.get("title", "Unknown")
        document = res.get("document", "")
        lines.append(f"[{i}] Title: {title}\n    Content: {document}")
    return "\n\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # rag command
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument(
        "query", type=str, help="Search query for RAG"
    )

    # summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize search results using multi-document synthesis",
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize (default: 5)",
    )

    # citations command
    citations_parser = subparsers.add_parser(
        "citations",
        help="Answer query with source citations",
    )
    citations_parser.add_argument(
        "query", type=str, help="Search query for citation-aware answer"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use as sources (default: 5)",
    )

    # question command
    question_parser = subparsers.add_parser(
        "question",
        help="Conversational question-answering based on search results",
    )
    question_parser.add_argument(
        "question", type=str, help="User question to answer"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use as context (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies()
            searcher = HybridSearch(documents)
            results = searcher.rrf_search(query=query, k=60, limit=5)

            print("Search Results:")
            if not results:
                print("- No results found")
            else:
                for res in results:
                    print(f"- {res.get('title', 'Unknown')}")

            docs_text = format_documents_for_prompt(results)

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs_text}

Answer:"""

            client = get_llm_client()

            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error generating response: {type(e).__name__}: {e}"

            print("\nRAG Response:")
            print(answer)

        case "summarize":
            query = args.query
            limit = args.limit

            documents = load_movies()
            searcher = HybridSearch(documents)
            results = searcher.rrf_search(query=query, k=60, limit=limit)

            print("Search Results:")
            if not results:
                print("  - No results found")
            else:
                for res in results:
                    print(f"  - {res.get('title', 'Unknown')}")

            results_text = format_documents_for_prompt(results)

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results_text}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            client = get_llm_client()

            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                summary = f"Error generating summary: {type(e).__name__}: {e}"

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            query = args.query
            limit = args.limit

            documents = load_movies()
            searcher = HybridSearch(documents)
            results = searcher.rrf_search(query=query, k=60, limit=limit)

            print("Search Results:")
            if not results:
                print("  - No results found")
            else:
                for res in results:
                    print(f"  - {res.get('title', 'Unknown')}")

            docs_text = format_documents_for_prompt(results)

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs_text}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            client = get_llm_client()

            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error generating answer: {type(e).__name__}: {e}"

            print("\nLLM Answer:")
            print(answer)

        case "question":
            question = args.question
            limit = args.limit

            documents = load_movies()
            searcher = HybridSearch(documents)
            results = searcher.rrf_search(query=question, k=60, limit=limit)

            print("Search Results:")
            if not results:
                print("  - No results found")
            else:
                for res in results:
                    print(f"  - {res.get('title', 'Unknown')}")

            context = format_documents_for_prompt(results)

            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            client = get_llm_client()

            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error generating answer: {type(e).__name__}: {e}"

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()