# File: ./cli/hybrid_search_cli.py
import argparse
import json
import os
import time
import logging

from dotenv import load_dotenv
from openai import OpenAI


logging.basicConfig(
    level=logging.WARNING, 
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def enhance_query(query: str, method: str) -> str:
    """Enhance a query using the specified method via LLM."""
    client = get_llm_client()

    if method == "spell":
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    elif method == "rewrite":
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    elif method == "expand":
        prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        expanded_terms = (
            response.choices[0].message.content.strip()
        )

        if expanded_terms and expanded_terms != query:
            return f"{query} {expanded_terms}"

        return query

    else:
        return query

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content.strip()


def evaluate_results_llm(query: str, results: list[dict]) -> None:
    """Evaluate search results using an LLM and print a formatted report."""
    if not results:
        print("\nNo results to evaluate.")
        return

    client = get_llm_client()

    # Format results for the prompt
    formatted_results = []
    for i, res in enumerate(results, start=1):
        title = res.get("title", "Unknown")
        document = res.get("document", "")
        formatted_results.append(f"[{i}] {title}: {document}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )

        raw_content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

        scores = json.loads(raw_content)

        if not isinstance(scores, list) or len(scores) != len(results):
            print(
                f"\nEvaluation failed: Expected list of {len(results)} scores, "
                f"got {type(scores).__name__} with length {len(scores) if isinstance(scores, list) else 'N/A'}"
            )
            return

        print("\n=== LLM Evaluation Report ===")
        for i, (res, score) in enumerate(zip(results, scores), start=1):
            title = res.get("title", "Unknown")
            safe_score = max(0, min(3, int(score)))
            print(f"{i}. {title}: {safe_score}/3")

    except Exception as e:
        print(f"\nEvaluation failed: {type(e).__name__}: {e}")


def rerank_individual(
    query: str,
    results: list[dict],
) -> list[dict]:
    """Re-rank results individually using LLM scoring."""
    client = get_llm_client()

    for i, result in enumerate(results, start=1):
        doc_title = result.get("title", "")
        doc_text = result.get("document", "")

        print(
            f"[{i}/{len(results)}] Re-ranking: {doc_title}...",
            flush=True,
        )

        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc_title} - {doc_text}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                timeout=30,
            )

            elapsed = time.time() - start_time

            score_text = (
                response.choices[0].message.content.strip()
            )

            try:
                score = float(score_text)
                score = max(0.0, min(10.0, score))

            except ValueError:
                print(
                    f"  Invalid score returned: {score_text!r}"
                )
                score = 0.0

            print(
                f"  Score: {score:.1f}/10 ({elapsed:.1f}s)",
                flush=True,
            )

        except Exception as e:
            print(
                f"  ERROR: {type(e).__name__}: {e}",
                flush=True,
            )
            score = 0.0

        result["rerank_score"] = score

        time.sleep(3)

    return sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )


def rerank_batch(
    query: str,
    results: list[dict],
) -> list[dict]:
    """Re-rank results in a single batch LLM call."""
    client = get_llm_client()

    # Build document list string for the prompt
    doc_lines = []
    for result in results:
        doc_id = result.get("id", "")
        doc_title = result.get("title", "")
        doc_text = result.get("document", "")
        doc_lines.append(f"[{doc_id}] {doc_title} - {doc_text}")

    doc_list_str = "\n".join(doc_lines)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            timeout=60,
        )

        raw_content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present despite instructions
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

        ranked_ids = json.loads(raw_content)

        # Assign rerank_rank based on position in returned list
        id_to_rank = {}
        for rank, doc_id in enumerate(ranked_ids, start=1):
            id_to_rank[doc_id] = rank

        # Assign ranks to results; unmatched docs get last rank
        max_rank = len(results) + 1
        for result in results:
            doc_id = result.get("id")
            result["rerank_rank"] = id_to_rank.get(doc_id, max_rank)

        # Sort by rerank_rank ascending (lower = better)
        return sorted(
            results,
            key=lambda x: x["rerank_rank"],
        )

    except Exception as e:
        print(f"Batch re-ranking failed: {type(e).__name__}: {e}")
        print("Falling back to original RRF order.")
        # Fallback: assign rank based on existing order
        for rank, result in enumerate(results, start=1):
            result["rerank_rank"] = rank
        return results


def rerank_cross_encoder(
    query: str,
    results: list[dict],
) -> list[dict]:
    """Re-rank results using a local cross-encoder model."""
    from sentence_transformers import CrossEncoder

    # Load model once per call
    try:
        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L2-v2"
        )
    except Exception:
        # Fallback to CPU if GPU causes issues
        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L2-v2",
            device="cpu",
        )

    # Build pairs: [query, "title - document"]
    pairs = []
    for result in results:
        doc_title = result.get("title", "")
        doc_text = result.get("document", "")
        pairs.append([query, f"{doc_title} - {doc_text}"])

    # Predict scores for all pairs at once
    scores = cross_encoder.predict(pairs)

    # Assign scores to results
    for result, score in zip(results, scores):
        result["cross_encoder_score"] = float(score)

    # Sort by cross-encoder score descending
    return sorted(
        results,
        key=lambda x: x["cross_encoder_score"],
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # normalize command
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores using min-max normalization",
    )

    normalize_parser.add_argument(
        "scores",
        nargs="*",
        type=float,
        help="Zero or more scores to normalize",
    )

    # weighted-search command
    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Run weighted hybrid search",
    )

    weighted_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 score. Semantic weight is 1 - alpha.",
    )

    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    # rrf-search command
    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Run Reciprocal Rank Fusion hybrid search",
    )

    rrf_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF constant k (default: 60)",
    )

    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to apply after initial search",
    )

    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate results using LLM after search",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores

            if scores:
                min_score = min(scores)
                max_score = max(scores)

                if min_score == max_score:
                    normalized_scores = [
                        1.0 for _ in scores
                    ]

                else:
                    normalized_scores = [
                        (score - min_score)
                        / (max_score - min_score)
                        for score in scores
                    ]

                for score in normalized_scores:
                    print(f"* {score:.4f}")

        case "weighted-search":
            from hybrid_search import HybridSearch
            from lib.search_utils import load_movies

            documents = load_movies()
            searcher = HybridSearch(documents)

            results = searcher.weighted_search(
                query=args.query,
                alpha=args.alpha,
                limit=args.limit,
            )

            for rank, result in enumerate(
                results[: args.limit],
                start=1,
            ):
                document = result.get("document", "")

                if len(document) > 100:
                    document = document[:100] + "..."

                print(
                    f"{rank}. "
                    f"{result.get('title', 'Unknown')}"
                )

                print(
                    f"  Hybrid Score: "
                    f"{result.get('hybrid_score', 0.0):.3f}"
                )

                print(
                    f"  BM25: "
                    f"{result.get('bm25_score', 0.0):.3f}, "
                    f"Semantic: "
                    f"{result.get('semantic_score', 0.0):.3f}"
                )

                print(f"  {document}")

        case "rrf-search":
            from hybrid_search import HybridSearch
            from lib.search_utils import load_movies

            query = args.query

            # 1. Log the original query
            logger.debug("Original query: '%s'", query)

            if args.enhance:
                enhanced_query = enhance_query(
                    query,
                    args.enhance,
                )

                if enhanced_query != query:
                    # 2. Log the query after enhancements
                    logger.debug(
                        "Enhanced query (%s): '%s' -> '%s'",
                        args.enhance,
                        query,
                        enhanced_query,
                    )

                query = enhanced_query

            documents = load_movies()
            searcher = HybridSearch(documents)

            # Determine fetch limit based on rerank method
            if args.rerank_method in ("individual", "batch", "cross_encoder"):
                fetch_limit = args.limit * 5
            else:
                fetch_limit = args.limit

            results = searcher.rrf_search(
                query=query,
                k=args.k,
                limit=fetch_limit,
            )

            # 3. Log the results after RRF search (Before re-ranking)
            logger.debug(
                "Results after RRF search (Top %d fetched): %s", 
                len(results),
                [r.get('title') for r in results]
            )

            # Apply re-ranking if requested
            if args.rerank_method == "individual":
                results = rerank_individual(query, results)
            elif args.rerank_method == "batch":
                results = rerank_batch(query, results)
            elif args.rerank_method == "cross_encoder":
                results = rerank_cross_encoder(query, results)

            # 4. Log the final results after re-ranking
            if args.rerank_method:
                logger.debug(
                    "Final results after %s re-ranking: %s", 
                    args.rerank_method,
                    [r.get('title') for r in results[:args.limit]]
                )

            print(
                f"\nReciprocal Rank Fusion Results "
                f"for '{query}' (k={args.k}):\n"
            )

            display_results = results[: args.limit]

            for rank, result in enumerate(
                display_results,
                start=1,
            ):
                document = result.get("document", "")

                if len(document) > 100:
                    document = document[:100] + "..."

                bm25_rank = result.get("bm25_rank")
                semantic_rank = result.get("semantic_rank")

                bm25_str = (
                    str(bm25_rank)
                    if bm25_rank is not None
                    else "-"
                )

                semantic_str = (
                    str(semantic_rank)
                    if semantic_rank is not None
                    else "-"
                )

                print(
                    f"{rank}. "
                    f"{result.get('title', 'Unknown')}"
                )

                if args.rerank_method == "individual":
                    print(
                        f"   Re-rank Score: "
                        f"{result.get('rerank_score', 0.0):.3f}/10"
                    )
                elif args.rerank_method == "batch":
                    print(
                        f"   Re-rank Rank: "
                        f"{result.get('rerank_rank', '-')}"
                    )
                elif args.rerank_method == "cross_encoder":
                    print(
                        f"   Cross Encoder Score: "
                        f"{result.get('cross_encoder_score', 0.0):.3f}"
                    )

                print(
                    f"   RRF Score: "
                    f"{result.get('rrf_score', 0.0):.3f}"
                )

                print(
                    f"   BM25 Rank: {bm25_str}, "
                    f"Semantic Rank: {semantic_str}"
                )

                print(f"   {document}\n")

            # Run LLM evaluation if requested
            if args.evaluate:
                evaluate_results_llm(query, display_results)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()