import argparse
import json
import os
import sys

# Ensure cli directory is on path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from hybrid_search import HybridSearch
from lib.search_utils import load_movies


def load_golden_dataset(filepath: str = "data/golden_dataset.json") -> list[dict]:
    """Load the golden dataset for evaluation."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def calculate_precision_at_k(retrieved_titles: list[str], relevant_titles: list[str], k: int) -> float:
    """
    Calculate Precision@K.

    Precision@K = (number of relevant docs in top K results) / K
    """
    if k == 0:
        return 0.0

    relevant_lower = {title.lower().strip() for title in relevant_titles}

    relevant_count = sum(
        1 for title in retrieved_titles
        if title.lower().strip() in relevant_lower
    )

    return relevant_count / k


def calculate_recall_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    """
    Calculate Recall@K.

    Recall@K = (number of relevant docs found in top K results) / (total number of relevant docs)
    """
    if not relevant_titles:
        return 0.0

    retrieved_lower = {title.lower().strip() for title in retrieved_titles}

    relevant_found = sum(
        1 for title in relevant_titles
        if title.lower().strip() in retrieved_lower
    )

    return relevant_found / len(relevant_titles)


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 Score.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load data
    documents = load_movies()
    golden_dataset = load_golden_dataset()

    # Initialize hybrid search
    searcher = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]

        # Run RRF search with k=60 and limit from args
        results = searcher.rrf_search(query=query, k=60, limit=limit)

        # Extract retrieved titles
        retrieved_titles = [result.get("title", "Unknown") for result in results]

        # Calculate metrics
        precision = calculate_precision_at_k(retrieved_titles, relevant_titles, limit)
        recall = calculate_recall_at_k(retrieved_titles, relevant_titles)
        f1 = calculate_f1(precision, recall)

        # Format output
        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(relevant_titles)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}")
        print()


if __name__ == "__main__":
    main()