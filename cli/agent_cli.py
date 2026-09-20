#!/usr/bin/env python3
"""
Interactive CLI Interface for the Webflyx RAG Search Agent.
Enables real-time conversational search, multi-turn follow-ups, and tool-call inspection.
"""

import argparse
import os
import sys

# Ensure root workspace is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.core import RAGAgent
from agent.tools import SearchTools


# ANSI styling helpers
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner() -> None:
    print(f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║               🎬 Webflyx AI RAG Search Agent                        ║
║     Hybrid Retrieval (BM25 + Semantic) • Multimodal • Re-ranking     ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
{DIM}Type your movie question or search query. Commands: /reset, /tools, /history, /exit{RESET}
""")


def on_tool_start(tool_name: str, args: dict) -> None:
    arg_summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"\n{YELLOW}⚙️  [Agent invoking tool: {BOLD}{tool_name}{RESET}{YELLOW}({arg_summary})]{RESET}", flush=True)


def on_tool_end(tool_name: str, output: any) -> None:
    if isinstance(output, list):
        count = len(output)
        print(f"{GREEN}✓  [{tool_name} returned {count} items]{RESET}\n", flush=True)
    elif isinstance(output, dict) and "error" in output:
        print(f"{MAGENTA}⚠  [{tool_name} returned error: {output['error']}]{RESET}\n", flush=True)
    else:
        print(f"{GREEN}✓  [{tool_name} completed]{RESET}\n", flush=True)


def on_thought(thought: str) -> None:
    print(f"{DIM}🧠 {thought}{RESET}", flush=True)


def run_interactive_session(agent: RAGAgent) -> None:
    print_banner()

    while True:
        try:
            user_input = input(f"\n{BOLD}{CYAN}You > {RESET}").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print(f"\n{CYAN}Goodbye! Enjoy your movies on Webflyx! 🍿{RESET}\n")
                break

            if user_input.lower() == "/reset":
                agent.reset()
                print(f"{GREEN}Conversation context reset.{RESET}")
                continue

            if user_input.lower() == "/tools":
                print(f"\n{BOLD}Registered Agent Tools:{RESET}")
                print("  • hybrid_search(query, limit): BM25 + Semantic Vector Search with RRF")
                print("  • bm25_search(query, limit): Exact lexical keyword matching")
                print("  • semantic_search(query, limit): Dense chunk embeddings similarity")
                print("  • get_movie_details(movie_id): Full description and metadata")
                print("  • multimodal_image_search(image_path, limit): CLIP image-to-text search")
                print("  • cross_encode_rerank(query, candidate_ids): Neural cross-encoder")
                continue

            if user_input.lower() == "/history":
                print(f"\n{BOLD}Conversation History:{RESET}")
                for msg in agent.memory.messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "system":
                        continue
                    if role == "user":
                        print(f"{CYAN}User:{RESET} {content}")
                    elif role == "assistant":
                        print(f"{MAGENTA}Agent:{RESET} {content[:120]}..." if len(content) > 120 else f"{MAGENTA}Agent:{RESET} {content}")
                    elif role == "tool":
                        print(f"{DIM}Tool ({msg.get('name')}): {content[:80]}...{RESET}")
                continue

            print(f"\n{MAGENTA}{BOLD}Agent CineBot:{RESET}")
            response = agent.run(
                user_input,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                on_thought=on_thought,
            )
            print(f"\n{response}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Session ended. See you next time! 🎬{RESET}\n")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Webflyx AI RAG Search Agent CLI")
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single query and exit instead of interactive mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier on OpenRouter/OpenAI.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/movies.json",
        help="Path to movies.json dataset.",
    )

    args = parser.parse_args()

    # Preload search tools
    tools = SearchTools(data_path=args.data)
    agent = RAGAgent(tools=tools, model=args.model)

    if args.query:
        print(f"\n{CYAN}{BOLD}Query:{RESET} {args.query}\n")
        response = agent.run(
            args.query,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_thought=on_thought,
        )
        print(f"\n{MAGENTA}{BOLD}Agent CineBot:{RESET}\n\n{response}\n")
    else:
        run_interactive_session(agent)


if __name__ == "__main__":
    main()
