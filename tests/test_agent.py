"""
Verification test suite for SearchTools, Agent Memory, and Agent Execution.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.core import RAGAgent
from agent.memory import ConversationMemory
from agent.tools import OPENAI_TOOLS_SCHEMA, SearchTools


def test_tools_initialization_and_schema():
    print("Testing Tool Schemas...")
    assert len(OPENAI_TOOLS_SCHEMA) >= 5
    tool_names = [t["function"]["name"] for t in OPENAI_TOOLS_SCHEMA]
    print(f"Tool names in schema: {tool_names}")
    assert "hybrid_search" in tool_names
    assert "bm25_search" in tool_names
    assert "semantic_search" in tool_names
    assert "get_movie_details" in tool_names
    assert "multimodal_image_search" in tool_names
    print("✓ Tool schemas verified.")


def test_search_execution():
    print("\nTesting Search Tools Execution...")
    tools = SearchTools()

    # Test BM25
    print("1. Testing BM25 Search for 'Paddington'...")
    bm25_res = tools.bm25_search("Paddington", limit=3)
    assert isinstance(bm25_res, list)
    assert len(bm25_res) > 0
    print(f"   BM25 found {len(bm25_res)} results. Top: {bm25_res[0]['title']}")

    # Test Movie Details
    test_id = bm25_res[0]["id"]
    print(f"2. Testing get_movie_details for ID {test_id}...")
    details = tools.get_movie_details(test_id)
    assert details["id"] == test_id
    assert "title" in details
    assert "description" in details
    print(f"   Details loaded: {details['title']}")

    # Test Semantic Search
    print("3. Testing Semantic Search for 'family friendly comedy bear'...")
    sem_res = tools.semantic_search("family friendly comedy bear", limit=3)
    assert isinstance(sem_res, list)
    assert len(sem_res) > 0
    print(f"   Semantic found {len(sem_res)} results. Top: {sem_res[0]['title']}")

    # Test Hybrid Search (RRF)
    print("4. Testing Hybrid Search (RRF) for 'space exploration black hole'...")
    hybrid_res = tools.hybrid_search("space exploration black hole", limit=3)
    assert isinstance(hybrid_res, list)
    assert len(hybrid_res) > 0
    print(f"   Hybrid found {len(hybrid_res)} results. Top: {hybrid_res[0]['title']}")

    print("✓ Search tools execution verified.")


def test_memory_management():
    print("\nTesting Conversation Memory...")
    mem = ConversationMemory(system_prompt="Test System", max_turns=3)
    assert len(mem.get_messages()) == 1

    mem.add_user_message("Hello")
    mem.add_assistant_message("Hi there!")
    assert len(mem.get_messages()) == 3

    mem.record_retrieved_movies([{"id": 100, "title": "Test Movie"}])
    assert len(mem.last_retrieved_movies) == 1
    assert mem.last_retrieved_movies[0]["id"] == 100

    mem.reset()
    assert len(mem.get_messages()) == 1
    assert len(mem.last_retrieved_movies) == 0
    print("✓ Conversation memory verified.")


def test_agent_multi_turn():
    print("\nTesting Agent Multi-Turn Execution...")
    tools = SearchTools()
    agent = RAGAgent(tools=tools)

    # Turn 1: Search
    print("Turn 1: User asks for animated bear movie...")
    resp1 = agent.run("scary horror bear movie")
    print(f"Agent response 1:\n{resp1[:200]}...")
    assert len(agent.memory.last_retrieved_movies) > 0

    # Turn 2: Follow-up details
    print("\nTurn 2: User asks for details of the first movie...")
    resp2 = agent.run("tell me more about the first one")
    print(f"Agent response 2:\n{resp2[:200]}...")
    assert "Synopsis" in resp2 or "Movie with ID" in resp2

    print("✓ Agent multi-turn execution verified.")


if __name__ == "__main__":
    test_tools_initialization_and_schema()
    test_search_execution()
    test_memory_management()
    test_agent_multi_turn()
    print("\n🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
