"""
Conversation Memory & Dialog Context Manager for RAG Agent.
"""

from typing import Any, Dict, List, Optional


class ConversationMemory:
    """Maintains multi-turn chat history and tracks active movie entities."""

    def __init__(self, system_prompt: str, max_turns: int = 10) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.messages: List[Dict[str, Any]] = []
        self.last_retrieved_movies: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        """Reset conversation back to initial system prompt."""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.last_retrieved_movies = []

    def add_user_message(self, content: str) -> None:
        """Record a user query."""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Record an assistant response, with optional tool call requests."""
        msg: Dict[str, Any] = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._trim_history()

    def add_tool_result(self, tool_call_id: str, tool_name: str, content: str) -> None:
        """Record the output of an executed tool call."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        })
        self._trim_history()

    def record_retrieved_movies(self, movies: List[Dict[str, Any]]) -> None:
        """Cache the most recently retrieved movies for fast contextual lookup."""
        if isinstance(movies, list):
            valid_movies = [m for m in movies if isinstance(m, dict) and "id" in m]
            if valid_movies:
                self.last_retrieved_movies = valid_movies

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return the current message sequence for API consumption."""
        return list(self.messages)

    def _trim_history(self) -> None:
        """
        Ensure conversation does not grow unbounded while preserving the system prompt
        and keeping complete tool call - tool response pairs intact.
        """
        # Keep system prompt at index 0, trim oldest turns if total messages exceed max_turns * 3
        max_total = self.max_turns * 4
        if len(self.messages) > max_total:
            # Keep index 0 (system prompt) and slice the tail
            self.messages = [self.messages[0]] + self.messages[-(max_total - 1):]
