"""
Core Agent Execution Engine.
Orchestrates multi-turn conversation, OpenAI function calling, autonomous
tool execution, and resilient fallback retrieval.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from agent.memory import ConversationMemory
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools import OPENAI_TOOLS_SCHEMA, SearchTools


class RAGAgent:
    """Autonomous RAG Agent for movie discovery, recommendation, and Q&A."""

    def __init__(
        self,
        tools: Optional[SearchTools] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_iterations: int = 5,
    ) -> None:
        load_dotenv()
        self.tools = tools or SearchTools()
        self.model = model or os.getenv("AGENT_MODEL", "openrouter/free")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.max_iterations = max_iterations
        self.memory = ConversationMemory(system_prompt=AGENT_SYSTEM_PROMPT)

        self._client: Optional[OpenAI] = None
        if self.api_key:
            try:
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            except Exception:
                self._client = None

    @property
    def client(self) -> Optional[OpenAI]:
        if self._client is None and self.api_key:
            try:
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            except Exception:
                self._client = None
        return self._client

    def reset(self) -> None:
        """Reset conversation context."""
        self.memory.reset()

    def run(
        self,
        user_prompt: str,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_end: Optional[Callable[[str, Any], None]] = None,
        on_thought: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute a user request through the agent loop.
        Handles autonomous tool calling, result assimilation, and conversational memory.
        """
        self.memory.add_user_message(user_prompt)

        # If no client or API key, run in local rule-based fallback mode
        if not self.client:
            return self._run_local_fallback(
                user_prompt,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                reason="No active API key provided",
            )

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            try:
                messages = self.memory.get_messages()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=OPENAI_TOOLS_SCHEMA,
                    tool_choice="auto",
                    timeout=45,
                )
            except Exception as e:
                # If API call fails (e.g. 401 unauthorized, rate limit, or model error), gracefully fall back
                if on_thought:
                    on_thought(f"LLM call encountered an issue ({type(e).__name__}: {e}). Switching to local retrieval engine...")
                return self._run_local_fallback(
                    user_prompt,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    reason=f"API Error ({type(e).__name__}: {e})",
                )

            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None)

            # If the model requested tool calls
            if tool_calls:
                # Record assistant message with tool calls
                raw_tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
                self.memory.add_assistant_message(
                    content=message.content,
                    tool_calls=raw_tool_calls,
                )

                # Execute each tool
                for tc in tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except Exception:
                        args = {}

                    if on_tool_start:
                        on_tool_start(fn_name, args)

                    tool_output = self.tools.execute_tool(fn_name, args)

                    if isinstance(tool_output, list):
                        self.memory.record_retrieved_movies(tool_output)
                    elif isinstance(tool_output, dict) and "id" in tool_output:
                        self.memory.record_retrieved_movies([tool_output])

                    if on_tool_end:
                        on_tool_end(fn_name, tool_output)

                    self.memory.add_tool_result(
                        tool_call_id=tc.id,
                        tool_name=fn_name,
                        content=json.dumps(tool_output, ensure_ascii=False),
                    )

                # Loop continues to give the LLM the chance to synthesize or make another tool call
                continue

            # No tool calls: final natural language answer reached
            final_content = message.content or "I couldn't find relevant information for that query."
            self.memory.add_assistant_message(content=final_content)
            return final_content

        # If max iterations reached, formulate fallback summary
        fallback_msg = "I gathered the requested movie search results but reached the maximum reasoning steps. Please refine your query."
        self.memory.add_assistant_message(content=fallback_msg)
        return fallback_msg

    def _run_local_fallback(
        self,
        user_prompt: str,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_end: Optional[Callable[[str, Any], None]] = None,
        reason: str = "",
    ) -> str:
        """
        Resilient heuristic retrieval agent fallback.
        Parses user intent, executes appropriate retrieval tools, and formats grounded results.
        """
        prompt_lower = user_prompt.lower().strip()

        # Check for image search intent
        image_match = re.search(r"(?:image|poster|pic|picture)[\s:=]+([^\s]+\.(?:jpg|jpeg|png|webp))", user_prompt, re.I)
        if image_match:
            img_path = image_match.group(1)
            tool_name = "multimodal_image_search"
            args = {"image_path": img_path, "limit": 5}
        # Check for movie details intent
        elif any(kw in prompt_lower for kw in ["tell me more", "details", "synopsis", "plot of", "movie #", "id:"]) or (
            ("first" in prompt_lower or "second" in prompt_lower or "third" in prompt_lower) and self.memory.last_retrieved_movies
        ):
            target_id = None
            id_match = re.search(r"(?:id[:\s]+|movie\s*#?\s*)(\d+)", prompt_lower)
            if id_match:
                target_id = int(id_match.group(1))
            elif "first" in prompt_lower and len(self.memory.last_retrieved_movies) >= 1:
                target_id = self.memory.last_retrieved_movies[0]["id"]
            elif "second" in prompt_lower and len(self.memory.last_retrieved_movies) >= 2:
                target_id = self.memory.last_retrieved_movies[1]["id"]
            elif "third" in prompt_lower and len(self.memory.last_retrieved_movies) >= 3:
                target_id = self.memory.last_retrieved_movies[2]["id"]
            elif self.memory.last_retrieved_movies:
                target_id = self.memory.last_retrieved_movies[0]["id"]

            if target_id is not None:
                tool_name = "get_movie_details"
                args = {"movie_id": target_id}
            else:
                tool_name = "hybrid_search"
                args = {"query": user_prompt, "limit": 5}
        # Exact keyword search if quotes or specific name patterns
        elif '"' in user_prompt or "'" in user_prompt:
            extracted_term = re.findall(r"['\"](.*?)['\"]", user_prompt)
            term = extracted_term[0] if extracted_term else user_prompt
            tool_name = "bm25_search"
            args = {"query": term, "limit": 5}
        # Default: Hybrid Search (BM25 + Semantic RRF)
        else:
            tool_name = "hybrid_search"
            args = {"query": user_prompt, "limit": 5}

        # Execute chosen tool
        if on_tool_start:
            on_tool_start(tool_name, args)

        results = self.tools.execute_tool(tool_name, args)

        if on_tool_end:
            on_tool_end(tool_name, results)

        # Synthesize grounded answer
        if tool_name == "get_movie_details":
            if "error" in results:
                response_text = f"❌ {results['error']}"
            else:
                title = results.get("title", "Unknown")
                m_id = results.get("id", "")
                desc = results.get("description", "No description available.")
                response_text = (
                    f"🎬 **{title}** [ID: {m_id}]\n\n"
                    f"**Synopsis**: {desc}\n"
                )
                self.memory.record_retrieved_movies([results])
        else:
            if not results:
                response_text = f"No movies found matching your query: *'{user_prompt}'*."
            else:
                lines = [f"Here are the top matches found in the Webflyx catalog for *'{user_prompt}'*:\n"]
                self.memory.record_retrieved_movies(results)
                for idx, r in enumerate(results, start=1):
                    title = r.get("title", "Unknown")
                    doc_id = r.get("id", "")
                    desc = r.get("description", "")
                    score_info = ""
                    if "rrf_score" in r:
                        score_info = f" (RRF Score: {r['rrf_score']})"
                    elif "similarity_score" in r:
                        score_info = f" (Similarity: {r['similarity_score']})"
                    elif "bm25_score" in r:
                        score_info = f" (BM25 Score: {r['bm25_score']})"

                    lines.append(f"{idx}. **{title}** [ID: {doc_id}]{score_info}")
                    if desc:
                        lines.append(f"   {desc.strip()}...")
                    lines.append("")

                lines.append("💡 *You can ask me for more details on any movie (e.g. 'tell me more about the first one')!*")
                response_text = "\n".join(lines)

        self.memory.add_assistant_message(content=response_text)
        return response_text
