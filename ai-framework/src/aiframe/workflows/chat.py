"""
Chat workflow: conversation memory + streaming.

WHY THIS PATTERN
----------------
LLM APIs are stateless: the model only "remembers" what you send it each
call. A chat session therefore keeps the message history itself and replays
it on every turn. Two practical concerns follow:

- Context windows are finite and tokens cost money, so history is trimmed
  to the most recent ``max_turns`` exchanges.
- Users expect text to appear as it is generated, so every method has a
  streaming variant.

``SessionStore`` maps a session id to a ``ChatSession``. The Gradio,
Streamlit and FastAPI apps (Step 4) use it to serve many users at once.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from aiframe.core import get_llm, get_prompt


@dataclass
class ChatSession:
    """A single conversation. Not thread-safe: use one session per user."""

    llm: BaseChatModel | None = None
    prompt: ChatPromptTemplate | None = None
    max_turns: int = 10
    history: list[BaseMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.llm = self.llm or get_llm()
        self.prompt = self.prompt or get_prompt("assistant")
        self._chain = self.prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------ #
    def _inputs(self, text: str) -> dict:
        # Keep the last N exchanges (2 messages per turn).
        return {"input": text, "history": self.history[-2 * self.max_turns :]}

    def _remember(self, user_text: str, reply: str) -> None:
        self.history += [HumanMessage(user_text), AIMessage(reply)]

    # ------------------------------------------------------------------ #
    def send(self, text: str) -> str:
        """Send a message and return the full reply."""
        reply = self._chain.invoke(self._inputs(text))
        self._remember(text, reply)
        return reply

    def stream(self, text: str) -> Iterator[str]:
        """Yield the reply chunk by chunk; history is saved when the stream ends."""
        parts: list[str] = []
        for chunk in self._chain.stream(self._inputs(text)):
            parts.append(chunk)
            yield chunk
        self._remember(text, "".join(parts))

    async def astream(self, text: str) -> AsyncIterator[str]:
        """Async streaming, for FastAPI and other async servers."""
        parts: list[str] = []
        async for chunk in self._chain.astream(self._inputs(text)):
            parts.append(chunk)
            yield chunk
        self._remember(text, "".join(parts))

    def reset(self) -> None:
        self.history.clear()

    @property
    def turns(self) -> int:
        return len(self.history) // 2


class SessionStore:
    """In-memory map of session id -> ChatSession.

    Fine for demos and single-process apps. For production, persist history
    in Redis or a database so it survives restarts and scales across workers.
    """

    def __init__(self, **session_kwargs) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._kwargs = session_kwargs

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(**self._kwargs)
        return self._sessions[session_id]

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)
