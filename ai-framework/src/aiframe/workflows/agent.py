"""
Agent workflow: an LLM that chooses which tools to call (LangGraph).

WHY THIS PATTERN
----------------
A chain follows a fixed path; an agent decides its own path at runtime:

    question -> model -> (tool call -> tool result -> model)* -> answer

The model sees each tool's schema, emits a tool call when it needs one, reads
the result, and repeats until it can answer. LangChain's ``create_agent``
builds this loop as a LangGraph graph, which gives you:

- Memory per conversation via a checkpointer + ``thread_id``.
- A step limit (``recursion_limit``) so a confused agent can't loop forever.
- A full trace of every tool call, for debugging and evaluation.

Rule of thumb: prefer a chain when the steps are known in advance. Agents
add flexibility at the cost of latency, tokens and predictability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from aiframe.core import get_llm
from aiframe.tools import DEFAULT_TOOLS

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant with tools. Use the calculator for any math, "
    "search_notes for company, product or ML-glossary questions, and "
    "current_datetime for anything date-related. If the tools can't answer, say so."
)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: str


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0


def build_agent(
    tools: Sequence[BaseTool] | None = None,
    llm: BaseChatModel | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    memory: bool = True,
):
    """Create a tool-calling agent. With ``memory=True``, conversations persist per thread_id."""
    return create_agent(
        model=llm or get_llm(temperature=0),
        tools=list(tools if tools is not None else DEFAULT_TOOLS),
        system_prompt=system_prompt,
        checkpointer=InMemorySaver() if memory else None,
    )


def run_agent(
    agent, question: str, thread_id: str = "default", max_steps: int = 10
) -> AgentResult:
    """Ask the agent a question and return the answer plus a trace of tool calls."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_steps * 2,
    }
    state = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)

    # Only this turn's messages: everything after the last user message.
    messages = state["messages"]
    last_user = max(i for i, m in enumerate(messages) if m.type == "human")
    turn = messages[last_user + 1 :]

    results = {
        m.tool_call_id: str(m.content) for m in turn if isinstance(m, ToolMessage)
    }
    calls = [
        ToolCall(name=c["name"], args=c["args"], result=results.get(c["id"], ""))
        for m in turn
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]
    final = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
    answer = final.text if final is not None else ""
    return AgentResult(answer=answer, tool_calls=calls, steps=len(turn))
