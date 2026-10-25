"""
Gradio app: Chat, RAG Q&A, and Agent in one UI.

WHY GRADIO
----------
Gradio is the fastest way to put a model in front of people: a few
components wired to Python functions, with streaming, per-user sessions and
a shareable link built in. It's ideal for demos, internal tools, and
Hugging Face Spaces.

Run from the ai-framework folder:
    python apps/gradio_app.py          # http://127.0.0.1:7860

What to try:
    Chat     tell it your name, then ask it later (memory is per browser tab)
    RAG      "What does error NX-404 mean?" and check the cited sources
    Agent    "12 users on Pro: annual cost?" and read the tool trace
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import threading
from collections.abc import Iterator

import gradio as gr

from aiframe.workflows.agent import build_agent, run_agent
from aiframe.workflows.chat import SessionStore
from aiframe.workflows.rag import RAGPipeline
from config.settings import ConfigError, get_settings


class AppState:
    """Workflow objects shared by all users of this process, created on first use.

    Lazy creation lets the UI start even when .env is incomplete; each tab
    then shows a clear configuration error instead of the app crashing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: SessionStore | None = None
        self._rag: RAGPipeline | None = None
        self._agent = None

    def sessions(self) -> SessionStore:
        with self._lock:
            if self._sessions is None:
                get_settings().require_llm()
                self._sessions = SessionStore()
            return self._sessions

    def rag(self) -> RAGPipeline:
        with self._lock:
            if self._rag is None:
                rag = RAGPipeline()
                rag.ingest(
                    get_settings().data_dir
                )  # indexes data/ (includes data/sample)
                self._rag = rag
            return self._rag

    def agent(self):
        with self._lock:
            if self._agent is None:
                self._agent = build_agent()
            return self._agent


STATE = AppState()


def _session_id(request: gr.Request | None) -> str:
    # Gradio gives each browser tab its own session_hash: a natural per-user key.
    return getattr(request, "session_hash", None) or "default"


# ---------------------------------------------------------------------- #
# Handlers: plain functions, so they can be unit-tested without a browser
# ---------------------------------------------------------------------- #
def chat_fn(message: str, history: list, request: gr.Request) -> Iterator[str]:
    try:
        session = STATE.sessions().get(_session_id(request))
    except ConfigError as err:
        raise gr.Error(str(err)) from err
    if not history:  # the user cleared the chat in the UI, so start fresh
        session.reset()
    reply = ""
    for chunk in session.stream(message):
        reply += chunk
        yield reply  # Gradio expects the full text so far, not the delta


def rag_fn(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""
    try:
        result = STATE.rag().ask(question)
    except ConfigError as err:
        raise gr.Error(str(err)) from err
    sources = "**Sources:** " + ", ".join(f"`{s}`" for s in result.sources)
    return result.answer, sources


def agent_fn(question: str, request: gr.Request) -> tuple[str, list[list[str]]]:
    if not question.strip():
        return "", []
    try:
        result = run_agent(STATE.agent(), question, thread_id=_session_id(request))
    except ConfigError as err:
        raise gr.Error(str(err)) from err
    trace = [[c.name, str(c.args), c.result[:300]] for c in result.tool_calls]
    return result.answer, trace


def config_status() -> str:
    settings = get_settings()
    try:
        provider, model = settings.require_llm()
        status = f"Model: `{provider} / {model}`"
    except ConfigError:
        return "⚠️ **Not configured.** Run `python examples/01_check_setup.py`."
    try:
        e_provider, e_model = settings.require_embeddings()
        status += f" · Embeddings: `{e_provider} / {e_model}`"
    except ConfigError:
        status += " · Embeddings: not set (RAG tab disabled)"
    return status


# ---------------------------------------------------------------------- #
# UI
# ---------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Framework") as demo:
        gr.Markdown(f"# AI Framework\n{config_status()}")

        with gr.Tab("Chat"):
            gr.ChatInterface(
                chat_fn,
                examples=[
                    "Explain the bias-variance tradeoff in two sentences.",
                    "Give me three ideas for an MLOps portfolio project.",
                ],
            )

        with gr.Tab("RAG Q&A"):
            gr.Markdown(
                "Answers come only from the documents in `data/`, with sources."
            )
            question = gr.Textbox(
                label="Question", placeholder="Ask about the sample docs"
            )
            ask = gr.Button("Ask", variant="primary")
            answer = gr.Markdown()
            sources = gr.Markdown()
            gr.Examples(
                [
                    "What does error NX-404 mean?",
                    "How many PTO days do new employees get?",
                    "What is the parental leave policy?",
                ],
                inputs=question,
            )
            ask.click(rag_fn, question, [answer, sources])
            question.submit(rag_fn, question, [answer, sources])

        with gr.Tab("Agent"):
            gr.Markdown(
                "The agent picks tools itself. The table shows every call it made."
            )
            task = gr.Textbox(
                label="Task", placeholder="Ask something that needs a tool"
            )
            run = gr.Button("Run", variant="primary")
            result = gr.Markdown()
            trace = gr.Dataframe(
                headers=["tool", "args", "result"], label="Tool trace", wrap=True
            )
            gr.Examples(
                [
                    "A team of 12 is on the Pro plan. What is their annual cost?",
                    "What does NX-500 mean, and what day is it today?",
                ],
                inputs=task,
            )
            run.click(agent_fn, task, [result, trace])
            task.submit(agent_fn, task, [result, trace])

    return demo


if __name__ == "__main__":
    # TRY: launch(share=True) for a temporary public link to send someone.
    build_ui().launch()
