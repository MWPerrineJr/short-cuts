"""
FastAPI service: the workflows as a REST API.

WHY AN API
----------
UIs are for people; APIs are for other programs: a web front end, a mobile
app, a scheduled job, another team's service. This is also the shape most
ML models take in production: a container serving HTTP endpoints, with a
health check the platform polls to decide whether to send it traffic.

Patterns worth noticing:
- Request/response bodies are Pydantic models: validated input, typed
  output, and free interactive docs at /docs.
- Blocking LLM calls use plain ``def`` endpoints. FastAPI runs those in a
  thread pool, so one slow call doesn't freeze the server. Streaming uses
  ``async def`` with the model's async API.
- Missing configuration returns HTTP 503 (service unavailable), not a 500.

Run from the ai-framework folder:
    uvicorn apps.api:app --reload --port 8000
    open http://127.0.0.1:8000/docs

Try it:
    curl localhost:8000/health
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \\
         -d '{"session_id": "u1", "message": "Hi, I am Michael"}'
    curl -N -X POST localhost:8000/chat/stream -H "Content-Type: application/json" \\
         -d '{"session_id": "u1", "message": "What is my name?"}'
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from aiframe import __version__
from aiframe.core.schemas import ContactList, Sentiment
from aiframe.workflows.chat import SessionStore
from aiframe.workflows.extract import extract
from aiframe.workflows.rag import RAGPipeline
from config.settings import ConfigError, get_settings

EXTRACT_SCHEMAS: dict[str, type[BaseModel]] = {
    "contacts": ContactList,
    "sentiment": Sentiment,
}


# ---------------------------------------------------------------------- #
# Request / response models
# ---------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turns: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    include_contexts: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str] | None = None


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    kind: Literal["contacts", "sentiment"] = "contacts"


# ---------------------------------------------------------------------- #
# App state: created lazily so /health works even before .env is complete
# ---------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_settings().apply_tracing()  # fail fast at startup if tracing is half-configured
    app.state.lock = threading.Lock()
    app.state.sessions = None
    app.state.rag = None
    yield


app = FastAPI(title="AI Framework API", version=__version__, lifespan=lifespan)


@app.exception_handler(ConfigError)
async def config_error_handler(_: Request, exc: ConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _sessions(request: Request) -> SessionStore:
    state = request.app.state
    with state.lock:
        if state.sessions is None:
            get_settings().require_llm()
            state.sessions = SessionStore()
        return state.sessions


def _rag(request: Request) -> RAGPipeline:
    state = request.app.state
    with (
        state.lock
    ):  # first request builds the index; others wait instead of duplicating
        if state.rag is None:
            rag = RAGPipeline()
            rag.ingest(get_settings().data_dir)
            state.rag = rag
        return state.rag


# ---------------------------------------------------------------------- #
# Endpoints
# ---------------------------------------------------------------------- #
@app.get("/health")
def health(request: Request) -> JSONResponse:
    """Readiness probe: 200 when a chat model is configured, 503 otherwise."""
    settings = get_settings()
    try:
        provider, model = settings.require_llm()
    except ConfigError as err:
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "detail": str(err)}
        )
    try:
        settings.require_embeddings()
        embeddings = True
    except ConfigError:
        embeddings = False
    sessions = request.app.state.sessions
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "llm": f"{provider}/{model}",
            "embeddings_configured": embeddings,
            "active_sessions": len(sessions) if sessions else 0,
        }
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    session = _sessions(request).get(body.session_id)
    reply = session.send(body.message)
    return ChatResponse(session_id=body.session_id, reply=reply, turns=session.turns)


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: one ``data:`` line per chunk, then ``data: [DONE]``."""
    session = _sessions(request).get(body.session_id)

    async def events() -> AsyncIterator[str]:
        async for chunk in session.astream(body.message):
            # JSON-encode each chunk so newlines inside it can't break SSE framing.
            yield f"data: {json.dumps({'delta': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/chat/{session_id}", status_code=204)
def end_chat(session_id: str, request: Request) -> Response:
    if request.app.state.sessions is not None:
        request.app.state.sessions.drop(session_id)
    return Response(status_code=204)


@app.post("/rag/ask", response_model=AskResponse)
def rag_ask(body: AskRequest, request: Request) -> AskResponse:
    result = _rag(request).ask(body.question)
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        contexts=result.contexts if body.include_contexts else None,
    )


@app.post("/extract")
def extract_endpoint(body: ExtractRequest) -> dict[str, Any]:
    """Structured extraction. ``kind`` picks the output schema."""
    result = extract(body.text, EXTRACT_SCHEMAS[body.kind])
    return {"kind": body.kind, "data": result.model_dump(mode="json")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.api:app", host="127.0.0.1", port=8000, reload=True)
