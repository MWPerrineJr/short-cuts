"""Tests for the FastAPI, Gradio, and Streamlit apps (Step 4)."""

from __future__ import annotations

import json

import pytest

from aiframe.core.schemas import Contact, ContactList, Sentiment
from aiframe.workflows import batch as batch_mod
from aiframe.workflows import chat as chat_mod
from aiframe.workflows import extract as extract_mod
from tests.conftest import ROOT, FakeStructured, make_fake_llm

pytestmark = pytest.mark.slow

REPLY = "Nice to meet you, Michael."


@pytest.fixture
def fake_workflows(monkeypatch, configured):
    """Configured settings + fake models wherever the apps create them."""
    monkeypatch.setattr(chat_mod, "get_llm", lambda **_: make_fake_llm(REPLY))
    monkeypatch.setattr(
        extract_mod,
        "get_llm",
        lambda **_: FakeStructured(
            ContactList(contacts=[Contact(name="Dana", email="d@a.io")])
        ),
    )
    monkeypatch.setattr(
        batch_mod,
        "get_llm",
        lambda **_: FakeStructured(
            Sentiment(label="positive", score=0.8, reason="praise")
        ),
    )


# ---------------------------------------------------------------------- #
# FastAPI
# ---------------------------------------------------------------------- #


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from apps.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_api_not_configured_returns_503(client):
    assert client.get("/health").status_code == 503
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert "LLM_PROVIDER" in response.json()["detail"]


def test_api_health_ok(fake_workflows, client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == "openai/gpt-test"


def test_api_chat_memory_and_isolation(fake_workflows, client):
    assert (
        client.post("/chat", json={"session_id": "u1", "message": "hi"}).json()["reply"]
        == REPLY
    )
    assert (
        client.post("/chat", json={"session_id": "u1", "message": "again"}).json()[
            "turns"
        ]
        == 2
    )
    assert (
        client.post("/chat", json={"session_id": "u2", "message": "x"}).json()["turns"]
        == 1
    )
    assert client.delete("/chat/u1").status_code == 204
    assert client.get("/health").json()["active_sessions"] == 1


def test_api_chat_stream_sse(fake_workflows, client):
    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert "".join(json.loads(line[6:])["delta"] for line in lines[:-1]) == REPLY


def test_api_validation_errors(fake_workflows, client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert (
        client.post("/extract", json={"text": "x", "kind": "bogus"}).status_code == 422
    )


def test_api_extract(fake_workflows, client):
    body = client.post(
        "/extract", json={"text": "Dana d@a.io", "kind": "contacts"}
    ).json()
    assert body["data"]["contacts"][0]["email"] == "d@a.io"


def test_api_rag(fake_workflows, client, rag):
    client.app.state.rag = rag
    body = client.post(
        "/rag/ask", json={"question": "PTO?", "include_contexts": True}
    ).json()
    assert body["answer"].startswith("15 days")
    assert len(body["contexts"]) == rag.k
    assert client.post("/rag/ask", json={"question": "PTO?"}).json()["contexts"] is None


# ---------------------------------------------------------------------- #
# Gradio
# ---------------------------------------------------------------------- #


class _Request:
    session_hash = "tab-1"


@pytest.fixture
def gradio_app(monkeypatch):
    from apps import gradio_app

    monkeypatch.setattr(gradio_app, "STATE", gradio_app.AppState())
    return gradio_app


def test_gradio_chat_streams_and_resets(fake_workflows, gradio_app):
    updates = list(gradio_app.chat_fn("hello", [], _Request()))
    assert updates[-1] == REPLY and len(updates) > 1  # cumulative text
    session = gradio_app.STATE.sessions().get("tab-1")
    list(
        gradio_app.chat_fn("again", [{"role": "user", "content": "hello"}], _Request())
    )
    assert session.turns == 2
    list(gradio_app.chat_fn("new chat", [], _Request()))  # UI cleared
    assert session.turns == 1


def test_gradio_unconfigured_shows_error(gradio_app):
    import gradio as gr

    with pytest.raises(gr.Error):
        list(gradio_app.chat_fn("x", [], _Request()))
    assert "Not configured" in gradio_app.config_status()


def test_gradio_rag_and_ui(fake_workflows, gradio_app, rag):
    gradio_app.STATE._rag = rag
    answer, sources = gradio_app.rag_fn("PTO?")
    assert answer.startswith("15 days") and "Sources" in sources
    assert gradio_app.rag_fn("   ") == ("", "")
    assert gradio_app.build_ui() is not None


# ---------------------------------------------------------------------- #
# Streamlit
# ---------------------------------------------------------------------- #

STREAMLIT_APP = str(ROOT / "apps" / "streamlit_app.py")


def test_streamlit_unconfigured_stops_cleanly():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STREAMLIT_APP, default_timeout=60).run()
    assert not app.exception
    assert "LLM_PROVIDER" in app.error[0].value


def test_streamlit_chat_and_batch(fake_workflows):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STREAMLIT_APP, default_timeout=60).run()
    assert app.success[0].value == "openai / gpt-test"

    app.chat_input[0].set_value("hello").run()
    app.chat_input[0].set_value("again").run()
    assert not app.exception
    assert len(app.chat_message) == 4  # history survives reruns

    app.sidebar.radio[0].set_value("Batch sentiment").run()
    app.button[0].click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Error rate"] == "0%"
    assert set(app.dataframe[0].value["label"]) == {"positive"}
