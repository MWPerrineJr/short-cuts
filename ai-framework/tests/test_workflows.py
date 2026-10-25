"""Tests for workflows and tools (Step 3). All offline, using fake models."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from aiframe.core.schemas import Contact, ContactList, Sentiment
from aiframe.tools import calculator, current_datetime, search_notes
from aiframe.workflows import batch as batch_mod
from aiframe.workflows.agent import build_agent, run_agent
from aiframe.workflows.batch import run_batch, summarize
from aiframe.workflows.chat import ChatSession, SessionStore
from aiframe.workflows.extract import extract, extract_many, to_dataframe
from aiframe.workflows.rag import RAGPipeline, load_documents, reciprocal_rank_fusion
from tests.conftest import SAMPLE_DIR, FakeStructured, make_fake_llm

# ---------------------------------------------------------------------- #
# Chat
# ---------------------------------------------------------------------- #


def test_chat_remembers_and_streams():
    session = ChatSession(llm=make_fake_llm("Hi Michael!", "Your name is Michael."))
    assert session.send("I'm Michael") == "Hi Michael!"
    assert "".join(session.stream("Name?")) == "Your name is Michael."
    assert session.turns == 2
    assert session.history[-1].content == "Your name is Michael."


def test_chat_history_is_trimmed():
    session = ChatSession(llm=make_fake_llm("ok"), max_turns=1)
    for text in ("a", "b", "c"):
        session.send(text)
    assert len(session._inputs("d")["history"]) == 2  # one turn = 2 messages


async def test_chat_async_stream():
    session = ChatSession(llm=make_fake_llm("async reply"))
    chunks = [c async for c in session.astream("hi")]
    assert "".join(chunks) == "async reply"
    assert session.turns == 1


def test_session_store_isolates_users():
    store = SessionStore(llm=make_fake_llm("ok"))
    store.get("u1").send("hello")
    assert store.get("u1").turns == 1
    assert store.get("u2").turns == 0
    store.drop("u1")
    assert len(store) == 1


# ---------------------------------------------------------------------- #
# RAG
# ---------------------------------------------------------------------- #


def test_load_sample_documents():
    docs = load_documents(SAMPLE_DIR)
    assert {d.metadata["source"] for d in docs} == {
        "company_handbook.md",
        "product_faq.md",
        "ml_glossary.md",
    }


def test_load_documents_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_keyword_search_finds_exact_code(rag):
    top = rag.keyword_search("What does error NX-404 mean?")[0]
    assert top.metadata["source"] == "product_faq.md"
    assert "NX-404" in top.page_content


def test_rrf_rewards_consensus():
    a, b, c = (
        Document(page_content=x, metadata={"chunk_id": i}) for i, x in enumerate("abc")
    )
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert fused[0].page_content == "b"


def test_rag_ask_returns_answer_sources_contexts(rag):
    result = rag.ask("How many PTO days?")
    assert result.answer.startswith("15 days")
    assert result.sources
    assert len(result.contexts) == rag.k


def test_retrieve_before_ingest_errors(fake_embeddings):
    pipeline = RAGPipeline(llm=make_fake_llm("x"), embeddings=fake_embeddings)
    with pytest.raises(RuntimeError, match="ingest"):
        pipeline.retrieve("q")


# ---------------------------------------------------------------------- #
# Tools
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("12*79*12", "11376"), ("(1250*0.07)/12", "7.29167"), ("-(2**3)", "-8")],
)
def test_calculator(expression, expected):
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "2**10000", "1/0", "open('x')", "x+1"],
)
def test_calculator_rejects_unsafe_input(expression):
    assert calculator.invoke({"expression": expression}).startswith("error")


def test_search_notes_and_datetime():
    assert "NX-500" in search_notes.invoke({"query": "NX-500 error"})
    assert search_notes.invoke({"query": "zzqqxx"}) == "No matching notes found."
    assert current_datetime.invoke({})


# ---------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------- #


def test_agent_traces_tool_calls_per_turn():
    llm = make_fake_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "12*79*12"}, "id": "c1"}
            ],
        ),
        AIMessage(content="Annual cost is $11,376."),
        AIMessage(content="Hello again."),
        loop=False,
    )
    agent = build_agent(llm=llm)

    first = run_agent(agent, "12 users on Pro?", thread_id="t1")
    assert first.answer == "Annual cost is $11,376."
    assert [(c.name, c.result) for c in first.tool_calls] == [("calculator", "11376")]

    second = run_agent(agent, "hi", thread_id="t1")
    assert second.answer == "Hello again."
    assert second.tool_calls == []  # trace covers only the current turn


# ---------------------------------------------------------------------- #
# Extraction
# ---------------------------------------------------------------------- #

CONTACTS = ContactList(
    contacts=[
        Contact(name="Dana Reyes", email="dana@acme.io"),
        Contact(name="Priya Shah"),
    ]
)


def test_extract_and_extract_many():
    llm = FakeStructured(CONTACTS)
    assert extract("text", ContactList, llm=llm) == CONTACTS
    assert (
        len(extract_many(["a", "b", "c"], ContactList, llm=llm, max_concurrency=2)) == 3
    )


def test_extract_to_dataframe_keeps_empty_and_failed_docs():
    df = to_dataframe([CONTACTS, ContactList(), ValueError("boom")], explode="contacts")
    assert len(df) == 4  # 2 contacts + 1 empty doc + 1 error row
    assert df["error"].notna().sum() == 1


# ---------------------------------------------------------------------- #
# Batch
# ---------------------------------------------------------------------- #


async def test_batch_order_retries_and_failures():
    calls: dict[str, int] = {}

    async def worker(item: str) -> str:
        calls[item] = calls.get(item, 0) + 1
        if item == "flaky" and calls[item] == 1:
            raise TimeoutError("transient")
        if item == "broken":
            raise ValueError("always fails")
        await asyncio.sleep(0)
        return item.upper()

    results = await run_batch(
        ["a", "flaky", "broken", "b"], worker, max_concurrency=2, progress=False
    )
    assert [r.output for r in results] == ["A", "FLAKY", None, "B"]
    assert results[1].attempts == 2 and results[1].ok
    assert results[2].attempts == 3 and "always fails" in results[2].error

    metrics = summarize(results)
    assert metrics["failed"] == 1
    assert metrics["error_rate"] == 0.25
    assert metrics["retries"] == 3


async def test_batch_respects_concurrency_limit():
    active = peak = 0

    async def worker(_: int) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    await run_batch(range(10), worker, max_concurrency=3, progress=False)
    assert peak == 3


async def test_classify_sentiment(monkeypatch, configured):
    fake = FakeStructured(Sentiment(label="positive", score=0.9, reason="praise"))
    monkeypatch.setattr(batch_mod, "get_llm", lambda **_: fake)
    results = await batch_mod.classify_sentiment(["great", "love it"], progress=False)
    assert [r.output.label for r in results] == ["positive", "positive"]
    assert batch_mod.to_dataframe(results)["label"].tolist() == ["positive", "positive"]
