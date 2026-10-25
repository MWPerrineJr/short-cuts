"""
Streamlit app: chat plus a batch-analysis dashboard.

WHY STREAMLIT
-------------
Streamlit turns a Python script into a data app. It suits analysis-heavy
tools: metrics, charts and DataFrames come built in. The one idea to learn:
the script RERUNS top to bottom on every interaction, so anything that must
survive a rerun (chat history, results) lives in ``st.session_state``.

Run from the ai-framework folder:
    streamlit run apps/streamlit_app.py      # http://localhost:8501
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import asyncio

import streamlit as st

from aiframe.workflows.batch import classify_sentiment, summarize, to_dataframe
from aiframe.workflows.chat import ChatSession
from config.settings import ConfigError, get_settings

SAMPLE_REVIEWS = [
    "Setup took five minutes and the dashboards look great.",
    "Forecasts are way off for our seasonal products. Disappointed.",
    "It's fine. Does what it says, nothing special.",
    "Support fixed our NX-404 issue within an hour. Impressed!",
    "Pricing jumped at renewal with no warning. Considering alternatives.",
    "Scheduled reports keep timing out (NX-500). Very frustrating.",
    "Our execs actually read the weekly reports now. Huge win.",
    "Exports to CSV drop columns randomly. Unusable for finance.",
]

st.set_page_config(page_title="AI Framework", page_icon="🧭", layout="wide")
settings = get_settings()

# ---------------------------------------------------------------------- #
# Sidebar: configuration status + navigation
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("AI Framework")
    try:
        provider, model = settings.require_llm()
    except ConfigError as err:
        st.error(str(err))
        st.info("Fix `.env`, then rerun. `python examples/01_check_setup.py` helps.")
        st.stop()  # nothing below runs until config is valid
    st.success(f"{provider} / {model}")
    page = st.radio("Workflow", ["Chat", "Batch sentiment"])


# ---------------------------------------------------------------------- #
# Chat
# ---------------------------------------------------------------------- #
def chat_page() -> None:
    st.title("Chat")
    if "chat" not in st.session_state:
        st.session_state.chat = ChatSession()
    session: ChatSession = st.session_state.chat

    if st.sidebar.button("Clear chat"):
        session.reset()

    for msg in session.history:  # replay history on every rerun
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask anything"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.write_stream(session.stream(prompt))


# ---------------------------------------------------------------------- #
# Batch sentiment dashboard
# ---------------------------------------------------------------------- #
def batch_page() -> None:
    st.title("Batch sentiment")
    st.caption(
        "Classify many texts concurrently, then analyze the run like any dataset."
    )

    text = st.text_area(
        "One text per line", value="\n".join(SAMPLE_REVIEWS), height=220
    )
    concurrency = st.slider("Max concurrent LLM calls", 1, 20, settings.max_concurrency)

    if st.button("Classify", type="primary"):
        texts = [line.strip() for line in text.splitlines() if line.strip()]
        with st.spinner(f"Classifying {len(texts)} texts..."):
            results = asyncio.run(
                classify_sentiment(texts, max_concurrency=concurrency, progress=False)
            )
        st.session_state.batch = (to_dataframe(results), summarize(results))

    if "batch" not in st.session_state:
        return
    df, metrics = st.session_state.batch

    cols = st.columns(4)
    cols[0].metric("Items", metrics["items"])
    cols[1].metric("Error rate", f"{metrics['error_rate']:.0%}")
    cols[2].metric("p50 latency", f"{metrics['p50_latency_s']:.2f}s")
    cols[3].metric("p95 latency", f"{metrics['p95_latency_s']:.2f}s")

    if "label" in df:
        left, right = st.columns(2)
        left.subheader("Label distribution")
        left.bar_chart(df["label"].value_counts())
        right.subheader("Score vs. latency")
        right.scatter_chart(df, x="latency_s", y="score", color="label")

    shown = [
        c
        for c in ("input", "label", "score", "reason", "latency_s", "error")
        if c in df
    ]
    st.dataframe(df[shown], hide_index=True)
    st.download_button(
        "Download CSV", df.to_csv(index=False), "sentiment_results.csv", "text/csv"
    )
    # TRY: paste 50 real reviews and compare p95 latency at concurrency 1 vs 10.


chat_page() if page == "Chat" else batch_page()
