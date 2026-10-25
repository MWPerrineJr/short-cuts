"""Workflows layer: the five patterns behind most AI applications.

    chat     conversational assistant with memory and streaming
    rag      answer questions from your own documents, with citations
    agent    LLM that decides which tools to call (LangGraph)
    extract  unstructured text -> validated Pydantic objects
    batch    async, rate-limited processing of many items

Each module depends only on ``aiframe.core``, so workflows stay
provider-agnostic and can be combined (e.g. an agent with a RAG tool).
"""
