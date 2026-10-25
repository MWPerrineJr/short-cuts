"""
RAG workflow: retrieval-augmented generation with hybrid search.

WHY THIS PATTERN
----------------
LLMs don't know your private documents and can hallucinate. RAG fixes both by
retrieving relevant passages and instructing the model to answer ONLY from
them, with citations. The pipeline:

    load -> split -> embed + index -> retrieve (hybrid) -> answer with sources

Hybrid retrieval combines two rankers that fail differently:

- Vector search finds passages with similar MEANING ("time off" ~ "vacation").
- BM25 keyword search finds EXACT terms (error codes, product names, IDs).

Their ranked lists are merged with Reciprocal Rank Fusion (RRF), a simple,
robust method that needs no score calibration: each document scores
``sum(1 / (k + rank))`` across lists. It's the same idea as rank-averaging
an ensemble of models.

Each answer returns the retrieved contexts too, so Step 5 can evaluate
retrieval quality separately from answer quality.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from aiframe.core import get_embeddings, get_llm, get_prompt

TEXT_SUFFIXES = {".md", ".txt"}


# ---------------------------------------------------------------------- #
# Loading & splitting
# ---------------------------------------------------------------------- #
def load_documents(path: str | Path) -> list[Document]:
    """Load .md/.txt (built in) and .pdf (via langchain-community) from a file or folder."""
    path = Path(path)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    docs: list[Document] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = file.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": file.name}))
        elif suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            for page in PyPDFLoader(str(file)).load():
                page.metadata["source"] = (
                    f"{file.name} p.{page.metadata.get('page', 0) + 1}"
                )
                docs.append(page)
    if not docs:
        raise FileNotFoundError(f"No .md, .txt or .pdf files found at {path}")
    return docs


def split_documents(
    docs: Sequence[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Document]:
    """Split into overlapping chunks. Overlap keeps facts that straddle a boundary intact.

    Chunk size is a tunable hyperparameter: too small loses context, too
    large dilutes relevance. Evaluate a few values on your own data.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# ---------------------------------------------------------------------- #
# Retrieval
# ---------------------------------------------------------------------- #
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]], k: int = 60
) -> list[Document]:
    """Merge ranked lists: score = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    by_id: dict[int, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.metadata["chunk_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            by_id[doc_id] = doc
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


def format_context(docs: Sequence[Document]) -> str:
    """Label each chunk with its source so the model can cite it."""
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[str]
    contexts: list[str] = field(repr=False)


class RAGPipeline:
    """Ingest documents once, then ask questions.

    Example:
        rag = RAGPipeline()
        rag.ingest("data/sample")
        result = rag.ask("How many PTO days do new employees get?")
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        k: int = 4,
        persist_dir: str | Path | None = None,
    ) -> None:
        self.llm = llm or get_llm(temperature=0)
        self.embeddings = embeddings or get_embeddings()
        self.k = k
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.chunks: list[Document] = []
        self._store: VectorStore | None = None
        self._bm25: BM25Okapi | None = None
        self._chain = get_prompt("rag") | self.llm | StrOutputParser()

    # ------------------------------------------------------------------ #
    def ingest(
        self, path: str | Path, chunk_size: int = 800, chunk_overlap: int = 100
    ) -> int:
        """Load, split, embed and index documents. Returns the number of chunks."""
        self.chunks = split_documents(load_documents(path), chunk_size, chunk_overlap)
        if self.persist_dir:
            # Persistent store survives restarts; requires `langchain-chroma`.
            from langchain_chroma import Chroma

            self._store = Chroma.from_documents(
                self.chunks, self.embeddings, persist_directory=str(self.persist_dir)
            )
        else:
            self._store = InMemoryVectorStore.from_documents(
                self.chunks, self.embeddings
            )
        self._bm25 = BM25Okapi([_tokenize(c.page_content) for c in self.chunks])
        return len(self.chunks)

    def _require_index(self) -> tuple[VectorStore, BM25Okapi]:
        if self._store is None or self._bm25 is None:
            raise RuntimeError("Call ingest() before retrieving.")
        return self._store, self._bm25

    def keyword_search(self, query: str, k: int | None = None) -> list[Document]:
        _, bm25 = self._require_index()
        scores = bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in top[: k or self.k] if scores[i] > 0]

    def vector_search(self, query: str, k: int | None = None) -> list[Document]:
        store, _ = self._require_index()
        return store.similarity_search(query, k=k or self.k)

    def retrieve(self, query: str) -> list[Document]:
        """Hybrid retrieval: fuse vector and keyword rankings, keep the top k."""
        fetch = self.k * 2  # over-fetch, then let fusion pick the best
        lists = [self.vector_search(query, fetch), self.keyword_search(query, fetch)]
        # TRY: compare vector_search, keyword_search and retrieve on the same query.
        return reciprocal_rank_fusion(lists)[: self.k]

    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> RAGResult:
        docs = self.retrieve(question)
        answer = self._chain.invoke(
            {"context": format_context(docs), "question": question}
        )
        return RAGResult(
            question=question,
            answer=answer,
            sources=sorted({d.metadata.get("source", "unknown") for d in docs}),
            contexts=[d.page_content for d in docs],
        )

    def stream(self, question: str) -> Iterator[str]:
        docs = self.retrieve(question)
        yield from self._chain.stream(
            {"context": format_context(docs), "question": question}
        )
