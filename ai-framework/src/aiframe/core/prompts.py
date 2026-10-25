"""
Versioned prompt registry.

WHY THIS PATTERN
----------------
A prompt is a model hyperparameter. Changing one wording can move accuracy as
much as changing the model. So prompts get the same treatment as any other
experiment variable:

- Each prompt has a name AND a version (``rag/v1``, ``rag/v2``).
- Old versions stay in the registry, so evaluations are reproducible and
  you can compare v1 vs v2 on the same golden set (Step 5).
- Workflows ask for ``get_prompt("rag")`` (latest) or pin ``version="v1"``.

To add a prompt: append a ``PromptSpec`` to ``_REGISTRY``. To revise one,
add a new version instead of editing the old one.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    description: str
    template: ChatPromptTemplate

    @property
    def key(self) -> str:
        return f"{self.name}/{self.version}"


_REGISTRY: list[PromptSpec] = [
    PromptSpec(
        name="assistant",
        version="v1",
        description="General assistant with optional chat history.",
        template=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful, precise assistant. Be concise. "
                        "If you are unsure, say so instead of guessing."
                    ),
                ),
                MessagesPlaceholder("history", optional=True),
                ("human", "{input}"),
            ]
        ),
    ),
    PromptSpec(
        name="rag",
        version="v1",
        description="Answer strictly from retrieved context, citing sources.",
        template=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Answer the question using ONLY the context below. "
                        "Cite the source of each fact in square brackets, like [source]. "
                        "If the context does not contain the answer, reply exactly: "
                        '"I don\'t know based on the provided documents."\n\n'
                        "Context:\n{context}"
                    ),
                ),
                ("human", "{question}"),
            ]
        ),
    ),
    PromptSpec(
        name="extract",
        version="v1",
        description="Pull structured fields out of unstructured text.",
        template=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Extract the requested information from the text. "
                        "Use only facts stated in the text. Leave a field empty "
                        "rather than inventing a value."
                    ),
                ),
                ("human", "{text}"),
            ]
        ),
    ),
    PromptSpec(
        name="judge",
        version="v1",
        description="LLM-as-judge: grade an answer against a reference (used in evaluation).",
        template=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a strict grader. Compare the candidate answer to the "
                        "reference answer for factual agreement, not wording. "
                        "Score 1 (wrong) to 5 (fully correct and complete)."
                    ),
                ),
                (
                    "human",
                    (
                        "Question: {question}\n\nReference answer: {reference}\n\n"
                        "Candidate answer: {candidate}"
                    ),
                ),
            ]
        ),
    ),
]

_BY_KEY: dict[str, PromptSpec] = {spec.key: spec for spec in _REGISTRY}


def _version_number(spec: PromptSpec) -> int:
    return int(spec.version.lstrip("v"))


def get_prompt(name: str, version: str = "latest") -> ChatPromptTemplate:
    """Return a prompt template by name; ``latest`` picks the highest version."""
    if version != "latest":
        spec = _BY_KEY.get(f"{name}/{version}")
        if spec is None:
            raise KeyError(
                f"No prompt '{name}/{version}'. Available: {', '.join(_BY_KEY)}"
            )
        return spec.template
    candidates = [s for s in _REGISTRY if s.name == name]
    if not candidates:
        raise KeyError(f"No prompt named '{name}'. Available: {', '.join(_BY_KEY)}")
    return max(candidates, key=_version_number).template


def list_prompts() -> list[PromptSpec]:
    """All registered prompts, for discovery and evaluation reports."""
    return list(_REGISTRY)
