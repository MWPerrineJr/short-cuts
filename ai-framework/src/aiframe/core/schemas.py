"""
Pydantic schemas for structured LLM output.

WHY THIS PATTERN
----------------
Free-text answers are hard to test, store, or feed into downstream code.
``llm.with_structured_output(Schema)`` makes the model return a validated
Pydantic object instead, which gives you:

- Type guarantees: ``confidence`` is a float in [0, 1] or the call fails loudly.
- Field descriptions double as instructions to the model, so write them well.
- Outputs that drop straight into a DataFrame: ``pd.DataFrame([r.model_dump() ...])``.

These schemas are reused by the workflows (Step 3) and evaluation (Step 5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class Answer(BaseModel):
    """A question answer with self-reported confidence and sources."""

    answer: str = Field(description="The direct answer to the question.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence from 0 (guessing) to 1 (certain)."
    )
    sources: list[str] = Field(
        default_factory=list, description="Documents or facts the answer relies on."
    )


class Contact(BaseModel):
    """One person mentioned in a piece of text."""

    name: str = Field(description="Full name as written in the text.")
    email: EmailStr | None = Field(
        default=None, description="Email address, if stated."
    )
    company: str | None = Field(default=None, description="Organization, if stated.")
    role: str | None = Field(default=None, description="Job title or role, if stated.")


class ContactList(BaseModel):
    """All contacts found in a document."""

    contacts: list[Contact] = Field(default_factory=list)


class Sentiment(BaseModel):
    """Sentiment classification, used by the batch-processing workflow."""

    label: Literal["positive", "negative", "neutral"]
    score: float = Field(
        ge=-1.0, le=1.0, description="-1 very negative to +1 very positive."
    )
    reason: str = Field(description="One short sentence explaining the label.")


class JudgeScore(BaseModel):
    """LLM-as-judge grade, used by the evaluation harness."""

    score: int = Field(
        ge=1, le=5, description="1 = wrong, 5 = fully correct and complete."
    )
    reasoning: str = Field(description="Brief justification for the score.")

    @property
    def passed(self) -> bool:
        return self.score >= 4
