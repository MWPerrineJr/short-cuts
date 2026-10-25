"""Core layer: everything workflows need to talk to a model.

from aiframe.core import get_llm, get_embeddings, get_prompt
"""

from aiframe.core.llm import get_embeddings, get_llm
from aiframe.core.prompts import get_prompt, list_prompts

__all__ = ["get_embeddings", "get_llm", "get_prompt", "list_prompts"]
