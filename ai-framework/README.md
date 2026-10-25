# AI Framework

A reusable, provider-agnostic scaffold for common AI engineering workflows:
**chat, RAG, agents, structured extraction, and async batch processing**, with
Gradio, Streamlit, and FastAPI front ends plus evaluation built in.

Stack: Python 3.11+ · LangChain / LangGraph · Pydantic · Gradio · Streamlit · FastAPI · VS Code

---

## Architecture

Every project follows the same lifecycle. Each layer lives in its own module, so a
new project usually means changing only the **workflows** layer.

```
 Config ──► Core ──► Workflows ──► Interfaces ──► Evaluation ──► Observability
 (.env,     (LLM     (chat, RAG,    (Gradio,       (test sets,    (LangSmith
 settings)  factory, agents,        Streamlit,     metrics,       tracing)
            prompts, extraction,    FastAPI,       regression)
            schemas) batch)         notebooks)
```

**Design principles**

1. **One source of config.** All settings come from `.env` through a typed
   Pydantic object (`config/settings.py`). There are no scattered `os.environ` calls.
2. **Provider-agnostic.** Workflows ask the core factory for a model. Switching
   OpenAI → Anthropic → Ollama is a `.env` change, not a code change.
3. **Fail fast.** Missing or invalid config raises a clear `ConfigError` at
   startup, not a cryptic error mid-request.
4. **Evaluate like a data scientist.** LLM outputs are predictions: score them
   against a held-out golden set and track regressions across prompt versions.

---

## Quickstart

Run everything from this `ai-framework` folder.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the stack (shared root requirements + dev extras)
pip install -r requirements-dev.txt
#    faster alternative: pip install uv && uv pip install -r requirements-dev.txt

# 3. Configure
cp .env.example .env               # Windows: copy .env.example .env
#    then set LLM_PROVIDER, LLM_MODEL, and that provider's API key

# 4. Verify
python examples/01_check_setup.py
```

### VS Code

Open **this folder** (`ai-framework/`) as the workspace so `.vscode/` settings apply.

- Select the interpreter with *Python: Select Interpreter* → `.venv`.
  On Windows, if it isn't detected, choose `.venv\Scripts\python.exe`.
- Recommended extensions: **Python**, **Pylance**, **Jupyter**, **Ruff**.
- The Run and Debug panel has ready-made configs for every example and app.
- Ruff formats and organizes imports on save.

---

## Project structure

| Path | Purpose | Status |
|---|---|---|
| `config/settings.py` | Typed settings, provider validation, tracing setup | Step 1 ✅ |
| `src/aiframe/core/` | LLM + embeddings factory, prompts, Pydantic schemas | Step 2 |
| `src/aiframe/workflows/` | chat, RAG, agent, extraction, batch | Step 3 |
| `src/aiframe/tools/` | Sample tools for agents | Step 3 |
| `apps/` | Gradio, Streamlit, FastAPI front ends | Step 4 |
| `src/aiframe/eval/` | Golden-set evaluation, LLM-as-judge | Step 5 |
| `tests/` | Unit tests with mocked LLMs (no API cost) | Step 5 |
| `notebooks/` | Quickstart notebook using `IPython.display` | Step 5 |
| `examples/` | Runnable, numbered walkthrough of every layer | ongoing |

---

## Examples

Each example runs standalone, explains **why** the pattern matters in its docstring,
and has `# TRY:` comments suggesting experiments.

| # | Script | Concept |
|---|---|---|
| 01 | `examples/01_check_setup.py` | Fail-fast readiness check: packages, `.env`, provider keys |
| 02 | `examples/02_llm_factory.py` | One prompt across providers; structured output *(Step 2)* |
| 03 | `examples/03_chat.py` | Chat with memory and streaming *(Step 3)* |
| 04 | `examples/04_rag.py` | Ingest → split → embed → retrieve → answer *(Step 3)* |
| 05 | `examples/05_agent.py` | LangGraph tool-calling agent *(Step 3)* |
| 06 | `examples/06_extract.py` | Text → validated Pydantic objects *(Step 3)* |
| 07 | `examples/07_batch.py` | Async concurrency with rate limiting *(Step 3)* |
| 08 | `examples/08_evaluate.py` | Scoring against a golden set *(Step 5)* |

---

## Configuration reference

All values are set in `.env` (see `.env.example`). Blank values use defaults.

| Variable | Required | Notes |
|---|---|---|
| `LLM_PROVIDER` | yes | `openai`, `anthropic`, `google_genai`, `ollama` |
| `LLM_MODEL` | yes | Model name from the provider's docs |
| `LLM_TEMPERATURE` | no | `0.0`–`2.0`, default `0.2` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | for RAG | `openai`, `google_genai`, `ollama` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | per provider | Only for providers you use |
| `OLLAMA_BASE_URL` | no | Default `http://localhost:11434` |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | no | Tracing and debugging |
| `DATA_DIR`, `VECTORSTORE_DIR` | no | Relative to this folder |
| `MAX_CONCURRENCY` | no | Parallel LLM calls in batch workflows, default `5` |

Using settings in code:

```python
from config.settings import get_settings

settings = get_settings()
provider, model = settings.require_llm()    # raises ConfigError with a fix-it message
settings.apply_tracing()                     # enables LangSmith if configured
```

---

## Security notes

- `.env` is git-ignored; only `.env.example` (no real values) is committed.
- API keys are `SecretStr` and print as `**********`.
- The setup check reports only whether a key is set, never any characters of it.
