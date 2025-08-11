# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course Materials RAG System - A full-stack application for intelligent Q&A about course content using semantic search and Claude AI.

## Common Commands

### Running the Application

**Quick Start (WSL2/Linux):**
```bash
./run.sh
```

**Manual Start:**
```bash
# Install dependencies
uv sync

# Set environment variable
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run server (use --host 0.0.0.0 for WSL2)
cd backend && uv run uvicorn app:app --reload --port 8000 --host 0.0.0.0
```

**Access:**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Development Commands

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (including dev tools)
uv sync --dev

# Run with uv
uv run python main.py

# Start FastAPI server with auto-reload
cd backend && uv run uvicorn app:app --reload
```

### Code Quality Tools

```bash
# Format code automatically
./dev.sh format

# Run all quality checks
./dev.sh check  # or just ./dev.sh

# Run tests
./dev.sh test

# Run everything (checks + tests)
./dev.sh all

# Install pre-commit hooks (one-time setup)
./dev.sh pre-commit
uv run pre-commit run --all-files  # Run hooks manually

# Individual tools
uv run black backend/ --exclude chroma_db
uv run isort backend/ --skip chroma_db
uv run flake8 backend/
uv run mypy backend/
```

**Quality Tools Configured:**
- **Black**: Code formatter (88 char line length)
- **isort**: Import sorter (Black-compatible profile)
- **flake8**: Linter (PEP8 compliance)
- **mypy**: Type checker
- **pre-commit**: Git hooks for automatic quality checks

## Architecture Overview

### RAG Pipeline Flow

1. **Query Entry** → `frontend/script.js` → POST to `/api/query`
2. **Request Handler** → `backend/app.py` → Creates/retrieves session
3. **RAG Orchestration** → `backend/rag_system.py` → Manages entire flow
4. **AI Decision** → `backend/ai_generator.py` → Claude decides: search or direct answer
5. **Tool Execution** (if needed) → `backend/search_tools.py` → Semantic search
6. **Vector Search** → `backend/vector_store.py` → ChromaDB queries
7. **Response Generation** → Claude synthesizes answer with sources
8. **Frontend Display** → Markdown rendering with source attribution

### Key Architectural Patterns

**Tool-Based Search System:**
- AI uses Anthropic function calling to decide when to search
- `ToolManager` orchestrates tool execution
- `CourseSearchTool` performs semantic search with filters
- Sources tracked through tool execution for attribution

**Document Processing:**
- Expected format: Course metadata (3 lines) → Lessons (marked with "Lesson N:")
- Sentence-aware chunking with 800 char chunks, 100 char overlap
- Dual ChromaDB collections: `course_catalog` (metadata) and `course_content` (chunks)
- Course name fuzzy matching via semantic search

**Session Management:**
- Stateful conversations with UUID session IDs
- Max 2 conversation exchanges kept for context
- History passed to AI for coherent responses

### Component Responsibilities

- **`rag_system.py`**: Main orchestrator - coordinates all components
- **`document_processor.py`**: Parses course docs, creates chunks with context
- **`vector_store.py`**: ChromaDB wrapper, handles search and filtering
- **`ai_generator.py`**: Claude API integration, tool execution handling
- **`search_tools.py`**: Abstract tool interface, search implementation
- **`session_manager.py`**: Conversation history and context management
- **`app.py`**: FastAPI endpoints, static serving, CORS configuration

### Configuration

Key settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges

Environment variable required:
- `ANTHROPIC_API_KEY` in `.env` file

## Document Format

Course documents in `docs/` must follow:
```
Course Title: [title]
Course Link: [optional URL]
Course Instructor: [name]

Lesson 0: Introduction
[lesson content...]

Lesson 1: [Title]
Lesson Link: [optional URL]
[lesson content...]
```

## Development Notes

- No test suite exists - consider adding pytest for new features
- Frontend uses vanilla JS with marked.js for markdown
- ChromaDB data persists in `./chroma_db` directory
- Server auto-loads documents from `docs/` on startup
- CORS is wide open for development - tighten for production
- Session data is in-memory only - resets on server restart

## Common Issues

**WSL2 Connection Issues:**
- Use `--host 0.0.0.0` when starting uvicorn
- Access via Windows browser at localhost:8000

**Missing API Key:**
- Create `.env` file from `.env.example`
- Add valid Anthropic API key

**Large Dependencies:**
- Initial `uv sync` downloads ~2GB (PyTorch, CUDA libraries)
- Be patient on first install

**Document Processing:**
- Ensure course docs follow expected format
- Check `backend/document_processor.py` for parsing logic
- Chunks include course/lesson context automatically
- always use uv to run the server do not use pip directly. use uv to run python
- make sure to use uv to manage all dependencies
- never run server for me, I run it myself