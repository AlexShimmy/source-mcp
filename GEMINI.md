# Source-MCP (formerly RAG-MCP) - Gemini Instructions & Guide

This file contains useful instructions and summaries of changes made during our sessions.

## 🚀 How to Run

### Manual Run (Terminal)

Ensure you have `uv` installed.

```bash
# Set required environment variables
export EMBEDDING_PROVIDER="openai" # or "fastembed"
export OPENAI_API_KEY="sk-..."    # required for openai
export ZVEC_PATH="./zvec_db"      # optional override

uv run python -m src.main --path .
```

### Environment Variables (`.env`)

You can create a `.env` file in the root directory:

```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_key
ZVEC_PATH=./zvec_db
```

## 🛠 Features

### 1. OpenAI & FastEmbed Support

- **OpenAI**: Defaults to `text-embedding-3-small` (1536 dims).
- **FastEmbed**: Defaults to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, multilingual).
- **Auto-Migration**: If you change providers/models and the vector dimension changes, the service detects it via `meta.json` in the DB folder and **automatically recreates** the index.

### 2. Incremental Indexing

- Uses `.source-mcp_manifest.json` to store file fingerprints (mtime + size).
- Only indexes new or modified files on startup.

### 3. Web Dashboard (Port 8000)

- **Live Logs**: Includes an **Auto-scroll toggle** (click the pulse indicator).
- **Reindex Base**: A red button to force-wipe the DB and manifest for a fresh full scan.
- **Search Debug**: Special endpoint `/api/search/debug?q=...` to see raw scores.

## 🧪 Testing

```bash
uv run python -m pytest tests/ -v
```

## 📝 Configuration

- `src/config.py`: Contains default settings.
- `mcp_config.json`: Use this to configure Source-MCP as an MCP server for Gemini/Claude.

## Communication Language

- ALL interactions, explanations, and commit messages MUST be in Russian unless explicitly requested otherwise.
