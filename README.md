<div align="center">
  <h1>🔍 Source-MCP</h1>
  <p><strong>A Model Context Protocol (MCP) server for semantic search and Retrieval-Augmented Generation (RAG) over local codebases and documents.</strong></p>
</div>

---

## 📖 Overview

**Source-MCP** leverages the [Model Context Protocol](https://github.com/anthropic/modelcontextprotocol) to provide AI assistants (like Claude, Gemini, and others) with direct access to local files through semantic search.

Instead of manually copy-pasting code or documentation into your prompts, Source-MCP automatically indexes your local repository, generates vector embeddings, and enables the AI to semantically search and retrieve only the most relevant files.

## ✨ Key Features

- **Dual Embedding Support:**
  - **OpenAI:** Uses robust `text-embedding-3-small` (1536 dimensions) for high-quality enterprise embeddings.
  - **FastEmbed (Local):** Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims). Runs entirely locally, no API keys required, and supports multilingual inquiries.
- **Smart Incremental Indexing:** Uses file fingerprints (modified time + size) to only index new or modified files, ensuring lightning-fast startup times.
- **Auto-Migration:** Automatically detects embedding dimension changes (e.g., switching from OpenAI to FastEmbed) and safely recreates the vector index.
- **Web Dashboard (Port 8000):**
  - **Live Logs:** View real-time indexing and search activity with auto-scroll.
  - **Reindex Base:** Force-wipe the vector DB and manifest for a completely fresh full scan.
  - **Reindex Base:** Force-wipe the vector DB and manifest for a completely fresh full scan.
  - **Search Debugging:** Special endpoint (`/api/search/debug?q=...`) to test raw semantic search scores.

## 🤔 Why local embeddings and `zvec`?

We use [**zvec**](https://github.com/alibaba/zvec), a lightweight, high-performance vector database maintained by Alibaba. `zvec` is embedded directly into the Python process, eliminating the need to set up or run external vector servers (like Pinecone, Milvus, or Qdrant). Combined with `FastEmbed`, this allows Source-MCP to build the entire semantic search pipeline **fully offline**, quickly, and entirely on your local machine.

## 🚀 Installation & Setup

1. **Prerequisites:** Ensure you have Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) installed.
2. **Clone the repository:**

   ```bash
   git clone https://github.com/AlexShimmy/source-mcp.git
   cd source-mcp
   ```

3. **Install Dependencies:**

   ```bash
   # uv will automatically handle virtual environment creation and dependencies
   uv sync
   ```

## ⚙️ Configuration

Create a `.env` file in the root directory (you can copy `.env.example` if available).

```bash
# Choose your provider: "openai" or "fastembed"
EMBEDDING_PROVIDER=openai

# Required ONLY if using OpenAI
# Required ONLY if using OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# Optional: Path to store the vector database (Defaults to `.source-mcp/zvec_db` in the index dir)
ZVEC_PATH=./zvec_db

# Optional: Which directory to index (Defaults to current directory)
SOURCE_MCP_INDEX_DIR=/path/to/your/project

# Optional: Port for the Web Dashboard (Defaults to 8000)
WEB_PORT=8000
```

## 🖱️ Usage

### Running Manually (Terminal & Dashboard)

To start the MCP server manually and access the web dashboard:

```bash
uv run python -m src.main --path .
```

- The **MCP protocol** will listen on `stdio`.
- The **Web Dashboard** will be available at [http://localhost:8000](http://localhost:8000).

```json
{
  "mcpServers": {
    "source-mcp": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/source-mcp", "run", "python", "-m", "src.main"]
    }
  }
}
```

All other settings (such as `SOURCE_MCP_INDEX_DIR`, `EMBEDDING_PROVIDER` or `OPENAI_API_KEY`) should be configured via the `.env` file in the root directory of Source-MCP.

### 💻 Using with Cursor IDE

Add the following MCP server in `Cursor Settings` > `Features` > `MCP` > `+ Add new MCP server`:

- **Name:** `Source-MCP`
- **Type:** `command`
- **Command:** `uv --directory /absolute/path/to/source-mcp run python -m src.main`

### 💻 Using with VS Code (Roo Code / Cline)

Add the following entry to your extension settings (`cline_mcp_settings.json`), and configure any remaining parameters within your project's `.env` file:

```json
{
  "mcpServers": {
    "source-mcp": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/source-mcp", "run", "python", "-m", "src.main"]
    }
  }
}
```

## 🧪 Testing

The project uses `pytest` for unit and end-to-end tests. To run the test suite:

```bash
uv run python -m pytest tests/ -v
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
