<div align="center">
  <h1>🔍 Source-MCP (formerly RAG-MCP)</h1>
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
  - **Search Debugging:** Special endpoint (`/api/search/debug?q=...`) to test raw semantic search scores.

## 🚀 Installation & Setup

1. **Prerequisites:** Ensure you have Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) installed.
2. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/source-mcp.git
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
OPENAI_API_KEY=sk-your-openai-api-key

# Optional: Path to store the vector database
ZVEC_PATH=./zvec_db
```

## 🖱️ Usage

### Running Manually (Terminal & Dashboard)

To start the MCP server manually and access the web dashboard:

```bash
uv run python -m src.main --path .
```
- The **MCP protocol** will listen on `stdio`.
- The **Web Dashboard** will be available at [http://localhost:8000](http://localhost:8000).

### Using with an MCP Client (e.g., Claude Desktop)

To use Source-MCP inside Claude Desktop or another MCP-compatible client, update your client's configuration file (e.g. `claude_desktop_config.json`).

```json
{
  "mcpServers": {
    "source-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/source-mcp",
        "run",
        "python",
        "-m",
        "src.main",
        "--path",
        "/absolute/path/to/your/target/project"
      ],
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-your-openai-key"
      }
    }
  }
}
```

## 🧪 Testing

The project uses `pytest` for unit and end-to-end tests. To run the test suite:

```bash
uv run python -m pytest tests/ -v
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
