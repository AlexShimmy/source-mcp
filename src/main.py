import sys
import os
import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .config import settings
from .services.indexer import indexer
from .services.monitor import logger, monitor
from .web.app import app as web_app

# ── MCP Server ──────────────────────────────────────────────
mcp = FastMCP("Source-MCP Local RAG Server")


@mcp.tool()
async def search_knowledge_base(query: str, limit: int = 5) -> str:
    """
    Search for relevant context in the indexed documents.

    Args:
        query: The question or topic to search for.
        limit: Maximum number of text chunks to return.
    """
    logger.info(f"Received search query: {query}")
    results = indexer.query(query, limit)

    if not results:
        return "No relevant information found in the local knowledge base."

    formatted_results = "\n\n---\n\n".join(results)
    return f"Found {len(results)} relevant chunks:\n\n{formatted_results}"


@mcp.tool()
async def get_index_stats() -> str:
    """Get current statistics about the vector index."""
    stats = indexer.get_stats()
    return str(stats)


# ── Background services ────────────────────────────────────
def start_background_services():
    logger.info("Starting background services...")
    indexer.start_watching()
    threading.Thread(target=indexer.index_directory, daemon=True).start()


def run_dashboard():
    """Run the FastAPI dashboard in a separate thread."""
    logger.info(f"Starting Dashboard at http://{settings.host}:{settings.web_port}")
    config = uvicorn.Config(
        web_app,
        host=settings.host,
        port=settings.web_port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    server.run()


# ── CLI entry-point ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Source-MCP Server")
    parser.add_argument("--path", type=str, help="Path to the documents directory")
    parser.add_argument("--embed-model", type=str, help="HuggingFace embedding model name")
    parser.add_argument("--web-port", type=int, help="Port for the Web Dashboard")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    args, _unknown = parser.parse_known_args()

    # Determine project root
    if args.path:
        project_path = Path(args.path).resolve()
    elif os.getenv("SOURCE_MCP_INDEX_DIR"):
        project_path = Path(os.getenv("SOURCE_MCP_INDEX_DIR")).resolve()
    else:
        project_path = Path(".").resolve()

    # Load environment from project root (override existing env vars)
    env_path = project_path / ".env"
    if env_path.exists():
        logger.info(f"Loading environment from {env_path}")
        load_dotenv(env_path, override=True)
    
    # Update settings from environment (pydantic settings are already init, so we update manually)
    settings.docs_path = str(project_path)
    
    # Use project-local storage for index
    sourcemcp_dir = project_path / ".source-mcp"
    settings.zvec_path = str(sourcemcp_dir / "zvec_db")
    
    # Update config from env vars if present
    if os.getenv("EMBEDDING_PROVIDER"):
        settings.embedding_provider = os.getenv("EMBEDDING_PROVIDER")
    if os.getenv("EMBEDDING_MODEL"):
        settings.embedding_model = os.getenv("EMBEDDING_MODEL")
    if os.getenv("OPENAI_API_KEY"):
        settings.openai_api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("WEB_PORT"):
        settings.web_port = int(os.getenv("WEB_PORT"))

    # CLI overrides env
    if args.embed_model:
        settings.embedding_model = args.embed_model
    if args.web_port:
        settings.web_port = args.web_port

    logger.info(f"Project path: {settings.docs_path}")
    logger.info(f"Index path: {settings.zvec_path}")

    # Ensure directories exist
    Path(settings.docs_path).mkdir(parents=True, exist_ok=True)
    Path(settings.zvec_path).mkdir(parents=True, exist_ok=True)

    # Initialize indexer (delayed to avoid side-effects on import)
    try:
        indexer.configure()
        indexer.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        sys.exit(1)

    # Start dashboard
    threading.Thread(target=run_dashboard, daemon=True).start()

    # Auto-open browser (with delay for server startup)
    if not args.no_browser:
        def open_browser():
            import time
            time.sleep(1.5)
            url = f"http://{settings.host}:{settings.web_port}"
            logger.info(f"Opening dashboard in browser: {url}")
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=open_browser, daemon=True).start()

    # Start indexer & watcher
    start_background_services()

    # Run MCP server (blocks)
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopping...")
    except Exception as e:
        logger.error(f"MCP Server error: {e}")
        sys.exit(1)
    finally:
        logger.info("Stopping indexer watcher...")
        indexer.stop_watching()


if __name__ == "__main__":
    main()
