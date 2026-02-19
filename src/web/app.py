from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

from ..config import settings
from ..services.monitor import monitor
from ..services.indexer import indexer

app = FastAPI(title="RMCP Dashboard")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# ── Version (read once from pyproject.toml at startup) ──────
_VERSION = "dev"
try:
    import importlib.metadata
    _VERSION = importlib.metadata.version("rag-mcp")
except Exception:
    pass


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = TEMPLATES_DIR / "index.html"
    try:
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse(
            content=(
                f"<html><body><h1>RMCP Dashboard</h1>"
                f"<p>Template not found: <code>{index_path}</code></p>"
                f"<p><a href='/api/stats'>API Stats</a></p>"
                f"</body></html>"
            ),
            status_code=200,
        )


@app.get("/api/stats")
async def get_stats():
    return monitor.get_stats()


@app.get("/api/logs")
async def get_logs():
    return monitor.get_logs()


@app.get("/api/config")
async def get_config():
    return {
        "version": _VERSION,
        "web_port": settings.web_port,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "docs_path": str(Path(settings.docs_path).resolve()),
        "zvec_path": str(Path(settings.zvec_path).resolve()),
    }


@app.post("/api/reindex")
async def reindex_knowledge_base():
    """Wipe DB and force a full re-index."""
    try:
        indexer.reindex()
        return {"status": "success", "message": "Reindexing started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/tools")
async def get_tools():
    return [
        {
            "name": "search_knowledge_base",
            "description": "Search for relevant context in the indexed documents.",
        },
        {
            "name": "get_index_stats",
            "description": "Get current statistics about the vector index.",
        },
    ]


@app.get("/api/search")
async def search(q: str = "", limit: int = 5):
    """Quick search endpoint for dashboard testing."""
    if not q.strip():
        return {"query": q, "results": [], "error": "Empty query"}
    try:
        results = indexer.query(q, limit)
        return {"query": q, "results": results}
    except Exception as exc:
        return {"query": q, "results": [], "error": str(exc)}


@app.get("/api/search/debug")
async def search_debug(q: str = "", limit: int = 10):
    """Debug search - shows raw scores."""
    if not q.strip():
        return {"query": q, "results": []}
    try:
        import zvec as _zvec
        vecs = indexer.embed([q])
        if not vecs:
            return {"query": q, "results": [], "error": "No embedding"}
        qvec = vecs[0]
        results = indexer.collection.query(
            vectors=[_zvec.VectorQuery(field_name="embedding", vector=qvec)],
            topk=limit,
        )
        return {
            "query": q,
            "total": len(results),
            "results": [
                {
                    "score": r.score,
                    "file": r.fields.get("file_path", ""),
                    "text": r.fields.get("text", "")[:120],
                }
                for r in results
            ],
        }
    except Exception as exc:
        return {"query": q, "results": [], "error": str(exc)}
