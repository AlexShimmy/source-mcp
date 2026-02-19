import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List


class MonitorService:
    """Collects logs and indexing statistics for the dashboard."""

    def __init__(self, max_logs: int = 1000):
        self.logs: deque = deque(maxlen=max_logs)
        self.stats: Dict[str, Any] = {
            "status": "Initializing",
            "files_discovered": 0,
            "files_indexed": 0,
            "files_failed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "index_size_mb": 0.0,
            "current_file": None,
            "indexing_active": False,
            "last_updated": datetime.now().isoformat(),
        }

    # ── Logging ─────────────────────────────────────────────
    def add_log(self, level: str, message: str):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        })

    def get_logs(self) -> List[Dict[str, str]]:
        return list(self.logs)

    # ── Stats ───────────────────────────────────────────────
    def update_stats(self, **kwargs):
        self.stats.update(kwargs)
        self.stats["last_updated"] = datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    # ── Convenience helpers for indexing progress ───────────
    def begin_scan(self, files_discovered: int, skipped: int = 0):
        self.update_stats(
            status="Indexing",
            files_discovered=files_discovered,
            files_indexed=0,
            files_failed=0,
            files_skipped=skipped,
            total_chunks=0,
            indexing_active=True,
            current_file=None,
        )

    def file_started(self, filename: str):
        self.update_stats(current_file=filename)

    def file_indexed(self, chunks: int):
        self.update_stats(
            files_indexed=self.stats["files_indexed"] + 1,
            total_chunks=self.stats["total_chunks"] + chunks,
            current_file=None,
        )

    def file_failed(self):
        self.update_stats(
            files_failed=self.stats["files_failed"] + 1,
            current_file=None,
        )

    def finish_scan(self, index_size_mb: float = 0.0):
        self.update_stats(
            status="Ready",
            indexing_active=False,
            current_file=None,
            index_size_mb=index_size_mb,
        )


monitor = MonitorService()


# ── Logging integration ─────────────────────────────────────
class BufferedHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        monitor.add_log(record.levelname, msg)


def setup_logging():
    log = logging.getLogger("rag-mcp")
    log.setLevel(logging.INFO)

    handler = BufferedHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    log.addHandler(handler)
    log.propagate = False  # prevent MCP stdio pollution
    return log


logger = setup_logging()
