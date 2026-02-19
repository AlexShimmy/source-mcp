from fastapi.testclient import TestClient
from unittest.mock import patch

from src.services.monitor import monitor
from src.web.app import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_stats():
    monitor.update_stats(
        status="Ready",
        files_discovered=10,
        files_indexed=8,
        files_failed=2,
        total_chunks=50,
        index_size_mb=1.5,
    )
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["files_indexed"] == 8
    assert data["files_discovered"] == 10
    assert data["status"] == "Ready"
    assert data["total_chunks"] == 50


def test_get_logs():
    monitor.logs.clear()
    monitor.add_log("INFO", "Test log message")

    response = client.get("/api/logs")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 1
    assert logs[0]["message"] == "Test log message"
    assert logs[0]["level"] == "INFO"


def test_get_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "web_port" in data
    assert "embedding_model" in data
    assert "docs_path" in data
    assert "zvec_path" in data


def test_get_tools():
    response = client.get("/api/tools")
    assert response.status_code == 200
    tools = response.json()
    assert isinstance(tools, list)
    assert len(tools) >= 2
    assert tools[0]["name"] == "search_knowledge_base"


def test_search_empty_query():
    response = client.get("/api/search?q=")
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["error"] == "Empty query"


def test_search_with_query():
    response = client.get("/api/search?q=test")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert data["query"] == "test"
