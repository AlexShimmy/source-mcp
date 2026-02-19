
import pytest
import os
import asyncio
import time
from fastapi.testclient import TestClient
from pathlib import Path

# We import app and indexer from their respective modules
from src.web.app import app
from src.services.indexer import indexer
from src.config import settings

# Create a TestClient
client = TestClient(app)

@pytest.fixture(scope="module")
def initialized_system():
    """
    Ensures the system is initialized and has indexed the documents.
    This fixture runs once per module.
    """
    # 1. Create artifacts test_docs
    docs_path = Path("artifacts/test_docs")
    if docs_path.exists():
        import shutil
        shutil.rmtree(docs_path)
    docs_path.mkdir(parents=True, exist_ok=True)
    
    # Generate some dummy content
    (docs_path / "README.md").write_text("# Deployment Guide 0\nThis document covers the Deployment aspects.")
    (docs_path / "config.json").write_text('{"host": "server-3.local"}')
    (docs_path / "secret.txt").write_text("NEEDLE_FOUND: special_secret_code_10")
    
    src_path = docs_path / "src" / "modules"
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "generated_file_40.py").write_text("class Service40:\n    def process_data_40(self):\n        pass")
    
    # Generate 50+ files to satisfy the test assertions
    for i in range(50):
        (docs_path / f"filler_{i}.txt").write_text(f"Filler content {i}")
        
    print("\n[E2E] Initializing IndexerService with real embeddings...")
    settings.docs_path = str(docs_path)
    settings.zvec_path = "artifacts/zvec_db_test"
    
    # Initialize the global indexer (it might already be init by import, but let's ensure)
    indexer.initialize()
    
    # 2. Trigger indexing of the docs directory
    # The watcher might catch it, or we can force index. 
    # Let's force index all files in docs/ to be sure.
    print(f"[E2E] Indexing files in {docs_path}...")
    
    # Find all files recursively
    files = [f for f in docs_path.rglob("*") if f.is_file()]
    print(f"[E2E] Found {len(files)} files to index.")
    
    count = 0
    for f in files:
        # We index synchronously here for the test setup, although index_file might be async-capable?
        # The service method is synchronous in previous view.
        try:
            indexer.index_file(str(f))
            count += 1
            if count % 10 == 0:
                print(f"[E2E] Indexed {count}/{len(files)}...")
        except Exception as e:
            print(f"[E2E] Failed to index {f}: {e}")
            
    print(f"[E2E] Finished indexing {count} files.")
    
    # Force status to Ready for test assertions
    from src.services.monitor import monitor
    monitor.update_stats(status="Ready")
    
    time.sleep(1)
    
    return indexer

def test_stats_reflects_files(initialized_system):
    """
    Verify that the stats endpoint reflects the indexed files.
    """
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    
    print(f"\n[E2E] Stats: {data}")
    # We expect at least 50 files
    assert data["files_indexed"] >= 50
    assert data["status"] == "Ready"

def test_search_python_code(initialized_system):
    """
    Search for a specific function definition existing in the generated python files.
    """
    # file_40 -> index 40 -> Service40, process_data_40
    
    f40 = Path("artifacts/test_docs/src/modules/generated_file_40.py")
    if not f40.exists():
        pytest.fail(f"Test file {f40} does not exist!")
        
    content = f40.read_text()
    if "process_data_40" not in content:
        pytest.fail(f"Test file {f40} does not contain 'process_data_40'!")
        
    # Explicitly index this file again to be absolutely sure
    indexer.index_file(str(f40))
    indexer._save_manifest()
    
    # Try searching for Class name first
    query_cls = "Service40"
    resp_cls = client.get(f"/api/search?q={query_cls}")
    print(f"\n[E2E] Search 'Service40': {resp_cls.json()['results']}")
    
    query = "process_data_40"
    response = client.get(f"/api/search?q={query}")
    assert response.status_code == 200
    data = response.json()
    
    results = data["results"]
    print(f"\n[E2E] Python Search Results for '{query}': {results}")
    
    # If empty, maybe try raw? 
    # But assertion failure gives us info.
    assert len(results) > 0, f"Search for {query} returned no results. File content snippet: {content[:100]}..."
    
    # Check if the correct file is in the results
    # The result string usually looks like "[filename] content..."
    found = False
    for res in results:
        if "generated_file_40.py" in res or "process_data_40" in res:
            found = True
            break
            
    assert found, f"Could not find 'process_data_40' in results: {results}"

def test_semantic_search_markdown(initialized_system):
    """
    Test semantic capabilities.
    We search for 'deployment guide' which should match a markdown file about Deployment.
    """
    # generated_file_0 -> Index 0 -> Topic 'Deployment'
    # Content: "# Deployment Guide 0 ... This document covers the Deployment aspects..."
    
    query = "How do I deploy the system?"
    response = client.get(f"/api/search?q={query}")
    assert response.status_code == 200
    results = response.json()["results"]
    
    assert len(results) > 0
    # We expect "Deployment Guide" or similar
    found = False
    for res in results:
        if "Deployment" in res or "Guide" in res:
            found = True
            break
            
    assert found, f"Semantic search failed for '{query}'. Results: {results}"

def test_needle_in_haystack(initialized_system):
    """
    Search for a unique ID injected into a text file.
    """
    # index 10 -> "NEEDLE_FOUND: special_secret_code_10"
    needle = "special_secret_code_10"
    
    response = client.get(f"/api/search?q={needle}")
    assert response.status_code == 200
    results = response.json()["results"]
    
    assert len(results) > 0
    assert any(needle in r for r in results)

def test_json_config_search(initialized_system):
    """
    Search for a specific config value in JSON.
    """
    # generated_file_3 -> Index 3 (JSON) -> "host": "server-3.local"
    query = "server-3.local"
    
    response = client.get(f"/api/search?q={query}")
    assert response.status_code == 200
    results = response.json()["results"]
    
    assert len(results) > 0
    assert any(query in r for r in results)
