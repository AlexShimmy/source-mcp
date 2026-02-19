import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import zvec

# We need to ensure we patch settings BEFORE importing IndexerService if it uses settings at module level?
# No, it uses settings inside methods/init.
from src.services.indexer import IndexerService

@pytest.fixture
def mock_settings(tmp_path):
    with patch("src.services.indexer.settings") as mock_settings:
        mock_settings.docs_path = str(tmp_path / "docs")
        mock_settings.zvec_path = str(tmp_path / "zvec_index")
        mock_settings.embedding_model = "test-model"
        mock_settings.embedding_provider = "fastembed"
        mock_settings.openai_api_key = None
        yield mock_settings

@pytest.fixture
def mock_embedding_model():
    with patch("src.services.indexer.TextEmbedding") as MockClass:
        mock_instance = MockClass.return_value
        
        # Mock embed to return a list of vectors
        # BGE-small is 384 dim. We should match that or match whatever we put in schema.
        # Indexer schema has hardcoded 384. 
        # So we MUST return 384-dim vectors.
        def mock_embed(texts):
            for _ in texts:
                yield np.random.rand(384).astype(np.float32)
        
        mock_instance.embed.side_effect = mock_embed
        yield mock_instance

@pytest.fixture
def indexer(mock_settings, mock_embedding_model):
    # Ensure Zvec path is clean
    zvec_path = Path(mock_settings.zvec_path)
    if zvec_path.exists():
        shutil.rmtree(zvec_path)
        
    service = IndexerService()
    service.initialize()
    return service

def test_indexer_initialization(indexer):
    assert indexer.collection is not None
    # Check stats
    stats = indexer.get_stats()
    assert stats["total_vectors"] == 0
    assert stats["backend"] == "zvec"

def test_index_file(indexer, tmp_path, mock_settings):
    # Create a dummy file
    d = Path(mock_settings.docs_path)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "hello.txt"
    p.write_text("Hello world content")

    # Call index_file
    indexer.index_file(str(p))

    # Assertions
    # Stats might be delayed or implementation specific. Querying is more robust.
    # With random vectors (mock), dot product is likely > 0.
    results = indexer.query("hello", threshold=0.0) 
    assert len(results) > 0
    
    # Also we can check manual property if we exposed it, but we didn't.
    # We need to mock embedding for query to match the indexed doc
    # But since we use random vectors in mock, we can't easily match unless we fix the seed or return constant.
    
    # Let's verify by querying with ANY vector and see if we get result (since we only have 1 doc)
    # But relevance threshold might filter it out if random vectors are orthogonal.
    # So we should force mock_embedding to return CONSTANT vector.

def test_query_no_results(indexer):
    # Empty index
    results = indexer.query("search term")
    assert results == []

def test_query_with_results(indexer, tmp_path, mock_settings, mock_embedding_model):
    # Override mock to return constant vector
    constant_vec = np.ones(384, dtype=np.float32)
    # Normalize it effectively? Zvec might expect normalized for cosine?
    constant_vec = constant_vec / np.linalg.norm(constant_vec)
    
    def constant_embed(texts):
        for _ in texts:
            yield constant_vec

    mock_embedding_model.embed.side_effect = constant_embed

    # Index a file
    d = Path(mock_settings.docs_path)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "test.txt"
    p.write_text("Relevant content")
    indexer.index_file(str(p))

    # Query with same vector (via same mock)
    results = indexer.query("query")
    assert len(results) >= 1
    assert "Relevant content" in results[0]

def test_query_threshold_filtering(indexer, tmp_path, mock_settings, mock_embedding_model):
    # Index a file
    d = Path(mock_settings.docs_path)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "test.txt"
    p.write_text("Content")
    
    # Vector A
    vec_a = np.zeros(384, dtype=np.float32)
    vec_a[0] = 1.0 # [1, 0, ...]
    
    def embed_a(texts):
        for _ in texts:
            yield vec_a
            
    mock_embedding_model.embed.side_effect = embed_a
    indexer.index_file(str(p))

    # Query with Vector B (Orthogonal)
    vec_b = np.zeros(384, dtype=np.float32)
    vec_b[1] = 1.0 # [0, 1, ...]
    # Dot product = 0.0
    
    def embed_b(texts):
        for _ in texts:
            yield vec_b
            
    mock_embedding_model.embed.side_effect = embed_b

    # Threshold 0.5 should filter it out
    results = indexer.query("orthogonal", threshold=0.5)
    assert len(results) == 0
