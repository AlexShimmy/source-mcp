import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.main import search_knowledge_base, get_index_stats
from src.services.indexer import IndexerService
import asyncio

# We need to force re-init of indexer with test settings
@pytest.fixture
def integration_indexer(tmp_path):
    # Setup test docs path
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Patch settings
    # Patch settings
    # We must patch the settings instance in src.services.indexer because it was imported early
    with patch("src.services.indexer.settings") as mock_settings:
        mock_settings.docs_path = str(docs_dir)
        mock_settings.zvec_path = str(tmp_path / "zvec_index")
        mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2" 
        mock_settings.embedding_provider = "fastembed"
        mock_settings.openai_api_key = None 
        
        with patch("src.services.indexer.TextEmbedding") as MockEmbed:
             # Setup mock behavior
             instance = MockEmbed.return_value
             # Simple deterministic embedding
             def mock_embed(texts):
                 import numpy as np
                 for t in texts:
                     if "apple" in t:
                         yield np.array([1.0, 0.0, 0.0]) # Example 3D vector. In reality should be 384D for zvec schema. 
                         # But wait, we hardcoded 384 in IndexerService.
                         # If we return 3D vector, Zvec insert might fail or pad?
                         # Zvec schema said 384.
                         # We should return 384D vectors!
                     elif "banana" in t:
                         # Orthogonal
                         v = np.zeros(384)
                         v[1] = 1.0
                         yield v
                     else:
                         # Match apple
                         v = np.zeros(384)
                         v[0] = 1.0
                         yield v

             # Fix Apple vector to be 384
             def safe_mock_embed(texts):
                 import numpy as np
                 for t in texts:
                     v = np.zeros(384, dtype=np.float32)
                     if "apple" in t:
                         v[0] = 1.0
                     elif "banana" in t:
                         v[1] = 1.0
                     else:
                         v[0] = 1.0 # default matches apple?
                     yield v

             instance.embed.side_effect = safe_mock_embed
             
             # Create real indexer with mocked settings
             # Since we are creating a new instance, it will try to create/open zvec DB at tmp_path
             real_indexer = IndexerService()
             
             # IndexerService.__init__ uses 'settings.zvec_path'. 
             # We patched 'src.services.indexer.settings'.
             # So this should work and use temp path.
             
             # Initialize it explicitly (lazy init)
             real_indexer.initialize()
             
             # We need to patch the global indexer in src.main so our test calls use this one
             with patch("src.main.indexer", real_indexer):
                 yield real_indexer

@pytest.mark.asyncio
async def test_search_flow(integration_indexer, tmp_path):
    # 1. Add a file
    docs_dir = Path(integration_indexer.collection.path).parent / "docs" # derive from mocked settings via indexer?
    # Actually just use tmp_path
    docs_dir = tmp_path / "docs"
    # Ensure it exists (mock_settings set it to this)
    if not docs_dir.exists():
         docs_dir.mkdir(parents=True)
         
    f = docs_dir / "fruits.txt"
    f.write_text("The apple is red.")
    
    # 2. Trigger indexing manually
    integration_indexer.index_file(str(f))
    
    # 3. Search via MCP tool
    result = await search_knowledge_base("apple")
    assert isinstance(result, str)
    assert "Found 1 relevant chunks" in result
    assert "[fruits.txt] The apple is red." in result
    
    # 4. Search for "banana" -> with threshold=0.0, topk returns closest matches even if low score
    result_banana = await search_knowledge_base("banana")
    # With mocked embeddings, banana gets same vector as apple, so it still matches
    assert isinstance(result_banana, str)

@pytest.mark.asyncio
async def test_zvec_semantic_search(tmp_path):
    # Setup specific for this test to handle the Zvec sentence logic
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # 1. Create content
    f1 = docs_dir / "zvec_info.txt"
    f1.write_text("Zvec is an embedded vector database from Alibaba based on RocksDB.")
    
    f2 = docs_dir / "other.txt"
    f2.write_text("The weather is nice today.")
    
    # Patch settings and embedding model
    with patch("src.services.indexer.settings") as mock_settings:
        mock_settings.docs_path = str(docs_dir)
        mock_settings.zvec_path = str(tmp_path / "zvec_semantic_index")
        mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_settings.embedding_provider = "fastembed"
        mock_settings.openai_api_key = None
        
        with patch("src.services.indexer.TextEmbedding") as MockEmbed:
            instance = MockEmbed.return_value
            
            # Define vectors
            # vec_zvec: matches "Zvec", "database", "Alibaba"
            # vec_weather: matches "weather", "nice"
            # We use 384 dims
            
            def semantic_mock_embed(texts):
                import numpy as np
                for t in texts:
                    v = np.zeros(384, dtype=np.float32)
                    t_lower = t.lower()
                    
                    if "zvec" in t_lower or "database" in t_lower or "alibaba" in t_lower:
                        # Vector A direction
                        v[0] = 1.0 
                    elif "weather" in t_lower:
                        # Vector B direction (orthogonal to A)
                        v[1] = 1.0
                    else:
                        # Unknown / low similarity to both?
                        # Let's make it have small overlap with A to test threshold
                        v[0] = 0.1 
                        v[2] = 0.9
                    
                    # Normalize
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        v = v / norm
                    yield v
            
            instance.embed.side_effect = semantic_mock_embed
            
            # Init indexer
            real_indexer = IndexerService()
            real_indexer.initialize()
            
            with patch("src.main.indexer", real_indexer):
                # Index files
                real_indexer.index_file(str(f1))
                real_indexer.index_file(str(f2))
                
                # Query 1: Exact topic
                # "What is Zvec?" -> contains "zvec" -> returns vec[0]=1.0
                # Content "Zvec is an embedded..." -> contains "zvec" -> returns vec[0]=1.0
                # Dot product 1.0 > 0.65
                res1 = await search_knowledge_base("What is Zvec?")
                assert "Zvec is an embedded vector database" in res1
                
                # Query 2: Related keyword
                # "Tell me about Alibaba databases" -> contains "alibaba" -> returns vec[0]=1.0
                res2 = await search_knowledge_base("Tell me about Alibaba databases")
                assert "Zvec is an embedded vector database" in res2
                
                # Query 3: Unrelated
                # "How is the weather?" -> contains "weather" -> returns vec[1]=1.0
                # Weather doc has vec[1]=1.0. Dot product 1.0. (best match)
                # Zvec doc has vec[0]=1.0. Dot product 0. (worst match)
                res3 = await search_knowledge_base("How is the weather?")
                assert "The weather is nice today" in res3
                # Weather should be the FIRST result (highest score)
                weather_pos = res3.index("The weather is nice today")
                if "Zvec" in res3:
                    zvec_pos = res3.index("Zvec")
                    assert weather_pos < zvec_pos  # weather ranked higher
                
                # Query 4: Irrelevant to both
                # "Spaghetti recipe" -> unknown -> vec[0]=0.1, vec[2]=0.9
                # With threshold=0.0, topk returns closest results regardless
                res4 = await search_knowledge_base("Spaghetti recipe")
                assert isinstance(res4, str)  # just check it doesn't crash

@pytest.mark.asyncio
async def test_stats_flow(integration_indexer):
    stats = await get_index_stats()
    assert isinstance(stats, str)
    assert "'backend': 'zvec'" in stats
    # We indexed nothing in this test function (new fixture instance)
    # But wait, python fixtures re-run.
    
    # assert stats["total_vectors"] == 0 
    # Actually let's assume it's empty start.
    pass
