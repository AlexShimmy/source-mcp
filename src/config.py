from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docs_path: str = os.getenv("SOURCE_MCP_INDEX_DIR", ".")
    zvec_path: str = "./zvec_db"
    
    # Embedding settings
    embedding_provider: str = "fastembed"  # "fastembed" or "openai"
    # FastEmbed model (384 dims)
    # embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # OpenAI model (1536 dims for text-embedding-3-small)
    embedding_model: str | None = None
    openai_api_key: str | None = None

    # Web Dashboard settings
    web_port: int = 8000
    host: str = "127.0.0.1"


settings = Settings()
