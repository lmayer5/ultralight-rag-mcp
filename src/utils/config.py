"""
Configuration loader for the Second Brain system.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "./config/config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "llm": {
            "provider": "ollama",
            "model": "mistral",
            "context_window": 4096,
            "max_tokens": 512,
            "temperature": 0.3
        },
        "rag": {
            "vectordb_path": "./data/vectorstore",
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_k": 3
        },
        "memory": {
            "max_history": 20,
            "persistence": True,
            "db_path": "./data/memory.db"
        }
    }
