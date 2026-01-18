"""
Embedding model initialization for RAG pipeline.
Uses sentence-transformers MiniLM-L6-v2 (~80MB, runs on CPU).
"""

from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initialize and return the embedding model.
    Uses caching to avoid reloading the model on each call.
    
    Args:
        model_name: HuggingFace model name for embeddings
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use CPU for embeddings
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded successfully")
    return embeddings
