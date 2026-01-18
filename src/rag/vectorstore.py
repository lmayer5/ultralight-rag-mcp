"""
FAISS Vector Store management for RAG pipeline.
Handles document ingestion, chunking, and similarity search.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .embeddings import get_embeddings


class VectorStoreManager:
    """Manages FAISS vector store for document retrieval."""
    
    def __init__(
        self,
        persist_directory: str = "./data/vectorstore",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.persist_directory = Path(persist_directory)
        self.embeddings = get_embeddings(embeddings_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore: Optional[FAISS] = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_or_create(self) -> FAISS:
        """Load existing vector store or create empty one."""
        index_path = self.persist_directory / "index.faiss"
        
        if index_path.exists():
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = FAISS.load_local(
                str(self.persist_directory),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded vector store with {self.vectorstore.index.ntotal} vectors")
        else:
            print("Creating new vector store...")
            # Create with a placeholder document (FAISS requires at least one)
            placeholder = Document(
                page_content="This is the Second Brain knowledge base.",
                metadata={"source": "system", "type": "placeholder"}
            )
            self.vectorstore = FAISS.from_documents([placeholder], self.embeddings)
            self.save()
            print("Created new vector store")
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store with chunking.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Number of chunks added
        """
        if not self.vectorstore:
            self.load_or_create()
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Add to vector store
        self.vectorstore.add_documents(chunks)
        self.save()
        
        return len(chunks)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> int:
        """
        Add raw texts to the vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            
        Returns:
            Number of chunks added
        """
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        return self.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            self.load_or_create()
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def save(self):
        """Persist vector store to disk."""
        if self.vectorstore:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.persist_directory))
            print(f"Saved vector store to {self.persist_directory}")
    
    def get_retriever(self, k: int = 3):
        """Get a retriever for use with LangChain chains."""
        if not self.vectorstore:
            self.load_or_create()
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
