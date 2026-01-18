"""
Document Ingestion Pipeline - Loads and processes documents for the knowledge base.

Supports markdown, text, and PDF files with chunking and metadata extraction.
"""

import os
from pathlib import Path
from typing import List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    file_path: str
    chunks_added: int
    success: bool
    error: Optional[str] = None


class DocumentIngester:
    """Ingests documents from various formats into the knowledge base."""
    
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".text", ".markdown"}
    
    def __init__(
        self,
        vectorstore_manager,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.vectorstore_manager = vectorstore_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_file(self, file_path: str) -> Optional[Document]:
        """Load a single file into a Document."""
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        ext = path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "source": str(path.name),
                "full_path": str(path.absolute()),
                "type": ext.lstrip("."),
                "ingested_at": datetime.now().isoformat(),
                "size_bytes": path.stat().st_size
            }
            
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> Generator[Document, None, None]:
        """Load all supported files from a directory."""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_file(str(file_path))
                if doc:
                    yield doc
    
    def ingest_file(self, file_path: str) -> IngestionResult:
        """Ingest a single file into the knowledge base."""
        try:
            doc = self.load_file(file_path)
            
            if not doc:
                return IngestionResult(
                    file_path=file_path,
                    chunks_added=0,
                    success=False,
                    error="Could not load file or unsupported format"
                )
            
            chunks = self.vectorstore_manager.add_documents([doc])
            
            return IngestionResult(
                file_path=file_path,
                chunks_added=chunks,
                success=True
            )
        except Exception as e:
            return IngestionResult(
                file_path=file_path,
                chunks_added=0,
                success=False,
                error=str(e)
            )
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> List[IngestionResult]:
        """Ingest all documents from a directory."""
        results = []
        
        for doc in self.load_directory(directory, recursive):
            try:
                chunks = self.vectorstore_manager.add_documents([doc])
                results.append(IngestionResult(
                    file_path=doc.metadata.get("full_path", "unknown"),
                    chunks_added=chunks,
                    success=True
                ))
            except Exception as e:
                results.append(IngestionResult(
                    file_path=doc.metadata.get("full_path", "unknown"),
                    chunks_added=0,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def ingest_text(
        self,
        text: str,
        source: str = "direct_input",
        metadata: Optional[dict] = None
    ) -> int:
        """Ingest raw text into the knowledge base."""
        doc_metadata = {
            "source": source,
            "type": "text",
            "ingested_at": datetime.now().isoformat()
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        doc = Document(page_content=text, metadata=doc_metadata)
        return self.vectorstore_manager.add_documents([doc])
    
    def get_ingestion_stats(self, directory: str) -> dict:
        """Get statistics about files that can be ingested."""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return {"exists": False}
        
        stats = {
            "exists": True,
            "total_files": 0,
            "supported_files": 0,
            "by_extension": {},
            "total_size_bytes": 0
        }
        
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                stats["total_files"] += 1
                ext = file_path.suffix.lower()
                
                if ext in self.SUPPORTED_EXTENSIONS:
                    stats["supported_files"] += 1
                    stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
                    stats["total_size_bytes"] += file_path.stat().st_size
        
        return stats


class PDFIngester:
    """Optional PDF ingestion support (requires pypdf)."""
    
    def __init__(self, vectorstore_manager):
        self.vectorstore_manager = vectorstore_manager
        self._check_pdf_support()
    
    def _check_pdf_support(self):
        """Check if PDF support is available."""
        try:
            from pypdf import PdfReader
            self.pdf_available = True
        except ImportError:
            self.pdf_available = False
            print("PDF support not available. Install with: pip install pypdf")
    
    def load_pdf(self, file_path: str) -> Optional[Document]:
        """Load a PDF file."""
        if not self.pdf_available:
            return None
        
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            if not text_parts:
                return None
            
            content = "\n\n".join(text_parts)
            
            metadata = {
                "source": Path(file_path).name,
                "full_path": str(Path(file_path).absolute()),
                "type": "pdf",
                "pages": len(reader.pages),
                "ingested_at": datetime.now().isoformat()
            }
            
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None
    
    def ingest_pdf(self, file_path: str) -> IngestionResult:
        """Ingest a PDF file."""
        if not self.pdf_available:
            return IngestionResult(
                file_path=file_path,
                chunks_added=0,
                success=False,
                error="PDF support not available"
            )
        
        doc = self.load_pdf(file_path)
        
        if not doc:
            return IngestionResult(
                file_path=file_path,
                chunks_added=0,
                success=False,
                error="Could not load PDF"
            )
        
        try:
            chunks = self.vectorstore_manager.add_documents([doc])
            return IngestionResult(
                file_path=file_path,
                chunks_added=chunks,
                success=True
            )
        except Exception as e:
            return IngestionResult(
                file_path=file_path,
                chunks_added=0,
                success=False,
                error=str(e)
            )
