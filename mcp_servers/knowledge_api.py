"""
Knowledge API MCP Server - Provides knowledge base operations.

This server exposes tools for interacting with the RAG knowledge base.
"""

import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class KnowledgeServer:
    """MCP Server for knowledge base operations."""
    
    def __init__(self, rag_pipeline=None, memory=None):
        self.rag_pipeline = rag_pipeline
        self.memory = memory
        
        if MCP_AVAILABLE:
            self.server = Server("knowledge")
            self._register_tools()
        else:
            self.server = None
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.server.tool()
        async def search_knowledge(query: str, num_results: int = 3) -> str:
            """Search the knowledge base for relevant information.
            
            Args:
                query: Search query
                num_results: Number of results to return (default: 3)
                
            Returns:
                Search results as formatted string
            """
            if not self.rag_pipeline:
                return "Error: Knowledge base not initialized"
            
            try:
                docs = self.rag_pipeline.vectorstore_manager.similarity_search(
                    query, k=num_results
                )
                
                results = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "unknown")
                    content = doc.page_content[:300]
                    results.append(f"{i}. [{source}]\n{content}...")
                
                return "\n\n".join(results) if results else "No relevant results found."
            except Exception as e:
                return f"Error searching: {e}"
        
        @self.server.tool()
        async def add_knowledge(content: str, source: str = "mcp") -> str:
            """Add new knowledge to the knowledge base.
            
            Args:
                content: Text content to add
                source: Source identifier
                
            Returns:
                Success message with number of chunks added
            """
            if not self.rag_pipeline:
                return "Error: Knowledge base not initialized"
            
            try:
                count = self.rag_pipeline.add_knowledge(
                    [content],
                    metadatas=[{"source": source}]
                )
                return f"Successfully added {count} chunk(s) to knowledge base"
            except Exception as e:
                return f"Error adding knowledge: {e}"
        
        @self.server.tool()
        async def ask_question(question: str) -> str:
            """Ask a question and get an answer from the knowledge base.
            
            Args:
                question: Question to ask
                
            Returns:
                Answer based on knowledge base content
            """
            if not self.rag_pipeline:
                return "Error: Knowledge base not initialized"
            
            try:
                response = self.rag_pipeline.query(question)
                return response["result"]
            except Exception as e:
                return f"Error answering question: {e}"
        
        @self.server.tool()
        async def store_fact(fact: str, importance: int = 1) -> str:
            """Store a fact in long-term memory.
            
            Args:
                fact: Fact to store
                importance: Importance level 1-5 (default: 1)
                
            Returns:
                Confirmation message
            """
            if not self.memory:
                return "Error: Memory not initialized"
            
            try:
                self.memory.add_fact(
                    content=fact,
                    source="mcp",
                    importance=min(max(importance, 1), 5)
                )
                return f"Stored fact with importance {importance}"
            except Exception as e:
                return f"Error storing fact: {e}"
        
        @self.server.tool()
        async def search_facts(query: str, limit: int = 5) -> str:
            """Search stored facts.
            
            Args:
                query: Search query
                limit: Maximum results (default: 5)
                
            Returns:
                Matching facts
            """
            if not self.memory:
                return "Error: Memory not initialized"
            
            try:
                facts = self.memory.search_facts(query, limit=limit)
                if not facts:
                    return "No matching facts found."
                
                results = []
                for f in facts:
                    results.append(f"[Importance: {f.importance}] {f.content}")
                return "\n".join(results)
            except Exception as e:
                return f"Error searching facts: {e}"
        
        @self.server.tool()
        async def get_memory_stats() -> str:
            """Get memory statistics.
            
            Returns:
                Memory stats as formatted string
            """
            if not self.memory:
                return "Error: Memory not initialized"
            
            try:
                stats = self.memory.get_stats()
                return (
                    f"Messages: {stats['total_messages']}\n"
                    f"Facts: {stats['total_facts']}\n"
                    f"Session: {stats['current_session_messages']}"
                )
            except Exception as e:
                return f"Error getting stats: {e}"
    
    # Synchronous methods for direct use
    def search(self, query: str, num_results: int = 3) -> List[str]:
        """Search knowledge base synchronously."""
        if not self.rag_pipeline:
            return []
        try:
            docs = self.rag_pipeline.vectorstore_manager.similarity_search(
                query, k=num_results
            )
            return [doc.page_content for doc in docs]
        except Exception:
            return []
    
    def add(self, content: str, source: str = "direct") -> bool:
        """Add knowledge synchronously."""
        if not self.rag_pipeline:
            return False
        try:
            self.rag_pipeline.add_knowledge([content], [{"source": source}])
            return True
        except Exception:
            return False
    
    def ask(self, question: str) -> str:
        """Ask a question synchronously."""
        if not self.rag_pipeline:
            return "Knowledge base not available"
        try:
            response = self.rag_pipeline.query(question)
            return response["result"]
        except Exception as e:
            return f"Error: {e}"


# Standalone run for MCP
if __name__ == "__main__":
    if MCP_AVAILABLE:
        import asyncio
        server = KnowledgeServer()
        asyncio.run(server.server.run())
    else:
        print("MCP not available. Install with: pip install mcp")
