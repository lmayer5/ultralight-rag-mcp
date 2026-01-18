"""
RAG Pipeline - Connects LLM with retrieval for question answering.
"""

from typing import Optional, Dict, Any

from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from .vectorstore import VectorStoreManager


# System prompt for the RAG chain
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant acting as a "Second Brain" knowledge system.
Use the following retrieved context to answer the user's question.
If you don't know the answer based on the context, say so honestly.
Always be concise and helpful.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """RAG pipeline connecting Ollama LLM with FAISS retrieval."""
    
    def __init__(
        self,
        model: str = "mistral",
        vectorstore_path: str = "./data/vectorstore",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        retrieval_k: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        self.model = model
        self.retrieval_k = retrieval_k
        
        # Initialize LLM
        print(f"Initializing Ollama LLM with model: {model}")
        self.llm = Ollama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens
        )
        
        # Initialize vector store
        self.vectorstore_manager = VectorStoreManager(
            persist_directory=vectorstore_path,
            embeddings_model=embeddings_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectorstore_manager.load_or_create()
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Build QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore_manager.get_retriever(k=retrieval_k),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        print("RAG pipeline initialized successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            
        Returns:
            Dict with 'result' (answer) and 'source_documents'
        """
        print(f"\n[Query] {question}")
        response = self.qa_chain.invoke({"query": question})
        print(f"[Answer] {response['result'][:200]}...")
        return response
    
    def add_knowledge(self, texts: list, metadatas: Optional[list] = None) -> int:
        """
        Add new knowledge to the system.
        
        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text
            
        Returns:
            Number of chunks added
        """
        return self.vectorstore_manager.add_texts(texts, metadatas)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verify all systems are operational.
        
        Returns:
            Health status dict
        """
        status = {"llm": False, "vectorstore": False, "embeddings": False}
        errors = []
        
        # Test LLM
        try:
            response = self.llm.invoke("Say 'OK' if you're working.")
            status["llm"] = len(response) > 0
        except Exception as e:
            errors.append(f"LLM error: {e}")
        
        # Test vector store
        try:
            results = self.vectorstore_manager.similarity_search("test", k=1)
            status["vectorstore"] = True
        except Exception as e:
            errors.append(f"VectorStore error: {e}")
        
        # Embeddings are tested via vector store
        status["embeddings"] = status["vectorstore"]
        
        return {
            "healthy": all(status.values()),
            "components": status,
            "errors": errors
        }
