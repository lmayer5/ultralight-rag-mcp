"""
Benchmark utilities for the Second Brain system.

Measures performance of RAG queries, agent responses, and system health.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    success_rate: float
    tokens_per_second: Optional[float] = None


class Benchmarker:
    """Benchmark utilities for the Second Brain system."""
    
    def __init__(self, rag_pipeline=None, orchestrator=None):
        self.rag = rag_pipeline
        self.orchestrator = orchestrator
    
    def benchmark_rag_query(
        self,
        queries: List[str],
        iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark RAG query performance."""
        if not self.rag:
            return BenchmarkResult(
                name="RAG Query",
                iterations=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                success_rate=0
            )
        
        times = []
        successes = 0
        
        for query in queries:
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    self.rag.query(query)
                    successes += 1
                except Exception:
                    pass
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        total = len(queries) * iterations
        
        return BenchmarkResult(
            name="RAG Query",
            iterations=total,
            avg_time_ms=mean(times) if times else 0,
            min_time_ms=min(times) if times else 0,
            max_time_ms=max(times) if times else 0,
            std_dev_ms=stdev(times) if len(times) > 1 else 0,
            success_rate=successes / total if total > 0 else 0
        )
    
    def benchmark_embedding(
        self,
        texts: List[str],
        iterations: int = 3
    ) -> BenchmarkResult:
        """Benchmark embedding generation performance."""
        if not self.rag:
            return BenchmarkResult(
                name="Embedding",
                iterations=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                success_rate=0
            )
        
        times = []
        successes = 0
        
        embeddings = self.rag.vectorstore_manager.embeddings
        
        for text in texts:
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    embeddings.embed_query(text)
                    successes += 1
                except Exception:
                    pass
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        total = len(texts) * iterations
        
        return BenchmarkResult(
            name="Embedding",
            iterations=total,
            avg_time_ms=mean(times) if times else 0,
            min_time_ms=min(times) if times else 0,
            max_time_ms=max(times) if times else 0,
            std_dev_ms=stdev(times) if len(times) > 1 else 0,
            success_rate=successes / total if total > 0 else 0
        )
    
    def benchmark_similarity_search(
        self,
        queries: List[str],
        k: int = 3,
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark vector similarity search."""
        if not self.rag:
            return BenchmarkResult(
                name="Similarity Search",
                iterations=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                success_rate=0
            )
        
        times = []
        successes = 0
        
        for query in queries:
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    self.rag.vectorstore_manager.similarity_search(query, k=k)
                    successes += 1
                except Exception:
                    pass
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        total = len(queries) * iterations
        
        return BenchmarkResult(
            name="Similarity Search",
            iterations=total,
            avg_time_ms=mean(times) if times else 0,
            min_time_ms=min(times) if times else 0,
            max_time_ms=max(times) if times else 0,
            std_dev_ms=stdev(times) if len(times) > 1 else 0,
            success_rate=successes / total if total > 0 else 0
        )
    
    def run_full_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run a comprehensive benchmark suite."""
        test_queries = [
            "What is the main purpose of this system?",
            "How does the knowledge base work?",
            "Explain the agent architecture."
        ]
        
        test_texts = [
            "This is a test document for embedding.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence and machine learning."
        ]
        
        results = {}
        
        print("Running benchmarks...")
        
        print("  - Embedding benchmark...")
        results["embedding"] = self.benchmark_embedding(test_texts, iterations=3)
        
        print("  - Similarity search benchmark...")
        results["similarity_search"] = self.benchmark_similarity_search(
            test_queries, iterations=3
        )
        
        print("  - RAG query benchmark...")
        results["rag_query"] = self.benchmark_rag_query(test_queries[:1], iterations=2)
        
        return results
    
    def print_results(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Test':<25} {'Avg (ms)':<12} {'Min':<10} {'Max':<10} {'Success':<10}")
        print("-" * 70)
        
        for name, result in results.items():
            print(
                f"{result.name:<25} "
                f"{result.avg_time_ms:<12.2f} "
                f"{result.min_time_ms:<10.2f} "
                f"{result.max_time_ms:<10.2f} "
                f"{result.success_rate * 100:<10.1f}%"
            )
        
        print("=" * 70)


class HealthChecker:
    """Comprehensive health checking for the system."""
    
    def __init__(self, rag_pipeline=None, orchestrator=None, tool_registry=None):
        self.rag = rag_pipeline
        self.orchestrator = orchestrator
        self.tool_registry = tool_registry
    
    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        results = {
            "llm": self._check_llm(),
            "embeddings": self._check_embeddings(),
            "vectorstore": self._check_vectorstore(),
            "memory": self._check_memory(),
            "tools": self._check_tools()
        }
        
        # Overall status
        results["overall"] = {
            "healthy": all(r.get("healthy", False) for r in results.values()),
            "components": sum(1 for r in results.values() if r.get("healthy", False)),
            "total": len(results) - 1  # Exclude "overall"
        }
        
        return results
    
    def _check_llm(self) -> Dict[str, Any]:
        """Check LLM connectivity."""
        try:
            if self.rag:
                response = self.rag.llm.invoke("Say OK")
                return {"healthy": len(response) > 0, "status": "connected"}
            return {"healthy": False, "status": "not initialized"}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}
    
    def _check_embeddings(self) -> Dict[str, Any]:
        """Check embedding model."""
        try:
            if self.rag:
                embedding = self.rag.vectorstore_manager.embeddings.embed_query("test")
                return {
                    "healthy": len(embedding) > 0,
                    "status": "working",
                    "dimensions": len(embedding)
                }
            return {"healthy": False, "status": "not initialized"}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}
    
    def _check_vectorstore(self) -> Dict[str, Any]:
        """Check vector store."""
        try:
            if self.rag:
                vs = self.rag.vectorstore_manager.vectorstore
                count = vs.index.ntotal if vs else 0
                return {
                    "healthy": True,
                    "status": "connected",
                    "vector_count": count
                }
            return {"healthy": False, "status": "not initialized"}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory system."""
        try:
            if self.orchestrator:
                stats = self.orchestrator.get_memory_stats()
                return {
                    "healthy": True,
                    "status": "connected",
                    "messages": stats.get("total_messages", 0),
                    "facts": stats.get("total_facts", 0)
                }
            return {"healthy": False, "status": "not initialized"}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}
    
    def _check_tools(self) -> Dict[str, Any]:
        """Check MCP tools."""
        try:
            if self.tool_registry:
                tools = self.tool_registry.get_all_tools()
                return {
                    "healthy": len(tools) > 0,
                    "status": "available",
                    "tool_count": len(tools)
                }
            return {"healthy": False, "status": "not initialized"}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}
    
    def print_report(self, results: Dict[str, Dict[str, Any]]):
        """Print health check report."""
        print("\n" + "=" * 50)
        print("HEALTH CHECK REPORT")
        print("=" * 50)
        
        for component, status in results.items():
            if component == "overall":
                continue
            
            icon = "✓" if status.get("healthy") else "✗"
            print(f"{icon} {component.upper()}: {status.get('status', 'unknown')}")
            
            # Print additional details
            for key, value in status.items():
                if key not in ["healthy", "status", "error"]:
                    print(f"    {key}: {value}")
            
            if "error" in status:
                print(f"    error: {status['error']}")
        
        print("-" * 50)
        overall = results.get("overall", {})
        healthy = overall.get("healthy", False)
        icon = "✓" if healthy else "✗"
        print(f"{icon} OVERALL: {overall.get('components', 0)}/{overall.get('total', 0)} components healthy")
        print("=" * 50)
