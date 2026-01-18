"""
Validation Test Suite - Comprehensive tests for real-world usability.

Tests the entire Second Brain system including:
- RAG retrieval accuracy
- Multi-agent routing
- MCP tool functionality
- Memory persistence
- Performance benchmarks
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rag import RAGPipeline, DocumentIngester
from agents import AgentOrchestrator, AgentRole
from utils.benchmark import Benchmarker, HealthChecker


@dataclass
class TestResult:
    """Result of a validation test."""
    name: str
    passed: bool
    duration_ms: float
    details: str
    expected: str = ""
    actual: str = ""


class ValidationSuite:
    """Comprehensive validation tests for the Second Brain system."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.rag = None
        self.orchestrator = None
    
    def setup(self) -> bool:
        """Initialize system components."""
        print("\n" + "=" * 60)
        print("SECOND BRAIN VALIDATION SUITE")
        print("=" * 60)
        
        print("\n[Setup] Initializing components...")
        
        try:
            # Initialize RAG
            self.rag = RAGPipeline(
                model="mistral",
                vectorstore_path="./data/vectorstore",
                embeddings_model="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("  ✓ RAG pipeline initialized")
            
            # Initialize Orchestrator
            self.orchestrator = AgentOrchestrator(
                model="mistral",
                memory_db_path="./data/memory.db",
                rag_pipeline=self.rag
            )
            print("  ✓ Agent orchestrator initialized")
            
            return True
        except Exception as e:
            print(f"  ✗ Setup failed: {e}")
            return False
    
    def run_all(self) -> Dict[str, Any]:
        """Run all validation tests."""
        if not self.setup():
            return {"success": False, "error": "Setup failed"}
        
        print("\n[Tests] Running validation suite...")
        
        # Document ingestion tests
        self.test_document_ingestion()
        
        # RAG retrieval tests
        self.test_rag_retrieval_accuracy()
        
        # Agent routing tests
        self.test_agent_routing()
        
        # Memory persistence tests
        self.test_memory_persistence()
        
        # Performance tests
        self.test_performance()
        
        # Real-world scenario tests
        self.test_real_world_scenarios()
        
        return self.summarize()
    
    def test_document_ingestion(self):
        """Test document ingestion pipeline."""
        print("\n  [1/6] Document Ingestion Tests")
        
        start = time.perf_counter()
        try:
            ingester = DocumentIngester(self.rag.vectorstore_manager)
            stats = ingester.get_ingestion_stats("./data/documents")
            
            passed = stats.get("supported_files", 0) > 0
            details = f"Found {stats.get('supported_files', 0)} documents"
            
            if passed:
                results = ingester.ingest_directory("./data/documents")
                success_count = sum(1 for r in results if r.success)
                total_chunks = sum(r.chunks_added for r in results)
                details = f"Ingested {success_count}/{len(results)} files, {total_chunks} chunks"
                passed = success_count > 0
            
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(
                name="Document Ingestion",
                passed=passed,
                duration_ms=duration,
                details=details
            ))
            print(f"    {'✓' if passed else '✗'} {details} ({duration:.0f}ms)")
        except Exception as e:
            self.results.append(TestResult(
                name="Document Ingestion",
                passed=False,
                duration_ms=0,
                details=f"Error: {e}"
            ))
            print(f"    ✗ Error: {e}")
    
    def test_rag_retrieval_accuracy(self):
        """Test RAG retrieval accuracy with known queries."""
        print("\n  [2/6] RAG Retrieval Accuracy Tests")
        
        # Test queries that should match our sample documents
        test_cases = [
            {
                "query": "What are the four specialized agents in the system?",
                "expected_keywords": ["research", "execution", "memory", "planner"],
                "source": "system_overview"
            },
            {
                "query": "What does SMART stand for in project management?",
                "expected_keywords": ["specific", "measurable", "achievable", "relevant", "time"],
                "source": "project_management"
            },
            {
                "query": "What is PEP 8 in Python?",
                "expected_keywords": ["style", "indentation", "snake_case"],
                "source": "python_guidelines"
            },
            {
                "query": "What are the types of machine learning?",
                "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
                "source": "machine_learning"
            },
            {
                "query": "What was achieved in Q4 2025?",
                "expected_keywords": ["v2.0", "infrastructure", "compliance"],
                "source": "meeting_notes"
            }
        ]
        
        passed_count = 0
        for tc in test_cases:
            start = time.perf_counter()
            try:
                # Search for relevant documents
                docs = self.rag.vectorstore_manager.similarity_search(tc["query"], k=3)
                
                # Check if any retrieved doc contains expected keywords
                all_content = " ".join([d.page_content.lower() for d in docs])
                keywords_found = sum(1 for kw in tc["expected_keywords"] if kw.lower() in all_content)
                keywords_ratio = keywords_found / len(tc["expected_keywords"])
                
                passed = keywords_ratio >= 0.5  # At least 50% of keywords found
                if passed:
                    passed_count += 1
                
                duration = (time.perf_counter() - start) * 1000
                status = "✓" if passed else "✗"
                print(f"    {status} Query: '{tc['query'][:40]}...' ({keywords_ratio*100:.0f}% match, {duration:.0f}ms)")
                
            except Exception as e:
                print(f"    ✗ Query failed: {e}")
        
        self.results.append(TestResult(
            name="RAG Retrieval Accuracy",
            passed=passed_count >= 3,  # At least 3/5 queries pass
            duration_ms=0,
            details=f"{passed_count}/{len(test_cases)} queries retrieved accurate results"
        ))
    
    def test_agent_routing(self):
        """Test that queries are routed to correct agents."""
        print("\n  [3/6] Agent Routing Tests")
        
        routing_tests = [
            ("What is machine learning?", AgentRole.RESEARCH),
            ("Remember that my API key is abc123", AgentRole.MEMORY),
            ("Create a task breakdown for launching a product", AgentRole.PLANNER),
        ]
        
        passed_count = 0
        for query, expected_role in routing_tests:
            actual_role = self.orchestrator.route_query(query)
            passed = actual_role == expected_role
            if passed:
                passed_count += 1
            status = "✓" if passed else "✗"
            print(f"    {status} '{query[:35]}...' → {actual_role.value} (expected: {expected_role.value})")
        
        self.results.append(TestResult(
            name="Agent Routing",
            passed=passed_count >= 2,
            duration_ms=0,
            details=f"{passed_count}/{len(routing_tests)} queries routed correctly"
        ))
    
    def test_memory_persistence(self):
        """Test that memory persists across operations."""
        print("\n  [4/6] Memory Persistence Tests")
        
        start = time.perf_counter()
        try:
            # Add a fact
            self.orchestrator.add_fact("Test fact: The sky is blue", importance=4)
            
            # Search for it
            facts = self.orchestrator.search_facts("sky blue")
            found = any("sky" in f.content.lower() for f in facts)
            
            # Check stats
            stats = self.orchestrator.get_memory_stats()
            
            duration = (time.perf_counter() - start) * 1000
            passed = found and stats.get("total_facts", 0) > 0
            
            self.results.append(TestResult(
                name="Memory Persistence",
                passed=passed,
                duration_ms=duration,
                details=f"Fact stored and retrieved, {stats.get('total_facts', 0)} total facts"
            ))
            print(f"    {'✓' if passed else '✗'} Fact storage and retrieval ({duration:.0f}ms)")
        except Exception as e:
            self.results.append(TestResult(
                name="Memory Persistence",
                passed=False,
                duration_ms=0,
                details=f"Error: {e}"
            ))
            print(f"    ✗ Error: {e}")
    
    def test_performance(self):
        """Test system performance meets requirements."""
        print("\n  [5/6] Performance Tests")
        
        benchmarker = Benchmarker(self.rag, self.orchestrator)
        
        # Embedding performance
        start = time.perf_counter()
        self.rag.vectorstore_manager.embeddings.embed_query("Test embedding query")
        embed_time = (time.perf_counter() - start) * 1000
        embed_passed = embed_time < 500  # Should be under 500ms
        print(f"    {'✓' if embed_passed else '✗'} Embedding: {embed_time:.0f}ms (target: <500ms)")
        
        # Similarity search performance
        start = time.perf_counter()
        self.rag.vectorstore_manager.similarity_search("test query", k=3)
        search_time = (time.perf_counter() - start) * 1000
        search_passed = search_time < 100  # Should be under 100ms
        print(f"    {'✓' if search_passed else '✗'} Similarity search: {search_time:.0f}ms (target: <100ms)")
        
        self.results.append(TestResult(
            name="Performance",
            passed=embed_passed and search_passed,
            duration_ms=embed_time + search_time,
            details=f"Embed: {embed_time:.0f}ms, Search: {search_time:.0f}ms"
        ))
    
    def test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        print("\n  [6/6] Real-World Scenario Tests")
        
        scenarios = [
            {
                "name": "Knowledge Worker Query",
                "query": "What are the key metrics I should track for project monitoring?",
                "expected_in_response": ["schedule", "cost", "scope", "risk"]
            },
            {
                "name": "Developer Query",
                "query": "How should I write docstrings in Python?",
                "expected_in_response": ["google", "docstring", "args", "returns"]
            },
            {
                "name": "Learning Query",
                "query": "Explain the difference between supervised and unsupervised learning",
                "expected_in_response": ["labeled", "classification", "clustering"]
            }
        ]
        
        passed_count = 0
        for scenario in scenarios:
            start = time.perf_counter()
            try:
                response = self.orchestrator.process(scenario["query"])
                duration = (time.perf_counter() - start) * 1000
                
                # Check if response contains expected keywords
                response_lower = response.content.lower()
                keywords_found = sum(1 for kw in scenario["expected_in_response"] 
                                    if kw.lower() in response_lower)
                
                passed = keywords_found >= 1 and response.success
                if passed:
                    passed_count += 1
                
                status = "✓" if passed else "✗"
                print(f"    {status} {scenario['name']}: {duration:.0f}ms ({keywords_found}/{len(scenario['expected_in_response'])} keywords)")
                
            except Exception as e:
                print(f"    ✗ {scenario['name']}: Error - {e}")
        
        self.results.append(TestResult(
            name="Real-World Scenarios",
            passed=passed_count >= 2,
            duration_ms=0,
            details=f"{passed_count}/{len(scenarios)} scenarios handled correctly"
        ))
    
    def summarize(self) -> Dict[str, Any]:
        """Summarize all test results."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status}: {result.name}")
            print(f"         {result.details}")
        
        print("-" * 60)
        overall = passed == total
        print(f"{'✓' if overall else '✗'} OVERALL: {passed}/{total} tests passed")
        print("=" * 60)
        
        return {
            "success": overall,
            "passed": passed,
            "total": total,
            "results": self.results
        }


def main():
    """Run the validation suite."""
    suite = ValidationSuite()
    results = suite.run_all()
    
    if results.get("success"):
        print("\n🎉 All validation tests passed!")
        print("The system is ready for real-world use.")
    else:
        print("\n⚠️ Some tests failed.")
        print("Review the results above and ensure Ollama is running.")
    
    return results


if __name__ == "__main__":
    main()
