"""
Second Brain - Local RAG Agent System
Main entry point with multi-agent and MCP tool support.
"""

import sys
import io

# Fix Windows encoding for emoji/unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag import RAGPipeline, DocumentIngester
from agents import AgentOrchestrator, AgentRole
from mcp import ToolRegistry
from utils.config import load_config

# Import MCP servers
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_servers"))
from filesystem import FilesystemServer
from knowledge_api import KnowledgeServer


def print_banner():
    """Display startup banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║           🧠 SECOND BRAIN - Multi-Agent + MCP             ║
║               Ultralight Edition v3.0                     ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_help():
    """Display help information."""
    print("""
Commands:
  [query]         Ask a question (auto-routed to best agent)
  @research       Force Research Agent
  @execute        Force Execution Agent  
  @memory         Force Memory Agent
  @plan           Force Planner Agent

MCP Tools:
  /tool list      List available MCP tools
  /tool <name>    Call a tool (e.g., /tool file_list ./data)
  
Knowledge:
  /add            Add knowledge to the system
  /ingest         Ingest documents from ./data/documents
  /ingest <path>  Ingest from specific path
  /fact           Add a fact to long-term memory
  /search <query> Search facts

System:
  /stats          Show memory statistics
  /agents         List available agents
  /health         Run health check
  /help           Show this help
  /quit           Exit the system
    """)


def main():
    """Main entry point."""
    print_banner()
    
    # Load configuration
    try:
        config = load_config("./config/config.json")
        print("✓ Configuration loaded")
    except FileNotFoundError:
        print("⚠ Config not found, using defaults")
        from utils.config import get_default_config
        config = get_default_config()
    
    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag = None
    try:
        rag = RAGPipeline(
            model=config["llm"]["model"],
            vectorstore_path=config["rag"]["vectordb_path"],
            embeddings_model=config["rag"]["embeddings_model"],
            chunk_size=config["rag"]["chunk_size"],
            chunk_overlap=config["rag"]["chunk_overlap"],
            retrieval_k=config["rag"]["retrieval_k"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"]
        )
        print("✓ RAG pipeline ready")
    except Exception as e:
        print(f"⚠ RAG initialization failed: {e}")
        print("  Continuing without RAG support...")
    
    # Initialize Agent Orchestrator
    print("\nInitializing Agent Orchestrator...")
    try:
        orchestrator = AgentOrchestrator(
            model=config["llm"]["model"],
            memory_db_path=config["memory"]["db_path"],
            rag_pipeline=rag
        )
        print("✓ Agent Orchestrator ready")
    except Exception as e:
        print(f"✗ Failed to initialize agents: {e}")
        print("\nMake sure Ollama is running with the Mistral model:")
        print("  ollama serve")
        print("  ollama pull mistral")
        sys.exit(1)
    
    # Initialize MCP Tool Registry
    print("\nInitializing MCP Tools...")
    tool_registry = ToolRegistry()
    
    # Register MCP servers
    fs_server = FilesystemServer()
    tool_registry.register_mcp_servers(filesystem_server=fs_server)
    
    if rag:
        knowledge_server = KnowledgeServer(
            rag_pipeline=rag,
            memory=orchestrator.memory
        )
        tool_registry.register_mcp_servers(knowledge_server=knowledge_server)
    
    # Register tools with Execution agent
    for tool_name, tool_info in tool_registry.get_all_tools().items():
        if tool_info.get("callable"):
            orchestrator.register_tool(
                tool_name,
                tool_info["callable"],
                tool_info.get("tool", {}).description if hasattr(tool_info.get("tool", {}), 'description') else ""
            )
    
    print(f"✓ {len(tool_registry.get_all_tools())} MCP tools registered")
    
    # Health check
    print("\nRunning health check...")
    if rag:
        health = rag.health_check()
        if health["healthy"]:
            print("✓ All systems operational")
        else:
            print(f"⚠ Health check warnings: {health['errors']}")
    else:
        print("⚠ Running in limited mode (no RAG)")
    
    print_help()
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🧠 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "/help":
                print_help()
                continue
            
            if user_input.lower() == "/stats":
                stats = orchestrator.get_memory_stats()
                print(f"\n📊 Memory Statistics:")
                print(f"   Total messages: {stats['total_messages']}")
                print(f"   Total facts: {stats['total_facts']}")
                print(f"   Current session: {stats['current_session_messages']}")
                continue
            
            if user_input.lower() == "/agents":
                print("\n🤖 Available Agents:")
                print("   • Research Agent - Knowledge retrieval and synthesis")
                print("   • Execution Agent - Task execution and tool calling")
                print("   • Memory Agent - Conversation and fact management")
                print("   • Planner Agent - Task decomposition and coordination")
                continue
            
            if user_input.lower() == "/tool list":
                print("\n🔧 Available MCP Tools:")
                print(tool_registry.get_tool_summary())
                continue
            
            if user_input.lower().startswith("/tool "):
                parts = user_input[6:].strip().split(" ", 1)
                tool_name = parts[0]
                
                if tool_name == "list":
                    print(tool_registry.get_tool_summary())
                    continue
                
                # Simple tool calling
                if tool_name == "file_list":
                    directory = parts[1] if len(parts) > 1 else "./data"
                    result = tool_registry.call(tool_name, directory=directory)
                    print(f"\n📁 Files: {result}")
                elif tool_name == "file_read":
                    if len(parts) > 1:
                        result = tool_registry.call(tool_name, path=parts[1])
                        print(f"\n📄 Content:\n{result[:500]}...")
                    else:
                        print("Usage: /tool file_read <path>")
                elif tool_name == "knowledge_search":
                    if len(parts) > 1:
                        result = tool_registry.call(tool_name, query=parts[1])
                        print(f"\n🔍 Results:\n{result}")
                    else:
                        print("Usage: /tool knowledge_search <query>")
                else:
                    print(f"Tool '{tool_name}' - use /tool list to see available tools")
                continue
            
            if user_input.lower() == "/health":
                if rag:
                    health = rag.health_check()
                    print(f"\n🏥 Health Check: {health}")
                else:
                    print("\n⚠ RAG not available - limited health check")
                continue
            
            if user_input.lower() == "/add":
                print("Enter text to add to knowledge base (empty line to finish):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                if lines and rag:
                    text = "\n".join(lines)
                    count = rag.add_knowledge([text])
                    print(f"✓ Added {count} chunk(s) to knowledge base")
                elif not rag:
                    print("⚠ RAG not available")
                continue
            
            if user_input.lower().startswith("/ingest"):
                if not rag:
                    print("⚠ RAG not available")
                    continue
                
                parts = user_input.split(" ", 1)
                directory = parts[1].strip() if len(parts) > 1 else "./data/documents"
                
                print(f"\n📂 Ingesting documents from: {directory}")
                ingester = DocumentIngester(rag.vectorstore_manager)
                
                # Show stats first
                stats = ingester.get_ingestion_stats(directory)
                if not stats.get("exists"):
                    print(f"⚠ Directory not found: {directory}")
                    continue
                
                print(f"   Found {stats['supported_files']} supported files")
                
                # Ingest
                results = ingester.ingest_directory(directory)
                
                success = sum(1 for r in results if r.success)
                total_chunks = sum(r.chunks_added for r in results)
                
                print(f"✓ Ingested {success}/{len(results)} files ({total_chunks} chunks)")
                
                # Show any errors
                for r in results:
                    if not r.success:
                        print(f"   ⚠ Failed: {r.file_path} - {r.error}")
                continue
            
            if user_input.lower().startswith("/search "):
                query = user_input[8:].strip()
                facts = orchestrator.search_facts(query)
                if facts:
                    print("\n📚 Found facts:")
                    for f in facts:
                        print(f"   [{f.importance}] {f.content}")
                else:
                    print("No matching facts found.")
                continue
            
            if user_input.lower().startswith("/fact "):
                fact_content = user_input[6:].strip()
                if fact_content:
                    orchestrator.add_fact(fact_content, importance=3)
                    print(f"✓ Fact stored: {fact_content[:50]}...")
                continue
            
            if user_input.lower() == "/fact":
                print("Enter fact to remember:")
                fact = input().strip()
                if fact:
                    orchestrator.add_fact(fact, importance=3)
                    print(f"✓ Fact stored")
                continue
            
            # Handle agent-specific routing
            force_agent = None
            query = user_input
            
            if user_input.startswith("@research "):
                force_agent = AgentRole.RESEARCH
                query = user_input[10:]
            elif user_input.startswith("@execute "):
                force_agent = AgentRole.EXECUTION
                query = user_input[9:]
            elif user_input.startswith("@memory "):
                force_agent = AgentRole.MEMORY
                query = user_input[8:]
            elif user_input.startswith("@plan "):
                force_agent = AgentRole.PLANNER
                query = user_input[6:]
            
            # Process with orchestrator
            response = orchestrator.process(query, force_agent=force_agent)
            
            # Display response
            print(f"\n🤖 [{response.agent_name}]: {response.content}")
            
            # Show sources if available
            if response.sources:
                print("\n📚 Sources:")
                for i, source in enumerate(response.sources[:3], 1):
                    print(f"   {i}. {source}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
