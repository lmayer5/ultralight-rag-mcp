# Ultralight RAG MCP - Second Brain Agent System

A lightweight, fully local "Second Brain" agent system combining RAG, multi-agent orchestration, MCP tools, and local LLM inference via Ollama.

## Features

- **Local RAG Pipeline** - FAISS vector store with sentence-transformer embeddings
- **Multi-Agent System** - Research, Execution, Memory, and Planner agents
- **MCP Integration** - Extensible tool protocol for filesystem and knowledge operations
- **Persistent Memory** - SQLite-backed conversation history and fact storage
- **Offline-First** - No external API dependencies at runtime

## Requirements

- **Python**: 3.10+ (3.12 recommended)
- **Ollama**: For local LLM inference
- **GPU**: 8GB VRAM recommended (RTX 2070 or equivalent)
- **RAM**: 8GB minimum, 16GB recommended

## Quick Start

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install.

```bash
# Pull Mistral 7B model
ollama pull mistral
```

### 2. Set Up Python Environment

```powershell
# Create virtual environment
py -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Documents

Place your documents (markdown, text, PDF) in `data/documents/`.

### 4. Run

```bash
py src/main.py
```

## Project Structure

```
├── src/
│   ├── main.py              # Entry point & CLI
│   ├── agents/              # Multi-agent system
│   │   ├── orchestrator.py  # Query routing & coordination
│   │   ├── specialized.py   # Research, Execution, Memory, Planner agents
│   │   ├── memory.py        # Conversation & fact memory
│   │   └── personas.py      # Agent role definitions
│   ├── rag/                 # RAG components
│   │   ├── ingestion.py     # Document loading & chunking
│   │   ├── retrieval.py     # Query retrieval pipeline
│   │   ├── vectorstore.py   # FAISS vector store manager
│   │   └── embeddings.py    # Embedding model wrapper
│   ├── mcp/                 # MCP tool registry
│   └── utils/               # Configuration & helpers
├── mcp_servers/             # MCP tool servers
│   ├── filesystem.py        # File I/O operations
│   └── knowledge_api.py     # Knowledge base operations
├── data/
│   ├── documents/           # Source documents
│   └── vectorstore/         # FAISS index (auto-generated)
├── config/
│   └── config.json          # Configuration
└── docs/
    └── performance_analysis.md
```

## Configuration

Edit `config/config.json` to customize:

```json
{
  "llm": {
    "model": "mistral",
    "temperature": 0.3
  },
  "rag": {
    "chunk_size": 512,
    "retrieval_k": 3
  },
  "memory": {
    "persistence": true,
    "db_path": "./data/memory.db"
  }
}
```

## Usage

The system provides an interactive CLI with the following commands:

| Command | Description |
|---------|-------------|
| Type a question | Query the knowledge base |
| `/ingest` | Ingest documents from data/documents/ |
| `/memory` | View memory statistics |
| `/agents` | List available agents |
| `/help` | Show available commands |
| `/quit` | Exit the application |

## Performance

| Metric | Expected |
|--------|----------|
| Query latency | 3-6 seconds |
| Embedding generation | 50-200ms |
| Token generation | 10-12 tok/s |
| VRAM usage | 4-6GB |

## Project Status

- **Phase 1**: Core Setup (RAG + LLM) ✅
- **Phase 2**: Multi-Agent System ✅
- **Phase 3**: MCP Integration ✅
- **Phase 4**: Knowledge Base & Persistence ✅
- **Phase 5**: Testing & Optimization ✅

## License

MIT
