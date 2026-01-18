# Ultralight RAG MCP - Project Summary

## Overview

**Ultralight RAG MCP** is a lightweight, fully local "Second Brain" agent system that combines **Retrieval-Augmented Generation (RAG)**, **Model Context Protocol (MCP)** integration, and **local LLM inference** via Ollama. The system is specifically designed to run on legacy/resource-constrained hardware (4-core CPU, 16GB RAM, 8GB VRAM).

---

## Core Architecture

```mermaid
graph TD
    A[User Query] --> B[Agent Orchestrator]
    B --> C{Intent Detection}
    C --> D[Research Agent]
    C --> E[Execution Agent]
    C --> F[Memory Agent]
    C --> G[Planner Agent]
    
    D --> H[RAG Pipeline]
    H --> I[FAISS Vector Store]
    H --> J[Ollama LLM]
    
    E --> K[MCP Tool Registry]
    K --> L[Filesystem Server]
    K --> M[Knowledge Server]
    
    F --> N[SQLite Memory DB]
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Inference** | Ollama + Mistral 7B | Local language model (4-bit quantized) |
| **Embeddings** | MiniLM-L6-v2 | Fast, lightweight sentence embeddings |
| **Vector Store** | FAISS (CPU) | Similarity search and retrieval |
| **Memory** | SQLite | Persistent conversation and fact storage |
| **Tool Protocol** | MCP (Model Context Protocol) | Extensible tool registration |

---

## Use Cases

### ✅ Personal Knowledge Base
- Ingest markdown, text, and PDF documents
- Query personal notes with natural language
- Expected: 3-5 second response time for 500+ documents

### ✅ Project Documentation Assistant  
- Index and search project documentation
- Answer technical questions with grounded context
- Cross-reference information across multiple files

### ✅ Learning & Research Aid
- Study topics with contextual grounding
- Extract and retain key facts automatically
- Build long-term knowledge through conversation memory

### ✅ Task Planning & Execution
- Decompose complex tasks into actionable steps
- Multi-agent coordination for research + execution workflows
- Persistent session memory for continuity

---

## Multi-Agent System

The system implements four specialized agents:

| Agent | Role | Key Capabilities |
|-------|------|------------------|
| **Research Agent** | Knowledge retrieval | RAG queries, document synthesis |
| **Execution Agent** | Task execution | MCP tool calling, file operations |
| **Memory Agent** | Information management | Fact storage, recall, conversation history |
| **Planner Agent** | Coordination | Task decomposition, multi-step workflows |

The **Agent Orchestrator** routes queries based on detected intent (question, action, remember, complex).

---

## MCP Servers

Two MCP servers provide extensible tool capabilities:

### 1. Filesystem Server
- `file_read` / `file_write` - Safe file I/O in allowed directories
- `file_list` / `file_delete` - Directory operations
- Path safety validation built-in

### 2. Knowledge Server
- `search_knowledge` - Query the vector store
- `add_knowledge` - Ingest new content dynamically
- `ask_question` - Full RAG query pipeline
- `store_fact` / `search_facts` - Memory operations

---

## Feasibility Assessment

### Performance Metrics (Target Hardware)

| Metric | Target | Status |
|--------|--------|--------|
| Query latency (with LLM) | 2-8 sec | ✅ Achievable (3-6 sec) |
| Embedding generation | <500ms | ✅ Excellent (50-200ms) |
| Similarity search | <100ms | ✅ Excellent (10-50ms) |
| Token generation | 8-15 tok/s | ✅ Achievable |
| VRAM usage | <8GB | ✅ Within limits (4-6GB) |

### Hardware Requirements

| Tier | CPU | RAM | GPU | Storage |
|------|-----|-----|-----|---------|
| **Minimum** | 4-core | 8GB | GTX 1060 6GB | 20GB SSD |
| **Recommended** | 6-core | 16GB | RTX 2070 8GB | 50GB SSD |

### Strengths
- ✅ Fully offline/local - no API dependencies
- ✅ Fast retrieval with FAISS
- ✅ Persistent memory across sessions
- ✅ Extensible via MCP tool protocol
- ✅ Low resource footprint

### Limitations
- ⚠️ LLM latency (3-6 seconds per query)
- ⚠️ 4096 token context window
- ⚠️ Offline-only (no internet search)
- ⚠️ PDF support requires optional `pypdf` dependency

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Core RAG + LLM setup |
| Phase 2 | ✅ Complete | Multi-Agent System |
| Phase 3 | ✅ Complete | MCP Integration |
| Phase 4 | ✅ Complete | Knowledge Base & Persistence |
| Phase 5 | ✅ Complete | Testing & Optimization |

---

## Quick Start

```powershell
# 1. Install Ollama and pull model
ollama pull mistral

# 2. Setup Python environment
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Add documents to data/documents/

# 4. Run
py src/main.py
```

---

## Conclusion

**Verdict: ✅ Production-Ready for Personal Use**

The Ultralight RAG MCP system is a well-architected, practical solution for building a local AI-powered knowledge assistant. It successfully balances performance, resource constraints, and extensibility, making it ideal for:

- **Knowledge workers** managing personal notes and documentation
- **Developers** building local AI tooling
- **Researchers** who need offline, privacy-preserving AI assistance

For enterprise or high-scale use, consider upgrading to larger models (13B+) and PostgreSQL + pgvector for the vector store.
