# Ultralight RAG MCP - Second Brain Agent System

A lightweight local "Second Brain" agent system with RAG, MCP, and local LLMs.

## Requirements

- **Python**: 3.10+ (3.12 recommended)
- **Ollama**: For local LLM inference
- **GPU**: 8GB VRAM recommended (RTX 2070 or equivalent)

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
│   ├── main.py           # Entry point
│   └── rag/              # RAG components
├── data/
│   ├── documents/        # Source documents
│   └── vectorstore/      # FAISS index
├── config/
│   └── config.json       # Configuration
└── mcp_servers/          # MCP tools (Phase 3)
```

## Configuration

Edit `config/config.json` to customize:
- LLM model and parameters
- RAG chunk size and retrieval settings
- Memory persistence options

## Phases

- **Phase 1**: Core Setup (RAG + LLM) ✅
- **Phase 2**: Multi-Agent System
- **Phase 3**: MCP Integration
- **Phase 4**: Knowledge Base & Persistence
- **Phase 5**: Testing & Optimization
