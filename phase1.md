# Implementation Plan: Local "Second Brain" Agent System

**Project:** Lightweight Local Agent System (LLM + RAG + MCP)  
**Target Hardware:** Legacy Workstation (4-core CPU, 16GB RAM, RTX 2070-equiv 8GB VRAM)  
**Primary Constraints:** - **Strict Offline/Local Priority:** No dependency on external APIs for runtime.  
- **VRAM Budget:** Max 8GB. Models must be quantized (4-bit GGUF).  
- **Latency Target:** <8 seconds per query (Retrieval + Inference).

---

## Phase 1: Environment & Core Inference Setup
**Objective:** Establish the local runtime environment and verify inference capabilities on target hardware.

### 1.1 Local LLM Runtime Initialization
- **Action:** Install and configure **Ollama** (preferred) or KoboldCPP.
- **Model Selection:** Download **Mistral 7B** or **Llama 3 8B** in GGUF format (Quantization: Q4_K_M).
- **Verification:**
    - Execute `ollama run mistral "Explain quantum entanglement in one sentence."`
    - **Pass Criteria:** Response generated >5 tokens/second; VRAM usage <5GB.

### 1.2 Python Environment Configuration
- **Action:** Initialize Python 3.11 virtual environment (`secondbrain`).
- **Dependencies:** Install core libraries.
  ```bash
  pip install langchain langchain-community faiss-cpu sentence-transformers ollama pydantic pydantic-ai