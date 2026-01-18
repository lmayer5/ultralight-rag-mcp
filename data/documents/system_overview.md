# Second Brain System Overview

The Second Brain is a personal knowledge management system that helps you capture, organize, and retrieve information efficiently.

## Core Capabilities

### 1. Knowledge Retrieval
The system uses RAG (Retrieval-Augmented Generation) to find relevant information from your documents. When you ask a question, it:
- Searches the vector database for semantically similar content
- Retrieves the top 3 most relevant chunks
- Uses the LLM to synthesize an answer grounded in your data

### 2. Multi-Agent Architecture
Four specialized agents handle different types of requests:
- **Research Agent**: Best for questions and information retrieval
- **Execution Agent**: Handles file operations and tool calls
- **Memory Agent**: Manages conversation history and facts
- **Planner Agent**: Breaks down complex tasks into steps

### 3. Persistent Memory
The system remembers:
- Conversation history (last 20 messages)
- Important facts you tell it (stored in SQLite)
- All documents you ingest (stored in FAISS vector database)

## Typical Use Cases

1. **Personal Knowledge Base**: Store notes, articles, research papers
2. **Project Documentation**: Query your project docs naturally
3. **Learning Assistant**: Ask questions about topics you're studying
4. **Task Planning**: Break down complex projects into actionable steps

## Performance Expectations

- Query latency: 2-8 seconds (depending on LLM speed)
- Token generation: 8-15 tokens/second on RTX 2070
- Knowledge base capacity: 50K-500K document chunks
