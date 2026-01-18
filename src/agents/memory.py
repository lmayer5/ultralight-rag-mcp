"""
Conversation Memory - Manages short-term and long-term memory for agents.

Uses SQLite for persistence and provides context for agent interactions.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class Message:
    """A single message in conversation history."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str
    agent_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Fact:
    """A stored fact/piece of knowledge."""
    id: Optional[int]
    content: str
    source: str  # "conversation", "document", "user"
    created_at: str
    tags: List[str]
    importance: int = 1  # 1-5 scale


class ConversationMemory:
    """Manages conversation history and fact storage."""
    
    def __init__(
        self,
        db_path: str = "./data/memory.db",
        max_history: int = 20
    ):
        self.db_path = Path(db_path)
        self.max_history = max_history
        self.conversation_history: List[Message] = []
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_name TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Facts table for long-term knowledge
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    tags TEXT,
                    importance INTEGER DEFAULT 1
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session 
                ON conversations(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_importance 
                ON facts(importance DESC)
            """)
            
            conn.commit()
    
    def add_message(
        self,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: str = "default"
    ) -> Message:
        """Add a message to conversation history."""
        timestamp = datetime.now().isoformat()
        message = Message(
            role=role,
            content=content,
            timestamp=timestamp,
            agent_name=agent_name,
            metadata=metadata
        )
        
        # Add to in-memory history
        self.conversation_history.append(message)
        
        # Trim if exceeds max
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (session_id, role, content, agent_name, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                agent_name,
                timestamp,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
        
        return message
    
    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """Get recent conversation history."""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def get_context_string(self, limit: int = 5) -> str:
        """Get conversation history as a formatted string for context."""
        history = self.get_history(limit)
        if not history:
            return "No previous conversation."
        
        lines = []
        for msg in history:
            prefix = msg.agent_name or msg.role.capitalize()
            lines.append(f"{prefix}: {msg.content}")
        
        return "\n".join(lines)
    
    def add_fact(
        self,
        content: str,
        source: str = "conversation",
        tags: Optional[List[str]] = None,
        importance: int = 1
    ) -> Fact:
        """Add a fact to long-term memory."""
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO facts (content, source, created_at, tags, importance)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content,
                source,
                created_at,
                json.dumps(tags) if tags else None,
                importance
            ))
            fact_id = cursor.lastrowid
            conn.commit()
        
        return Fact(
            id=fact_id,
            content=content,
            source=source,
            created_at=created_at,
            tags=tags or [],
            importance=importance
        )
    
    def search_facts(
        self,
        query: str,
        limit: int = 5
    ) -> List[Fact]:
        """Search facts by content (simple LIKE search)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content, source, created_at, tags, importance
                FROM facts
                WHERE content LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            
            facts = []
            for row in cursor.fetchall():
                facts.append(Fact(
                    id=row[0],
                    content=row[1],
                    source=row[2],
                    created_at=row[3],
                    tags=json.loads(row[4]) if row[4] else [],
                    importance=row[5]
                ))
            
            return facts
    
    def get_important_facts(self, min_importance: int = 3, limit: int = 10) -> List[Fact]:
        """Get most important facts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content, source, created_at, tags, importance
                FROM facts
                WHERE importance >= ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (min_importance, limit))
            
            facts = []
            for row in cursor.fetchall():
                facts.append(Fact(
                    id=row[0],
                    content=row[1],
                    source=row[2],
                    created_at=row[3],
                    tags=json.loads(row[4]) if row[4] else [],
                    importance=row[5]
                ))
            
            return facts
    
    def clear_session(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        self.conversation_history = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM conversations WHERE session_id = ?
            """, (session_id,))
            conn.commit()
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM facts")
            total_facts = cursor.fetchone()[0]
            
            return {
                "total_messages": total_messages,
                "total_facts": total_facts,
                "current_session_messages": len(self.conversation_history)
            }
