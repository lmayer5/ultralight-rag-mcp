"""
Base Agent - Abstract base class for all specialized agents.

Provides common functionality for LLM interaction and tool calling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from langchain_community.llms import Ollama

from .personas import AgentPersona, AgentRole
from .memory import ConversationMemory


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    agent_name: str
    agent_role: AgentRole
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for specialized agents."""
    
    def __init__(
        self,
        persona: AgentPersona,
        llm: Optional[Ollama] = None,
        memory: Optional[ConversationMemory] = None,
        model: str = "mistral"
    ):
        self.persona = persona
        self.memory = memory
        
        # Initialize LLM if not provided
        if llm:
            self.llm = llm
        else:
            self.llm = Ollama(
                model=model,
                temperature=persona.temperature,
                num_predict=persona.max_tokens
            )
    
    @property
    def name(self) -> str:
        return self.persona.name
    
    @property
    def role(self) -> AgentRole:
        return self.persona.role
    
    def _build_prompt(self, user_input: str, context: str = "") -> str:
        """Build the full prompt with system prompt and context."""
        parts = [self.persona.system_prompt]
        
        # Add memory context if available
        if self.memory:
            history = self.memory.get_context_string(limit=3)
            if history and history != "No previous conversation.":
                parts.append(f"\n\nRecent conversation:\n{history}")
        
        # Add additional context if provided
        if context:
            parts.append(f"\n\nContext:\n{context}")
        
        # Add the user input
        parts.append(f"\n\nUser: {user_input}")
        parts.append("\n\nAssistant:")
        
        return "".join(parts)
    
    @abstractmethod
    def process(self, user_input: str, context: str = "") -> AgentResponse:
        """Process user input and return a response. Must be implemented by subclasses."""
        pass
    
    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with the given prompt."""
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}")
    
    def _log_interaction(self, user_input: str, response: str):
        """Log the interaction to memory if available."""
        if self.memory:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response, agent_name=self.name)
