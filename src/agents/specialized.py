"""
Specialized Agents - Concrete implementations for each agent role.
"""

from typing import Optional, List, Dict, Any

from langchain_community.llms import Ollama

from .base import BaseAgent, AgentResponse
from .personas import AgentPersona, AgentRole, AGENT_PERSONAS
from .memory import ConversationMemory


class ResearchAgent(BaseAgent):
    """Agent specialized in knowledge retrieval and synthesis."""
    
    def __init__(
        self,
        llm: Optional[Ollama] = None,
        memory: Optional[ConversationMemory] = None,
        model: str = "mistral",
        rag_pipeline=None  # RAGPipeline instance
    ):
        super().__init__(
            persona=AGENT_PERSONAS[AgentRole.RESEARCH],
            llm=llm,
            memory=memory,
            model=model
        )
        self.rag_pipeline = rag_pipeline
    
    def process(self, user_input: str, context: str = "") -> AgentResponse:
        """Process a research query using RAG."""
        sources = []
        
        try:
            # Use RAG if available
            if self.rag_pipeline:
                rag_response = self.rag_pipeline.query(user_input)
                content = rag_response["result"]
                
                # Extract sources
                if rag_response.get("source_documents"):
                    sources = [
                        doc.metadata.get("source", "unknown")
                        for doc in rag_response["source_documents"]
                    ]
            else:
                # Fallback to direct LLM
                prompt = self._build_prompt(user_input, context)
                content = self._invoke_llm(prompt)
            
            self._log_interaction(user_input, content)
            
            return AgentResponse(
                content=content,
                agent_name=self.name,
                agent_role=self.role,
                sources=sources,
                success=True
            )
        except Exception as e:
            return AgentResponse(
                content=f"Research failed: {str(e)}",
                agent_name=self.name,
                agent_role=self.role,
                success=False,
                error=str(e)
            )


class ExecutionAgent(BaseAgent):
    """Agent specialized in executing tasks and calling tools."""
    
    def __init__(
        self,
        llm: Optional[Ollama] = None,
        memory: Optional[ConversationMemory] = None,
        model: str = "mistral",
        tools: Optional[Dict[str, callable]] = None
    ):
        super().__init__(
            persona=AGENT_PERSONAS[AgentRole.EXECUTION],
            llm=llm,
            memory=memory,
            model=model
        )
        self.tools = tools or {}
    
    def register_tool(self, name: str, func: callable, description: str = ""):
        """Register a tool for the agent to use."""
        self.tools[name] = {"func": func, "description": description}
    
    def process(self, user_input: str, context: str = "") -> AgentResponse:
        """Process an execution request."""
        try:
            # Build prompt with tool information
            tool_info = "\n".join([
                f"- {name}: {info.get('description', 'No description')}"
                for name, info in self.tools.items()
            ])
            
            enhanced_context = context
            if tool_info:
                enhanced_context = f"Available tools:\n{tool_info}\n\n{context}"
            
            prompt = self._build_prompt(user_input, enhanced_context)
            content = self._invoke_llm(prompt)
            
            self._log_interaction(user_input, content)
            
            return AgentResponse(
                content=content,
                agent_name=self.name,
                agent_role=self.role,
                success=True,
                metadata={"available_tools": list(self.tools.keys())}
            )
        except Exception as e:
            return AgentResponse(
                content=f"Execution failed: {str(e)}",
                agent_name=self.name,
                agent_role=self.role,
                success=False,
                error=str(e)
            )


class MemoryAgent(BaseAgent):
    """Agent specialized in managing conversation and facts."""
    
    def __init__(
        self,
        llm: Optional[Ollama] = None,
        memory: Optional[ConversationMemory] = None,
        model: str = "mistral"
    ):
        super().__init__(
            persona=AGENT_PERSONAS[AgentRole.MEMORY],
            llm=llm,
            memory=memory,
            model=model
        )
    
    def process(self, user_input: str, context: str = "") -> AgentResponse:
        """Process a memory-related request."""
        try:
            # Add memory stats to context
            if self.memory:
                stats = self.memory.get_stats()
                memory_context = f"Memory stats: {stats}\n\n{context}"
            else:
                memory_context = context
            
            prompt = self._build_prompt(user_input, memory_context)
            content = self._invoke_llm(prompt)
            
            self._log_interaction(user_input, content)
            
            return AgentResponse(
                content=content,
                agent_name=self.name,
                agent_role=self.role,
                success=True
            )
        except Exception as e:
            return AgentResponse(
                content=f"Memory operation failed: {str(e)}",
                agent_name=self.name,
                agent_role=self.role,
                success=False,
                error=str(e)
            )
    
    def extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text using LLM."""
        prompt = f"""Extract the key facts from the following text. 
Return each fact on a new line, prefixed with "- ".
Only include important, factual information.

Text:
{text}

Facts:"""
        
        try:
            response = self._invoke_llm(prompt)
            facts = [
                line.strip().lstrip("- ")
                for line in response.split("\n")
                if line.strip().startswith("-")
            ]
            return facts
        except Exception:
            return []


class PlannerAgent(BaseAgent):
    """Agent specialized in task decomposition and coordination."""
    
    def __init__(
        self,
        llm: Optional[Ollama] = None,
        memory: Optional[ConversationMemory] = None,
        model: str = "mistral"
    ):
        super().__init__(
            persona=AGENT_PERSONAS[AgentRole.PLANNER],
            llm=llm,
            memory=memory,
            model=model
        )
    
    def process(self, user_input: str, context: str = "") -> AgentResponse:
        """Process a planning request."""
        try:
            prompt = self._build_prompt(user_input, context)
            content = self._invoke_llm(prompt)
            
            self._log_interaction(user_input, content)
            
            return AgentResponse(
                content=content,
                agent_name=self.name,
                agent_role=self.role,
                success=True
            )
        except Exception as e:
            return AgentResponse(
                content=f"Planning failed: {str(e)}",
                agent_name=self.name,
                agent_role=self.role,
                success=False,
                error=str(e)
            )
    
    def decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """Decompose a complex task into steps with agent assignments."""
        prompt = f"""Decompose the following task into clear steps.
For each step, specify which agent should handle it (Research, Execution, or Memory).

Task: {task}

Format your response as:
1. [AgentType] Step description
2. [AgentType] Step description
...

Steps:"""
        
        try:
            response = self._invoke_llm(prompt)
            steps = []
            
            for line in response.split("\n"):
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                
                # Parse step
                try:
                    # Remove leading number and period
                    content = line.split(".", 1)[1].strip() if "." in line else line
                    
                    # Extract agent type
                    if "[Research]" in content:
                        agent = AgentRole.RESEARCH
                        content = content.replace("[Research]", "").strip()
                    elif "[Execution]" in content:
                        agent = AgentRole.EXECUTION
                        content = content.replace("[Execution]", "").strip()
                    elif "[Memory]" in content:
                        agent = AgentRole.MEMORY
                        content = content.replace("[Memory]", "").strip()
                    else:
                        agent = AgentRole.RESEARCH  # Default
                    
                    steps.append({
                        "step": len(steps) + 1,
                        "agent": agent,
                        "description": content
                    })
                except Exception:
                    continue
            
            return steps
        except Exception:
            return []
