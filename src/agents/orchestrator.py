"""
Agent Orchestrator - Coordinates multiple agents to handle complex tasks.

Routes queries to appropriate agents and manages multi-step workflows.
"""

from typing import Dict, Optional, List, Any
from enum import Enum

from langchain_community.llms import Ollama

from .personas import AgentRole
from .memory import ConversationMemory
from .base import AgentResponse
from .specialized import ResearchAgent, ExecutionAgent, MemoryAgent, PlannerAgent


class QueryIntent(Enum):
    """Detected intent of a user query."""
    QUESTION = "question"      # Research/retrieval needed
    ACTION = "action"          # Execution needed
    REMEMBER = "remember"      # Memory operation
    COMPLEX = "complex"        # Multi-step, needs planning
    UNKNOWN = "unknown"


class AgentOrchestrator:
    """Coordinates multiple specialized agents."""
    
    def __init__(
        self,
        model: str = "mistral",
        memory_db_path: str = "./data/memory.db",
        rag_pipeline=None  # Optional RAGPipeline instance
    ):
        self.model = model
        
        # Shared LLM instance
        self.llm = Ollama(model=model, temperature=0.3)
        
        # Shared memory
        self.memory = ConversationMemory(db_path=memory_db_path)
        
        # Initialize specialized agents
        self.agents: Dict[AgentRole, Any] = {
            AgentRole.RESEARCH: ResearchAgent(
                llm=self.llm,
                memory=self.memory,
                rag_pipeline=rag_pipeline
            ),
            AgentRole.EXECUTION: ExecutionAgent(
                llm=self.llm,
                memory=self.memory
            ),
            AgentRole.MEMORY: MemoryAgent(
                llm=self.llm,
                memory=self.memory
            ),
            AgentRole.PLANNER: PlannerAgent(
                llm=self.llm,
                memory=self.memory
            )
        }
        
        print(f"Agent Orchestrator initialized with {len(self.agents)} agents")
    
    def detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a user query."""
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        # In production, this would use the LLM for classification
        
        # Action keywords
        action_keywords = ["do", "create", "make", "write", "save", "delete", "run", "execute"]
        for kw in action_keywords:
            if query_lower.startswith(kw) or f" {kw} " in query_lower:
                return QueryIntent.ACTION
        
        # Memory keywords
        memory_keywords = ["remember", "forget", "recall", "what did i", "history", "previous"]
        for kw in memory_keywords:
            if kw in query_lower:
                return QueryIntent.REMEMBER
        
        # Question patterns
        question_patterns = ["what", "who", "where", "when", "why", "how", "tell me", "explain", "?"]
        for pattern in question_patterns:
            if query_lower.startswith(pattern) or pattern in query_lower:
                return QueryIntent.QUESTION
        
        # Check for complex multi-part requests
        if " and " in query_lower and len(query.split()) > 15:
            return QueryIntent.COMPLEX
        
        return QueryIntent.UNKNOWN
    
    def route_query(self, query: str, force_agent: Optional[AgentRole] = None) -> AgentRole:
        """Route a query to the appropriate agent."""
        if force_agent:
            return force_agent
        
        intent = self.detect_intent(query)
        
        routing = {
            QueryIntent.QUESTION: AgentRole.RESEARCH,
            QueryIntent.ACTION: AgentRole.EXECUTION,
            QueryIntent.REMEMBER: AgentRole.MEMORY,
            QueryIntent.COMPLEX: AgentRole.PLANNER,
            QueryIntent.UNKNOWN: AgentRole.RESEARCH  # Default to research
        }
        
        return routing[intent]
    
    def process(
        self,
        query: str,
        force_agent: Optional[AgentRole] = None,
        context: str = ""
    ) -> AgentResponse:
        """Process a query using the appropriate agent."""
        # Route to appropriate agent
        target_role = self.route_query(query, force_agent)
        agent = self.agents[target_role]
        
        print(f"[Orchestrator] Routing to {agent.name}")
        
        # Process with the selected agent
        response = agent.process(query, context)
        
        return response
    
    def process_complex(self, query: str) -> List[AgentResponse]:
        """Process a complex query that requires multiple agents."""
        responses = []
        
        # Use planner to decompose
        planner = self.agents[AgentRole.PLANNER]
        steps = planner.decompose_task(query)
        
        if not steps:
            # Fallback to simple processing
            return [self.process(query)]
        
        print(f"[Orchestrator] Decomposed into {len(steps)} steps")
        
        # Execute each step
        context = ""
        for step in steps:
            agent = self.agents[step["agent"]]
            print(f"[Orchestrator] Step {step['step']}: {agent.name}")
            
            response = agent.process(step["description"], context)
            responses.append(response)
            
            # Build context for next step
            if response.success:
                context += f"\nPrevious step result: {response.content[:500]}"
        
        return responses
    
    def get_agent(self, role: AgentRole):
        """Get a specific agent by role."""
        return self.agents.get(role)
    
    def register_tool(self, name: str, func: callable, description: str = ""):
        """Register a tool with the Execution agent."""
        execution_agent = self.agents[AgentRole.EXECUTION]
        execution_agent.register_tool(name, func, description)
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return self.memory.get_stats()
    
    def add_fact(self, content: str, importance: int = 1) -> None:
        """Add a fact to long-term memory."""
        self.memory.add_fact(content, source="user", importance=importance)
    
    def search_facts(self, query: str, limit: int = 5) -> List[Any]:
        """Search facts in memory."""
        return self.memory.search_facts(query, limit)
