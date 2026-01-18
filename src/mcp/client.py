"""
MCP Client - Connects to MCP servers and provides tool access to agents.

Manages MCP server connections and tool invocation.
"""

import asyncio
import subprocess
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str
    callable: Optional[Callable] = None


class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, Any] = {}
        self.tools: Dict[str, MCPTool] = {}
    
    def register_local_server(self, name: str, server_instance: Any):
        """Register a local server instance (for in-process use)."""
        self.servers[name] = {
            "type": "local",
            "instance": server_instance
        }
        
        # Register synchronous methods as tools
        if hasattr(server_instance, 'read_file'):
            self.tools["file_read"] = MCPTool(
                name="file_read",
                description="Read content from a file",
                parameters={"path": "str"},
                server_name=name,
                callable=server_instance.read_file
            )
        
        if hasattr(server_instance, 'write_file'):
            self.tools["file_write"] = MCPTool(
                name="file_write",
                description="Write content to a file",
                parameters={"path": "str", "content": "str"},
                server_name=name,
                callable=server_instance.write_file
            )
        
        if hasattr(server_instance, 'list_directory'):
            self.tools["file_list"] = MCPTool(
                name="file_list",
                description="List files in a directory",
                parameters={"directory": "str"},
                server_name=name,
                callable=server_instance.list_directory
            )
        
        if hasattr(server_instance, 'delete_file'):
            self.tools["file_delete"] = MCPTool(
                name="file_delete",
                description="Delete a file",
                parameters={"path": "str"},
                server_name=name,
                callable=server_instance.delete_file
            )
        
        if hasattr(server_instance, 'search'):
            self.tools["knowledge_search"] = MCPTool(
                name="knowledge_search",
                description="Search the knowledge base",
                parameters={"query": "str", "num_results": "int"},
                server_name=name,
                callable=server_instance.search
            )
        
        if hasattr(server_instance, 'add'):
            self.tools["knowledge_add"] = MCPTool(
                name="knowledge_add",
                description="Add knowledge to the base",
                parameters={"content": "str", "source": "str"},
                server_name=name,
                callable=server_instance.add
            )
        
        if hasattr(server_instance, 'ask'):
            self.tools["knowledge_ask"] = MCPTool(
                name="knowledge_ask",
                description="Ask a question",
                parameters={"question": "str"},
                server_name=name,
                callable=server_instance.ask
            )
        
        print(f"[MCP] Registered local server: {name}")
    
    def register_stdio_server(
        self,
        name: str,
        command: str,
        args: List[str],
        cwd: Optional[str] = None
    ):
        """Register an MCP server that runs via stdio."""
        self.servers[name] = {
            "type": "stdio",
            "command": command,
            "args": args,
            "cwd": cwd,
            "process": None
        }
        print(f"[MCP] Registered stdio server: {name}")
    
    def get_available_tools(self) -> List[MCPTool]:
        """Get list of available tools."""
        return list(self.tools.values())
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools."""
        lines = []
        for tool in self.tools.values():
            params = ", ".join([f"{k}: {v}" for k, v in tool.parameters.items()])
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool by name with arguments."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.tools[tool_name]
        
        if tool.callable:
            try:
                return tool.callable(**kwargs)
            except Exception as e:
                return f"Error calling {tool_name}: {e}"
        else:
            return f"Error: Tool '{tool_name}' has no callable"
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        return tool_name in self.tools


class ToolRegistry:
    """Central registry for all tools available to agents."""
    
    def __init__(self):
        self.mcp_client = MCPClient()
        self.custom_tools: Dict[str, Callable] = {}
    
    def register_mcp_servers(self, filesystem_server=None, knowledge_server=None):
        """Register MCP servers."""
        if filesystem_server:
            self.mcp_client.register_local_server("filesystem", filesystem_server)
        if knowledge_server:
            self.mcp_client.register_local_server("knowledge", knowledge_server)
    
    def register_custom_tool(
        self,
        name: str,
        func: Callable,
        description: str = ""
    ):
        """Register a custom tool function."""
        self.custom_tools[name] = {
            "func": func,
            "description": description
        }
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Get all available tools (MCP + custom)."""
        tools = {}
        
        # Add MCP tools
        for tool in self.mcp_client.get_available_tools():
            tools[tool.name] = {
                "type": "mcp",
                "tool": tool,
                "callable": tool.callable
            }
        
        # Add custom tools
        for name, info in self.custom_tools.items():
            tools[name] = {
                "type": "custom",
                "callable": info["func"],
                "description": info["description"]
            }
        
        return tools
    
    def call(self, tool_name: str, **kwargs) -> Any:
        """Call any registered tool."""
        # Check MCP tools first
        if self.mcp_client.has_tool(tool_name):
            return self.mcp_client.call_tool(tool_name, **kwargs)
        
        # Check custom tools
        if tool_name in self.custom_tools:
            try:
                return self.custom_tools[tool_name]["func"](**kwargs)
            except Exception as e:
                return f"Error: {e}"
        
        return f"Error: Tool '{tool_name}' not found"
    
    def get_tool_summary(self) -> str:
        """Get a summary of all available tools."""
        lines = ["Available Tools:"]
        
        # MCP tools
        for tool in self.mcp_client.get_available_tools():
            lines.append(f"  [MCP] {tool.name}: {tool.description}")
        
        # Custom tools
        for name, info in self.custom_tools.items():
            lines.append(f"  [Custom] {name}: {info.get('description', 'No description')}")
        
        return "\n".join(lines)
