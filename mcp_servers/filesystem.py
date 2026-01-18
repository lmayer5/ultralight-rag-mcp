"""
Filesystem MCP Server - Provides file read/write operations.

This server exposes tools for file system operations that agents can use.
"""

import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP not available. Install with: pip install mcp")


# Allowed directories for safety
ALLOWED_DIRECTORIES = [
    "./data",
    "./data/documents",
]


def is_path_safe(path: str) -> bool:
    """Check if path is within allowed directories."""
    abs_path = os.path.abspath(path)
    for allowed in ALLOWED_DIRECTORIES:
        allowed_abs = os.path.abspath(allowed)
        if abs_path.startswith(allowed_abs):
            return True
    return False


class FilesystemServer:
    """MCP Server for filesystem operations."""
    
    def __init__(self, allowed_dirs: Optional[list] = None):
        self.allowed_dirs = allowed_dirs or ALLOWED_DIRECTORIES
        
        if MCP_AVAILABLE:
            self.server = Server("filesystem")
            self._register_tools()
        else:
            self.server = None
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.server.tool()
        async def file_read(path: str) -> str:
            """Read content from a file.
            
            Args:
                path: Path to the file to read
                
            Returns:
                File contents as string
            """
            if not is_path_safe(path):
                return f"Error: Access denied to path: {path}"
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            except Exception as e:
                return f"Error reading file: {e}"
        
        @self.server.tool()
        async def file_write(path: str, content: str) -> str:
            """Write content to a file.
            
            Args:
                path: Path to the file to write
                content: Content to write
                
            Returns:
                Success message or error
            """
            if not is_path_safe(path):
                return f"Error: Access denied to path: {path}"
            
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote {len(content)} bytes to {path}"
            except Exception as e:
                return f"Error writing file: {e}"
        
        @self.server.tool()
        async def file_list(directory: str) -> str:
            """List files in a directory.
            
            Args:
                directory: Directory path to list
                
            Returns:
                JSON list of files
            """
            if not is_path_safe(directory):
                return f"Error: Access denied to directory: {directory}"
            
            try:
                files = []
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    stat = os.stat(item_path)
                    files.append({
                        "name": item,
                        "type": "directory" if os.path.isdir(item_path) else "file",
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                return json.dumps(files, indent=2)
            except FileNotFoundError:
                return f"Error: Directory not found: {directory}"
            except Exception as e:
                return f"Error listing directory: {e}"
        
        @self.server.tool()
        async def file_delete(path: str) -> str:
            """Delete a file.
            
            Args:
                path: Path to the file to delete
                
            Returns:
                Success message or error
            """
            if not is_path_safe(path):
                return f"Error: Access denied to path: {path}"
            
            try:
                os.remove(path)
                return f"Successfully deleted {path}"
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            except Exception as e:
                return f"Error deleting file: {e}"
    
    # Synchronous fallback methods for direct use
    def read_file(self, path: str) -> str:
        """Read file synchronously."""
        if not is_path_safe(path):
            return f"Error: Access denied to path: {path}"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"
    
    def write_file(self, path: str, content: str) -> str:
        """Write file synchronously."""
        if not is_path_safe(path):
            return f"Error: Access denied to path: {path}"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes"
        except Exception as e:
            return f"Error: {e}"
    
    def list_directory(self, directory: str) -> list:
        """List directory synchronously."""
        if not is_path_safe(directory):
            return []
        try:
            return os.listdir(directory)
        except Exception:
            return []
    
    def delete_file(self, path: str) -> str:
        """Delete file synchronously."""
        if not is_path_safe(path):
            return f"Error: Access denied to path: {path}"
        try:
            os.remove(path)
            return f"Deleted {path}"
        except Exception as e:
            return f"Error: {e}"


# Standalone run for MCP
if __name__ == "__main__":
    if MCP_AVAILABLE:
        import asyncio
        server = FilesystemServer()
        asyncio.run(server.server.run())
    else:
        print("MCP not available. Install with: pip install mcp")
