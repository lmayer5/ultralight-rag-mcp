"""
Direct Ollama API client for streaming responses.
"""

import requests
import json
from typing import Generator, Optional, Dict, Any, List


class OllamaClient:
    """Direct HTTP client for Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.current_model: Optional[str] = None
    
    def is_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 512,
        context: Optional[List[int]] = None
    ) -> str:
        """Generate a response (non-streaming)."""
        model = model or self.current_model
        if not model:
            raise ValueError("No model specified")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama error: {response.status_code}")
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """Generate a streaming response."""
        model = model or self.current_model
        if not model:
            raise ValueError("No model specified")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code}"
        except requests.exceptions.Timeout:
            yield "Error: Request timed out"
        except requests.exceptions.ConnectionError:
            yield "Error: Cannot connect to Ollama"
    
    def set_model(self, model: str):
        """Set the current model."""
        self.current_model = model
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """Chat with conversation history."""
        model = model or self.current_model
        if not model:
            raise ValueError("No model specified")
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=120
            )
            
            if response.status_code == 200:
                if stream:
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    data = response.json()
                    yield data.get("message", {}).get("content", "")
            else:
                yield f"Error: {response.status_code}"
        except requests.exceptions.Timeout:
            yield "Error: Request timed out"
        except requests.exceptions.ConnectionError:
            yield "Error: Cannot connect to Ollama"
