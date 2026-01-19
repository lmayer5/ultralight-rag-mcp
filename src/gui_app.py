"""
Second Brain Desktop GUI Application
Main entry point for the graphical interface.
"""

import sys
import os
from pathlib import Path
from threading import Thread
from typing import Dict, Any, List, Optional

import customtkinter as ctk

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.theme import COLORS, FONTS, DIMENSIONS, configure_theme
from gui.chat_panel import ChatPanel
from gui.model_selector import ModelSelector
from gui.file_manager import FileManager
from gui.settings_dialog import SettingsDialog
from ollama_client import OllamaClient


class SecondBrainApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Configure theme
        configure_theme()
        
        # Window setup
        self.title("🧠 Second Brain")
        self.geometry(f"{DIMENSIONS['window_width']}x{DIMENSIONS['window_height']}")
        self.configure(fg_color=COLORS["bg_dark"])
        self.minsize(700, 500)
        
        # State
        self.settings: Dict[str, Any] = {
            "temperature": 0.3,
            "max_tokens": 512,
            "retrieval_k": 3,
            "chunk_size": 512,
            "memory_enabled": True,
            "streaming": True
        }
        self.current_model: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.rag_pipeline = None
        
        # Initialize Ollama client
        self.ollama = OllamaClient()
        
        # Try to initialize RAG pipeline
        self._init_rag()
        
        # Build UI
        self._setup_ui()
        
        # Check Ollama status
        self.after(100, self._check_ollama)
    
    def _init_rag(self):
        """Initialize RAG pipeline if available."""
        try:
            from rag.retrieval import RAGPipeline
            self.rag_pipeline = RAGPipeline(
                model=self.current_model or "mistral",
                retrieval_k=self.settings["retrieval_k"],
                chunk_size=self.settings["chunk_size"]
            )
        except Exception as e:
            print(f"RAG initialization skipped: {e}")
            self.rag_pipeline = None
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Top bar
        self._setup_topbar()
        
        # Main content area (split view)
        content = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"])
        content.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left sidebar (file manager)
        sidebar = ctk.CTkFrame(
            content, 
            fg_color=COLORS["bg_medium"],
            width=DIMENSIONS["sidebar_width"],
            corner_radius=12
        )
        sidebar.pack(side="left", fill="y", padx=(0, 10), pady=5)
        sidebar.pack_propagate(False)
        
        self.file_manager = FileManager(sidebar, self._on_files_added)
        self.file_manager.pack(fill="both", expand=True)
        
        # Right side (chat)
        chat_container = ctk.CTkFrame(content, fg_color=COLORS["bg_medium"], corner_radius=12)
        chat_container.pack(side="right", fill="both", expand=True, pady=5)
        
        self.chat_panel = ChatPanel(chat_container, self._on_send_message)
        self.chat_panel.pack(fill="both", expand=True)
        
        # Status bar
        self._setup_statusbar()
        
        # Add welcome message
        self.after(500, self._show_welcome)
    
    def _setup_topbar(self):
        """Setup the top bar with model selector and settings."""
        topbar = ctk.CTkFrame(self, fg_color=COLORS["bg_medium"], height=60, corner_radius=0)
        topbar.pack(fill="x", padx=10, pady=(10, 5))
        topbar.pack_propagate(False)
        
        # App title
        title = ctk.CTkLabel(
            topbar,
            text="🧠 Second Brain",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["text_primary"]
        )
        title.pack(side="left", padx=15)
        
        # Settings button
        settings_btn = ctk.CTkButton(
            topbar,
            text="⚙️ Settings",
            font=FONTS["button"],
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["accent_secondary"],
            border_width=1,
            border_color=COLORS["border"],
            width=100,
            command=self._open_settings
        )
        settings_btn.pack(side="right", padx=15, pady=12)
        
        # Clear chat button
        clear_btn = ctk.CTkButton(
            topbar,
            text="🗑️ Clear",
            font=FONTS["button"],
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["accent_error"],
            border_width=1,
            border_color=COLORS["border"],
            width=80,
            command=self._clear_chat
        )
        clear_btn.pack(side="right", padx=5, pady=12)
        
        # Model selector
        self.model_selector = ModelSelector(topbar, self._on_model_change)
        self.model_selector.pack(side="left", padx=20, pady=10)
    
    def _setup_statusbar(self):
        """Setup the status bar at the bottom."""
        statusbar = ctk.CTkFrame(self, fg_color=COLORS["bg_medium"], height=30, corner_radius=0)
        statusbar.pack(fill="x", padx=10, pady=(5, 10))
        statusbar.pack_propagate(False)
        
        # Model status
        self.model_status = ctk.CTkLabel(
            statusbar,
            text="⚡ No model selected",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.model_status.pack(side="left", padx=15)
        
        # Document count
        self.doc_status = ctk.CTkLabel(
            statusbar,
            text="📚 0 documents",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.doc_status.pack(side="left", padx=15)
        
        # Ollama status
        self.ollama_status = ctk.CTkLabel(
            statusbar,
            text="🔴 Ollama: Checking...",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.ollama_status.pack(side="right", padx=15)
    
    def _check_ollama(self):
        """Check if Ollama is running."""
        if self.ollama.is_running():
            self.ollama_status.configure(
                text="🟢 Ollama: Connected",
                text_color=COLORS["accent_success"]
            )
        else:
            self.ollama_status.configure(
                text="🔴 Ollama: Not running",
                text_color=COLORS["accent_error"]
            )
        
        # Check again in 10 seconds
        self.after(10000, self._check_ollama)
    
    def _show_welcome(self):
        """Show welcome message."""
        welcome = """👋 Welcome to Second Brain!

I'm your local AI assistant powered by Ollama. Here's how to get started:

1. **Select a model** from the dropdown above
2. **Upload documents** on the left to build your knowledge base
3. **Ask me anything** - I'll use your documents for context

💡 Tip: Smaller models (like phi, gemma) run faster on laptops without GPUs."""
        
        self.chat_panel.add_assistant_message(welcome)
    
    def _on_model_change(self, model: str):
        """Handle model selection change."""
        self.current_model = model
        self.ollama.set_model(model)
        self.model_status.configure(
            text=f"⚡ {model}",
            text_color=COLORS["text_primary"]
        )
    
    def _on_files_added(self, files: List[Path]):
        """Handle new files being added."""
        # Update status
        total = len(self.file_manager.get_documents())
        self.doc_status.configure(text=f"📚 {total} document(s)")
        
        # Ingest files into RAG if available
        if self.rag_pipeline:
            def ingest():
                try:
                    for f in files:
                        content = f.read_text(encoding="utf-8", errors="ignore")
                        self.rag_pipeline.add_knowledge(
                            [content],
                            [{"source": str(f), "filename": f.name}]
                        )
                    self.after(0, lambda: self.chat_panel.add_assistant_message(
                        f"✅ Successfully indexed {len(files)} file(s) into the knowledge base."
                    ))
                except Exception as e:
                    self.after(0, lambda: self.chat_panel.add_assistant_message(
                        f"⚠️ Error indexing files: {e}"
                    ))
            
            Thread(target=ingest, daemon=True).start()
        else:
            self.chat_panel.add_assistant_message(
                f"📎 Added {len(files)} file(s). Note: RAG indexing is not available - files will be used for reference only."
            )
    
    def _on_send_message(self, message: str):
        """Handle user sending a message."""
        if not self.current_model:
            self.chat_panel.add_assistant_message(
                "⚠️ Please select a model from the dropdown first."
            )
            return
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Disable input while generating
        self.chat_panel.set_input_enabled(False)
        
        # Start generating response
        if self.settings["streaming"]:
            self._generate_streaming(message)
        else:
            self._generate_blocking(message)
    
    def _generate_streaming(self, message: str):
        """Generate a streaming response."""
        bubble = self.chat_panel.start_streaming()
        
        def generate():
            full_response = ""
            try:
                # Build system prompt
                system = self._build_system_prompt()
                
                # Generate with streaming
                for chunk in self.ollama.generate_stream(
                    prompt=message,
                    system=system,
                    temperature=self.settings["temperature"],
                    max_tokens=self.settings["max_tokens"]
                ):
                    full_response += chunk
                    self.after(0, lambda t=full_response: bubble.update_text(t))
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
            except Exception as e:
                full_response = f"❌ Error: {e}"
                self.after(0, lambda: bubble.update_text(full_response))
            
            # Re-enable input
            self.after(0, self._finish_generation)
        
        Thread(target=generate, daemon=True).start()
    
    def _generate_blocking(self, message: str):
        """Generate a non-streaming response."""
        def generate():
            try:
                system = self._build_system_prompt()
                response = self.ollama.generate(
                    prompt=message,
                    system=system,
                    temperature=self.settings["temperature"],
                    max_tokens=self.settings["max_tokens"]
                )
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                self.after(0, lambda: self.chat_panel.add_assistant_message(response))
                
            except Exception as e:
                self.after(0, lambda: self.chat_panel.add_assistant_message(f"❌ Error: {e}"))
            
            self.after(0, self._finish_generation)
        
        Thread(target=generate, daemon=True).start()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with optional RAG context."""
        base = """You are a helpful AI assistant acting as a "Second Brain" knowledge system.
Be concise, accurate, and helpful. If you don't know something, say so honestly."""
        
        # Add RAG context if available
        if self.rag_pipeline and self.conversation_history:
            last_user_msg = self.conversation_history[-1]["content"]
            try:
                results = self.rag_pipeline.vectorstore_manager.similarity_search(
                    last_user_msg, 
                    k=self.settings["retrieval_k"]
                )
                if results:
                    context = "\n\n".join([doc.page_content for doc in results])
                    base += f"\n\nRelevant context from the knowledge base:\n{context}"
            except:
                pass
        
        return base
    
    def _finish_generation(self):
        """Called when generation is complete."""
        self.chat_panel.finish_streaming()
        self.chat_panel.set_input_enabled(True)
    
    def _open_settings(self):
        """Open settings dialog."""
        SettingsDialog(self, self.settings, self._on_settings_save)
    
    def _on_settings_save(self, new_settings: Dict[str, Any]):
        """Handle settings saved."""
        self.settings = new_settings
        self.chat_panel.add_assistant_message("✅ Settings updated.")
    
    def _clear_chat(self):
        """Clear chat history."""
        self.chat_panel.clear_messages()
        self.conversation_history.clear()
        self._show_welcome()


def main():
    """Main entry point."""
    app = SecondBrainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
