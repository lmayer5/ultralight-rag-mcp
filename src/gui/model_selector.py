"""
Model selector component - manages Ollama model selection.
"""

import customtkinter as ctk
import requests
from typing import Callable, List, Optional
from threading import Thread
from .theme import COLORS, FONTS


class ModelSelector(ctk.CTkFrame):
    """Dropdown for selecting Ollama models with refresh capability."""
    
    def __init__(self, parent, on_model_change: Callable[[str], None]):
        super().__init__(parent, fg_color=COLORS["bg_medium"])
        self.on_model_change = on_model_change
        self.current_model: Optional[str] = None
        self._models: List[str] = []
        
        self._setup_ui()
        self.refresh_models()
    
    def _setup_ui(self):
        # Label
        label = ctk.CTkLabel(
            self,
            text="🧠 Model:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        label.pack(side="left", padx=(10, 5))
        
        # Model dropdown
        self.model_dropdown = ctk.CTkComboBox(
            self,
            values=["Loading..."],
            command=self._on_selection,
            font=FONTS["body"],
            width=200,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            button_color=COLORS["accent_secondary"],
            button_hover_color=COLORS["accent_primary"],
            dropdown_fg_color=COLORS["bg_light"],
            dropdown_hover_color=COLORS["accent_secondary"],
            text_color=COLORS["text_primary"]
        )
        self.model_dropdown.pack(side="left", padx=5, pady=10)
        
        # Refresh button
        self.refresh_btn = ctk.CTkButton(
            self,
            text="🔄",
            width=36,
            height=36,
            font=FONTS["body"],
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["accent_secondary"],
            border_width=1,
            border_color=COLORS["border"],
            command=self.refresh_models
        )
        self.refresh_btn.pack(side="left", padx=5, pady=10)
        
        # Model info label
        self.info_label = ctk.CTkLabel(
            self,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.info_label.pack(side="left", padx=10)
    
    def _on_selection(self, model_name: str):
        """Handle model selection change."""
        if model_name != self.current_model and model_name != "Loading...":
            self.current_model = model_name
            self._update_info(model_name)
            self.on_model_change(model_name)
    
    def _update_info(self, model_name: str):
        """Update the info label with model details."""
        # Show model size estimate based on name
        size_hints = {
            "0.5b": "~300MB",
            "1b": "~600MB",
            "2b": "~1.5GB",
            "3b": "~2GB",
            "7b": "~4GB",
            "8b": "~5GB",
            "13b": "~8GB",
        }
        
        size = "Unknown size"
        for hint, estimate in size_hints.items():
            if hint in model_name.lower():
                size = estimate
                break
        
        self.info_label.configure(text=f"({size})")
    
    def refresh_models(self):
        """Refresh the list of available models from Ollama."""
        self.model_dropdown.configure(values=["Loading..."])
        self.model_dropdown.set("Loading...")
        
        def fetch():
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    self._update_models(models if models else ["No models found"])
                else:
                    self._update_models(["Ollama error"])
            except requests.exceptions.ConnectionError:
                self._update_models(["Ollama not running"])
            except Exception as e:
                self._update_models(["Error loading"])
        
        Thread(target=fetch, daemon=True).start()
    
    def _update_models(self, models: List[str]):
        """Update dropdown with fetched models (thread-safe)."""
        def update():
            self._models = models
            self.model_dropdown.configure(values=models)
            
            # Set default model
            if models and models[0] not in ["No models found", "Ollama not running", "Error loading"]:
                # Prefer smaller models for laptops
                preferred = ["ministral", "phi", "gemma", "tinyllama", "qwen"]
                default = models[0]
                
                for pref in preferred:
                    for m in models:
                        if pref in m.lower():
                            default = m
                            break
                    else:
                        continue
                    break
                
                self.model_dropdown.set(default)
                self.current_model = default
                self._update_info(default)
                self.on_model_change(default)
        
        # Schedule on main thread
        self.after(0, update)
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently selected model."""
        return self.current_model
    
    def set_model(self, model_name: str):
        """Set the current model."""
        if model_name in self._models:
            self.model_dropdown.set(model_name)
            self.current_model = model_name
            self._update_info(model_name)
