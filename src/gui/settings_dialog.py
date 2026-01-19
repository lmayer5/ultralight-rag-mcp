"""
Settings dialog for Second Brain configuration.
"""

import customtkinter as ctk
from typing import Dict, Any, Callable, Optional
from .theme import COLORS, FONTS, DIMENSIONS


class SettingsDialog(ctk.CTkToplevel):
    """Modal settings dialog."""
    
    def __init__(self, parent, current_settings: Dict[str, Any], on_save: Callable[[Dict[str, Any]], None]):
        super().__init__(parent)
        self.on_save = on_save
        self.settings = current_settings.copy()
        
        # Window setup
        self.title("⚙️ Settings")
        self.geometry("450x500")
        self.configure(fg_color=COLORS["bg_dark"])
        self.resizable(False, False)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self._setup_ui()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 450) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 500) // 2
        self.geometry(f"+{x}+{y}")
    
    def _setup_ui(self):
        # Header
        header = ctk.CTkLabel(
            self,
            text="⚙️ Settings",
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS["text_primary"]
        )
        header.pack(pady=20)
        
        # Settings container
        container = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["bg_medium"],
            corner_radius=12
        )
        container.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # LLM Settings Section
        self._add_section(container, "🤖 LLM Settings")
        
        # Temperature
        self.temp_var = ctk.DoubleVar(value=self.settings.get("temperature", 0.3))
        self._add_slider(
            container, 
            "Temperature", 
            self.temp_var, 
            0.0, 1.0, 
            "Lower = more focused, Higher = more creative"
        )
        
        # Max tokens
        self.tokens_var = ctk.IntVar(value=self.settings.get("max_tokens", 512))
        self._add_slider(
            container,
            "Max Response Tokens",
            self.tokens_var,
            128, 2048,
            "Maximum length of responses"
        )
        
        # RAG Settings Section
        self._add_section(container, "📚 Retrieval Settings")
        
        # Retrieval K
        self.k_var = ctk.IntVar(value=self.settings.get("retrieval_k", 3))
        self._add_slider(
            container,
            "Retrieved Chunks",
            self.k_var,
            1, 10,
            "Number of document chunks to use for context"
        )
        
        # Chunk size
        self.chunk_var = ctk.IntVar(value=self.settings.get("chunk_size", 512))
        self._add_slider(
            container,
            "Chunk Size",
            self.chunk_var,
            256, 1024,
            "Size of document chunks for indexing"
        )
        
        # Memory Section
        self._add_section(container, "🧠 Memory Settings")
        
        # Enable memory
        self.memory_var = ctk.BooleanVar(value=self.settings.get("memory_enabled", True))
        self._add_toggle(
            container,
            "Enable Memory",
            self.memory_var,
            "Store conversation history and facts"
        )
        
        # Streaming
        self.stream_var = ctk.BooleanVar(value=self.settings.get("streaming", True))
        self._add_toggle(
            container,
            "Streaming Responses",
            self.stream_var,
            "Show responses as they're generated"
        )
        
        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=15)
        
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color=COLORS["bg_medium"],
            hover_color=COLORS["bg_light"],
            border_width=1,
            border_color=COLORS["border"],
            command=self.destroy
        )
        cancel_btn.pack(side="left", expand=True, padx=5)
        
        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save Settings",
            fg_color=COLORS["accent_primary"],
            hover_color=COLORS["accent_secondary"],
            command=self._save
        )
        save_btn.pack(side="right", expand=True, padx=5)
    
    def _add_section(self, parent, title: str):
        """Add a section header."""
        label = ctk.CTkLabel(
            parent,
            text=title,
            font=FONTS["heading"],
            text_color=COLORS["text_primary"],
            anchor="w"
        )
        label.pack(fill="x", padx=10, pady=(15, 5))
        
        sep = ctk.CTkFrame(parent, fg_color=COLORS["separator"], height=1)
        sep.pack(fill="x", padx=10, pady=(0, 10))
    
    def _add_slider(self, parent, label: str, var, min_val, max_val, description: str):
        """Add a slider setting."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=5)
        
        # Label and value
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        
        lbl = ctk.CTkLabel(
            header,
            text=label,
            font=FONTS["body"],
            text_color=COLORS["text_primary"]
        )
        lbl.pack(side="left")
        
        val_label = ctk.CTkLabel(
            header,
            text=str(var.get()),
            font=FONTS["mono"],
            text_color=COLORS["accent_primary"]
        )
        val_label.pack(side="right")
        
        # Slider
        def on_change(value):
            if isinstance(var, ctk.IntVar):
                var.set(int(value))
                val_label.configure(text=str(int(value)))
            else:
                var.set(round(value, 2))
                val_label.configure(text=f"{value:.2f}")
        
        slider = ctk.CTkSlider(
            frame,
            from_=min_val,
            to=max_val,
            variable=var,
            command=on_change,
            fg_color=COLORS["bg_input"],
            progress_color=COLORS["accent_secondary"],
            button_color=COLORS["accent_primary"],
            button_hover_color=COLORS["accent_primary"]
        )
        slider.pack(fill="x", pady=(5, 0))
        
        # Description
        desc = ctk.CTkLabel(
            frame,
            text=description,
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        desc.pack(anchor="w")
    
    def _add_toggle(self, parent, label: str, var, description: str):
        """Add a toggle setting."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=5)
        
        # Label and switch
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        
        lbl = ctk.CTkLabel(
            header,
            text=label,
            font=FONTS["body"],
            text_color=COLORS["text_primary"]
        )
        lbl.pack(side="left")
        
        switch = ctk.CTkSwitch(
            header,
            text="",
            variable=var,
            onvalue=True,
            offvalue=False,
            fg_color=COLORS["bg_input"],
            progress_color=COLORS["accent_success"],
            button_color=COLORS["text_primary"],
            button_hover_color=COLORS["accent_primary"]
        )
        switch.pack(side="right")
        
        # Description
        desc = ctk.CTkLabel(
            frame,
            text=description,
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        desc.pack(anchor="w")
    
    def _save(self):
        """Save settings and close."""
        self.settings = {
            "temperature": self.temp_var.get(),
            "max_tokens": self.tokens_var.get(),
            "retrieval_k": self.k_var.get(),
            "chunk_size": self.chunk_var.get(),
            "memory_enabled": self.memory_var.get(),
            "streaming": self.stream_var.get()
        }
        self.on_save(self.settings)
        self.destroy()
