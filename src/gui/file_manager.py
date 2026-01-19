"""
File manager component - handles document upload and display.
"""

import customtkinter as ctk
from pathlib import Path
from typing import Callable, List
from tkinter import filedialog
from .theme import COLORS, FONTS


class FileManager(ctk.CTkFrame):
    """Panel for uploading and managing documents."""
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
    
    def __init__(self, parent, on_files_added: Callable[[List[Path]], None]):
        super().__init__(parent, fg_color=COLORS["bg_medium"])
        self.on_files_added = on_files_added
        self._documents: List[Path] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        title = ctk.CTkLabel(
            header,
            text="📚 Documents",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        )
        title.pack(side="left")
        
        # Document count
        self.count_label = ctk.CTkLabel(
            header,
            text="(0)",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.count_label.pack(side="left", padx=5)
        
        # Upload button
        upload_btn = ctk.CTkButton(
            self,
            text="📎 Upload Files",
            font=FONTS["button"],
            fg_color=COLORS["accent_secondary"],
            hover_color=COLORS["accent_primary"],
            height=36,
            command=self._open_file_dialog
        )
        upload_btn.pack(fill="x", padx=10, pady=5)
        
        # Drag and drop area
        self.drop_zone = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg_input"],
            border_width=2,
            border_color=COLORS["border"],
            corner_radius=8,
            height=60
        )
        self.drop_zone.pack(fill="x", padx=10, pady=5)
        self.drop_zone.pack_propagate(False)
        
        drop_label = ctk.CTkLabel(
            self.drop_zone,
            text="Drag & drop files here\n(.md, .txt, .pdf)",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        drop_label.pack(expand=True)
        
        # File list (scrollable)
        self.file_list = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["bg_dark"],
            height=150
        )
        self.file_list.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="No documents loaded",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.status_label.pack(pady=(0, 10))
    
    def _open_file_dialog(self):
        """Open file selection dialog."""
        filetypes = [
            ("Supported files", "*.txt *.md *.pdf"),
            ("Text files", "*.txt"),
            ("Markdown files", "*.md"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=filetypes
        )
        
        if files:
            paths = [Path(f) for f in files]
            self.add_files(paths)
    
    def add_files(self, files: List[Path]):
        """Add files to the document list."""
        added = []
        for f in files:
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                if f not in self._documents:
                    self._documents.append(f)
                    added.append(f)
                    self._add_file_item(f)
        
        if added:
            self._update_count()
            self.on_files_added(added)
            self.status_label.configure(
                text=f"Added {len(added)} file(s)",
                text_color=COLORS["accent_success"]
            )
        else:
            self.status_label.configure(
                text="No valid files to add",
                text_color=COLORS["accent_warning"]
            )
    
    def _add_file_item(self, path: Path):
        """Add a file item to the list."""
        item = ctk.CTkFrame(self.file_list, fg_color=COLORS["bg_medium"], corner_radius=6)
        item.pack(fill="x", pady=2)
        
        # File icon based on type
        icons = {".txt": "📄", ".md": "📝", ".pdf": "📕"}
        icon = icons.get(path.suffix.lower(), "📄")
        
        # File name
        name_label = ctk.CTkLabel(
            item,
            text=f"{icon} {path.name}",
            font=FONTS["small"],
            text_color=COLORS["text_primary"],
            anchor="w"
        )
        name_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)
        
        # Size
        try:
            size = path.stat().st_size
            size_str = self._format_size(size)
        except:
            size_str = "?"
        
        size_label = ctk.CTkLabel(
            item,
            text=size_str,
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        size_label.pack(side="right", padx=10, pady=5)
        
        # Remove button
        remove_btn = ctk.CTkButton(
            item,
            text="×",
            width=24,
            height=24,
            font=FONTS["small"],
            fg_color="transparent",
            hover_color=COLORS["accent_error"],
            command=lambda p=path, i=item: self._remove_file(p, i)
        )
        remove_btn.pack(side="right", padx=2, pady=5)
    
    def _remove_file(self, path: Path, item):
        """Remove a file from the list."""
        if path in self._documents:
            self._documents.remove(path)
            item.destroy()
            self._update_count()
    
    def _update_count(self):
        """Update the document count label."""
        count = len(self._documents)
        self.count_label.configure(text=f"({count})")
        if count == 0:
            self.status_label.configure(
                text="No documents loaded",
                text_color=COLORS["text_muted"]
            )
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def get_documents(self) -> List[Path]:
        """Get list of loaded documents."""
        return self._documents.copy()
    
    def clear_documents(self):
        """Clear all loaded documents."""
        self._documents.clear()
        for widget in self.file_list.winfo_children():
            widget.destroy()
        self._update_count()
