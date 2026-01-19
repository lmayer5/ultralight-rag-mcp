"""
Chat panel component - displays conversation and handles input.
"""

import customtkinter as ctk
from datetime import datetime
from typing import Callable, Optional
from .theme import COLORS, FONTS, DIMENSIONS


class MessageBubble(ctk.CTkFrame):
    """A single message bubble in the chat."""
    
    def __init__(self, parent, message: str, is_user: bool = False, timestamp: str = None):
        bg_color = COLORS["bubble_user"] if is_user else COLORS["bubble_assistant"]
        super().__init__(parent, fg_color=bg_color, corner_radius=12)
        
        # Icon
        icon = "👤" if is_user else "🤖"
        role = "You" if is_user else "Assistant"
        
        # Header with icon and timestamp
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 0))
        
        icon_label = ctk.CTkLabel(
            header, 
            text=f"{icon} {role}",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"]
        )
        icon_label.pack(side="left")
        
        if timestamp:
            time_label = ctk.CTkLabel(
                header,
                text=timestamp,
                font=FONTS["small"],
                text_color=COLORS["text_muted"]
            )
            time_label.pack(side="right")
        
        # Message content
        content = ctk.CTkLabel(
            self,
            text=message,
            font=FONTS["chat"],
            text_color=COLORS["text_primary"],
            wraplength=500,
            justify="left",
            anchor="w"
        )
        content.pack(fill="x", padx=12, pady=(4, 10))


class StreamingBubble(ctk.CTkFrame):
    """A message bubble that supports streaming text updates."""
    
    def __init__(self, parent):
        super().__init__(parent, fg_color=COLORS["bubble_assistant"], corner_radius=12)
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 0))
        
        icon_label = ctk.CTkLabel(
            header,
            text="🤖 Assistant",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"]
        )
        icon_label.pack(side="left")
        
        # Message content (will be updated)
        self.content_label = ctk.CTkLabel(
            self,
            text="⏳ Thinking...",
            font=FONTS["chat"],
            text_color=COLORS["text_primary"],
            wraplength=500,
            justify="left",
            anchor="w"
        )
        self.content_label.pack(fill="x", padx=12, pady=(4, 10))
        
        self._full_text = ""
    
    def update_text(self, text: str):
        """Update the displayed text (for streaming)."""
        self._full_text = text
        self.content_label.configure(text=text if text else "⏳ Thinking...")
    
    def append_text(self, chunk: str):
        """Append text chunk (for streaming)."""
        self._full_text += chunk
        self.content_label.configure(text=self._full_text)
    
    def get_text(self) -> str:
        return self._full_text


class ChatPanel(ctk.CTkFrame):
    """Main chat panel with message history and input."""
    
    def __init__(self, parent, on_send: Callable[[str], None]):
        super().__init__(parent, fg_color=COLORS["bg_dark"])
        self.on_send = on_send
        self._streaming_bubble: Optional[StreamingBubble] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Scrollable message area
        self.messages_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["bg_dark"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent_secondary"]
        )
        self.messages_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input area
        input_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_medium"], height=60)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        input_frame.pack_propagate(False)
        
        # Text input
        self.input_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message...",
            font=FONTS["body"],
            height=40,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"]
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.input_entry.bind("<Return>", self._handle_send)
        
        # Send button
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="➤",
            width=50,
            height=40,
            font=FONTS["button"],
            fg_color=COLORS["accent_primary"],
            hover_color=COLORS["accent_secondary"],
            command=self._handle_send
        )
        self.send_btn.pack(side="right", padx=(0, 10), pady=10)
    
    def _handle_send(self, event=None):
        """Handle send button click or Enter key."""
        text = self.input_entry.get().strip()
        if text:
            self.input_entry.delete(0, "end")
            self.add_user_message(text)
            self.on_send(text)
    
    def add_user_message(self, message: str):
        """Add a user message to the chat."""
        timestamp = datetime.now().strftime("%H:%M")
        bubble = MessageBubble(
            self.messages_frame, 
            message, 
            is_user=True, 
            timestamp=timestamp
        )
        bubble.pack(fill="x", pady=5, padx=(50, 5), anchor="e")
        self._scroll_to_bottom()
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the chat."""
        timestamp = datetime.now().strftime("%H:%M")
        bubble = MessageBubble(
            self.messages_frame,
            message,
            is_user=False,
            timestamp=timestamp
        )
        bubble.pack(fill="x", pady=5, padx=(5, 50), anchor="w")
        self._scroll_to_bottom()
    
    def start_streaming(self) -> StreamingBubble:
        """Start a streaming response bubble."""
        self._streaming_bubble = StreamingBubble(self.messages_frame)
        self._streaming_bubble.pack(fill="x", pady=5, padx=(5, 50), anchor="w")
        self._scroll_to_bottom()
        return self._streaming_bubble
    
    def finish_streaming(self):
        """Finish streaming and convert to regular message."""
        if self._streaming_bubble:
            self._streaming_bubble = None
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        self.messages_frame._parent_canvas.yview_moveto(1.0)
    
    def set_input_enabled(self, enabled: bool):
        """Enable or disable input."""
        state = "normal" if enabled else "disabled"
        self.input_entry.configure(state=state)
        self.send_btn.configure(state=state)
    
    def clear_messages(self):
        """Clear all messages."""
        for widget in self.messages_frame.winfo_children():
            widget.destroy()
