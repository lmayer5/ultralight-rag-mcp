"""
Dark theme configuration for the Second Brain GUI.
Modern, clean aesthetic with accent colors.
"""

# Color Palette - Dark Mode
COLORS = {
    # Background colors
    "bg_dark": "#1a1a2e",
    "bg_medium": "#16213e", 
    "bg_light": "#0f3460",
    "bg_input": "#1f1f38",
    
    # Text colors
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0b0",
    "text_muted": "#6b6b7b",
    
    # Accent colors
    "accent_primary": "#e94560",
    "accent_secondary": "#0f4c75",
    "accent_success": "#00b894",
    "accent_warning": "#fdcb6e",
    "accent_error": "#d63031",
    
    # Message bubbles
    "bubble_user": "#0f4c75",
    "bubble_assistant": "#2d2d44",
    
    # Borders and separators
    "border": "#3d3d5c",
    "separator": "#2a2a4a",
}

# Font Configuration
FONTS = {
    "heading": ("Segoe UI", 16, "bold"),
    "body": ("Segoe UI", 12),
    "small": ("Segoe UI", 10),
    "mono": ("Consolas", 11),
    "chat": ("Segoe UI", 12),
    "button": ("Segoe UI", 11, "bold"),
}

# Dimensions
DIMENSIONS = {
    "window_width": 900,
    "window_height": 700,
    "sidebar_width": 250,
    "input_height": 50,
    "padding": 15,
    "corner_radius": 12,
    "button_height": 36,
}

# Apply theme to CustomTkinter
def configure_theme():
    """Configure CustomTkinter appearance."""
    import customtkinter as ctk
    
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
