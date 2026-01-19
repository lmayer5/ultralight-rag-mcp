"""
Build script to create standalone .exe for Second Brain.
Uses PyInstaller to package the application.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Build the executable."""
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    dist_dir = project_root / "dist"
    
    print("=" * 60)
    print("🏗️  Building Second Brain Desktop Application")
    print("=" * 60)
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "SecondBrain",
        "--onefile",           # Single executable
        "--windowed",          # No console window
        "--noconfirm",         # Overwrite without asking
        
        # Add data files
        "--add-data", f"{project_root / 'config'};config",
        "--add-data", f"{project_root / 'prompts'};prompts",
        
        # Hidden imports for dependencies
        "--hidden-import", "customtkinter",
        "--hidden-import", "sentence_transformers",
        "--hidden-import", "faiss",
        "--hidden-import", "langchain",
        "--hidden-import", "langchain_community",
        "--hidden-import", "langchain_huggingface",
        "--hidden-import", "PIL",
        "--hidden-import", "requests",
        "--hidden-import", "pydantic",
        
        # Exclude unnecessary modules
        "--exclude-module", "matplotlib",
        "--exclude-module", "numpy.testing",
        "--exclude-module", "scipy",
        
        # Output directory
        "--distpath", str(dist_dir),
        "--workpath", str(project_root / "build"),
        "--specpath", str(project_root),
        
        # Entry point
        str(src_dir / "gui_app.py")
    ]
    
    print("\n📦 Running PyInstaller...")
    print(f"   Entry point: {src_dir / 'gui_app.py'}")
    print(f"   Output: {dist_dir / 'SecondBrain.exe'}")
    print()
    
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode == 0:
        exe_path = dist_dir / "SecondBrain.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print()
            print("=" * 60)
            print(f"✅ Build successful!")
            print(f"   Output: {exe_path}")
            print(f"   Size: {size_mb:.1f} MB")
            print("=" * 60)
        else:
            print("❌ Build completed but .exe not found")
            return 1
    else:
        print("❌ Build failed!")
        return result.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
