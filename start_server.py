#!/usr/bin/env python
"""
Cross-platform server launcher for Quantum Network Simulator
Works on Windows, Linux, and Mac
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Quantum Network Simulator - Server Launcher")
    print("=" * 60)

    # Get project root directory
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"

    # Check if backend directory exists
    if not backend_dir.exists():
        print("ERROR: backend directory not found!")
        sys.exit(1)

    # Change to project root
    os.chdir(project_root)

    # Check Python version
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)

    print(f"\nPython version: {sys.version.split()[0]} ✓")

    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("\nWARNING: Not running in virtual environment!")
        print("It's recommended to activate venv first:")
        if sys.platform == "win32":
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        print("Virtual environment: Active ✓")

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import fastapi
        import uvicorn
        import torch
        import networkx
        print("Core dependencies: Installed ✓")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e.name}")
        print("\nInstalling dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            str(backend_dir / "requirements.txt")
        ])

    # Start server
    print("\n" + "=" * 60)
    print("Starting FastAPI Server")
    print("=" * 60)
    print("\nServer URLs:")
    print("  Frontend API:    http://localhost:8000")
    print("  API Docs:        http://localhost:8000/docs")
    print("  Health Check:    http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")

    # Run uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
