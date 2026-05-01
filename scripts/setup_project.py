"""
setup_project.py — Initial setup utility for SentinelIQ.

Ensures the directory structure is in place, checks for required
dependencies, and verifies that the Ollama server is reachable.
"""

import os
import sys
import requests
from pathlib import Path

# Project directories to ensure exist
DIRECTORIES = [
    "data/synthetic",
    "data/models",
    "data/graphs",
    "data/processed",
    "data/raw",
    "data/chroma",
    "logs",
]

def check_ollama():
    """Verify that Ollama is running and accessible."""
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running and reachable.")
            return True
        else:
            print(f"⚠️  Ollama server returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is not running. Please start Ollama before continuing.")
    return False

def setup_dirs():
    """Create project directories if they don't exist."""
    project_root = Path(__file__).parent.parent
    print(f"Setting up SentinelIQ in: {project_root}\n")
    
    for dir_path in DIRECTORIES:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory exists: {dir_path}")

def check_env():
    """Verify that .env exists."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        example_file = project_root / ".env.example"
        if example_file.exists():
            with open(example_file, "r") as src, open(env_file, "w") as dst:
                dst.write(src.read())
            print("✅ .env created. Please review and update your API keys/model names.")
        else:
            print("❌ .env.example not found. Please create a .env file manually.")
    else:
        print("✅ .env file exists.")

def main():
    print("=== SentinelIQ Project Setup ===\n")
    setup_dirs()
    print("")
    check_env()
    print("")
    check_ollama()
    print("\nSetup complete! Next steps:")
    print("1. Update your .env file.")
    print("2. Run: python scripts/generate_data.py")
    print("3. Run: python scripts/train_model.py")
    print("4. Run: python scripts/ingest_and_run.py --embed-cases")

if __name__ == "__main__":
    main()
