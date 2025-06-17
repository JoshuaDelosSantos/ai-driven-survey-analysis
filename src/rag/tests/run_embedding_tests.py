#!/usr/bin/env python3
"""
Simple test runner for embedding functionality tests.

This script provides a convenient way to run the embedding tests
with proper environment setup and clear output formatting.

Usage:
    python run_embedding_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_embedding_tests():
    """Run the embedding tests with pytest."""
    print("🧪 Running RAG Embedding Tests with all-MiniLM-L6-v2")
    print("=" * 60)
    
    # Get the test file path
    test_file = Path(__file__).parent / "test_embeddings_manager.py"
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_file),
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        
        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("✅ All embedding tests passed!")
            print("🎉 Local sentence transformer model is working correctly.")
        else:
            print("❌ Some embedding tests failed.")
            print("📝 Check the output above for details.")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 Failed to run tests: {e}")
        return False

if __name__ == "__main__":
    success = run_embedding_tests()
    sys.exit(0 if success else 1)
