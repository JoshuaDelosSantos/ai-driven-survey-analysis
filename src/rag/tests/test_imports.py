#!/usr/bin/env python3
"""
Quick test script to verify imports work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    print("🧪 Testing all imports...")
    
    try:
        from src.rag.core.vector_search.search_result import SearchMetadata, VectorSearchResult, VectorSearchResponse
        print("✅ search_result imports OK")
    except Exception as e:
        print(f"❌ search_result import error: {e}")
        return False
    
    try:
        from src.rag.core.vector_search.vector_search_tool import VectorSearchTool
        print("✅ vector_search_tool imports OK")
    except Exception as e:
        print(f"❌ vector_search_tool import error: {e}")
        return False
    
    try:
        # Test basic instantiation
        metadata = SearchMetadata(field_name='test')
        print("✅ SearchMetadata instantiation OK")
        
        tool = VectorSearchTool()
        print("✅ VectorSearchTool instantiation OK")
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False
    
    print("🎉 All imports and instantiations successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
