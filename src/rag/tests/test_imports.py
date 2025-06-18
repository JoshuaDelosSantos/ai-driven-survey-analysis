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
    """Test that all vector search imports work correctly."""
    print("🧪 Testing all imports...")
    
    # Test search_result imports
    try:
        from src.rag.core.vector_search.search_result import SearchMetadata, VectorSearchResult, VectorSearchResponse
        print("✅ search_result imports OK")
    except Exception as e:
        print(f"❌ search_result import error: {e}")
        assert False, f"Failed to import search_result modules: {e}"
    
    # Test vector_search_tool imports
    try:
        from src.rag.core.vector_search.vector_search_tool import VectorSearchTool
        print("✅ vector_search_tool imports OK")
    except Exception as e:
        print(f"❌ vector_search_tool import error: {e}")
        assert False, f"Failed to import vector_search_tool: {e}"
    
    # Test basic instantiation
    try:
        metadata = SearchMetadata(field_name='test')
        print("✅ SearchMetadata instantiation OK")
        assert metadata.field_name == 'test', "SearchMetadata field_name not set correctly"
        
        tool = VectorSearchTool()
        print("✅ VectorSearchTool instantiation OK")
        assert tool.name == "vector_search", "VectorSearchTool name not set correctly"
        assert hasattr(tool, 'description'), "VectorSearchTool missing description"
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        assert False, f"Failed to instantiate objects: {e}"
    
    print("🎉 All imports and instantiations successful!")
    # No return statement needed for pytest

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
