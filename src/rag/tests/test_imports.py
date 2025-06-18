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
        from src.rag.core.vector_search.search_result import (
            SearchMetadata, VectorSearchResult, VectorSearchResponse, RelevanceCategory
        )
        print("✅ search_result imports OK")
    except Exception as e:
        print(f"❌ search_result import error: {e}")
        assert False, f"Failed to import search_result modules: {e}"
    
    # Test vector_search_tool imports
    try:
        from src.rag.core.vector_search.vector_search_tool import (
            VectorSearchTool, SearchParameters, VectorSearchInput
        )
        print("✅ vector_search_tool imports OK")
    except Exception as e:
        print(f"❌ vector_search_tool import error: {e}")
        assert False, f"Failed to import vector_search_tool: {e}"
    
    # Test package-level imports
    try:
        from src.rag.core.vector_search import VectorSearchTool as PackageVectorSearchTool
        from src.rag.core.vector_search import SearchMetadata as PackageSearchMetadata
        print("✅ Package-level imports OK")
        assert VectorSearchTool == PackageVectorSearchTool, "Package import mismatch for VectorSearchTool"
        assert SearchMetadata == PackageSearchMetadata, "Package import mismatch for SearchMetadata"
    except Exception as e:
        print(f"❌ Package import error: {e}")
        assert False, f"Failed to import from package: {e}"
    
    # Test basic instantiation
    try:
        # Test SearchMetadata
        metadata = SearchMetadata(field_name='test')
        print("✅ SearchMetadata instantiation OK")
        assert metadata.field_name == 'test', "SearchMetadata field_name not set correctly"
        assert hasattr(metadata, 'to_dict'), "SearchMetadata missing to_dict method"
        
        # Test VectorSearchTool
        tool = VectorSearchTool()
        print("✅ VectorSearchTool instantiation OK")
        assert tool.name == "vector_search", "VectorSearchTool name not set correctly"
        assert hasattr(tool, 'description'), "VectorSearchTool missing description"
        assert hasattr(tool, 'args_schema'), "VectorSearchTool missing args_schema"
        assert len(tool.description) > 50, "VectorSearchTool description too short"
        
        # Test SearchParameters
        params = SearchParameters(query="test query")
        print("✅ SearchParameters instantiation OK")
        assert params.query == "test query", "SearchParameters query not set correctly"
        assert params.max_results == 10, "SearchParameters default max_results incorrect"
        assert params.similarity_threshold == 0.75, "SearchParameters default similarity_threshold incorrect"
        
        # Test validation
        params.validate()  # Should not raise
        print("✅ SearchParameters validation OK")
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        assert False, f"Failed to instantiate objects: {e}"
    
    print("🎉 All imports and instantiations successful!")
    # All assertions passed if we reach here

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
