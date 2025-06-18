"""
Vector Search Module

This module provides comprehensive vector search capabilities for the RAG system.

Main Components:
- Embedder: Async embedding generation with multiple provider support
- VectorSearchTool: LangChain-compatible semantic search tool
- Search Result Structures: Type-safe result containers with rich metadata

Usage:
    from src.rag.core.vector_search import VectorSearchTool, Embedder
    from src.rag.core.vector_search import VectorSearchResponse, VectorSearchResult
"""

# Core search functionality
from .embedder import Embedder, EmbeddingResult, EmbeddingBatch
from .vector_search_tool import VectorSearchTool, SearchParameters, VectorSearchInput

# Result data structures
from .search_result import (
    SearchMetadata,
    VectorSearchResult, 
    VectorSearchResponse,
    RelevanceCategory
)

__all__ = [
    # Core components
    "Embedder",
    "EmbeddingResult", 
    "EmbeddingBatch",
    "VectorSearchTool",
    "SearchParameters",
    "VectorSearchInput",
    
    # Result structures
    "SearchMetadata",
    "VectorSearchResult",
    "VectorSearchResponse", 
    "RelevanceCategory"
]
