"""
Vector Search Module for RAG System

This module provides vector search capabilities including embedding generation,
text chunking, semantic retrieval, and vector indexing for the RAG system.

Current Implementation:
- embedder.py: Clean async embedding generation interface
- vector_search_tool.py: LangChain-compatible vector search tool
- search_result.py: Result data structures and containers

Planned Components:
- chunk_processor.py: Text chunking strategies  
- retriever.py: Semantic retrieval
- indexer.py: Vector indexing management

Classes:
- Embedder: Main embedding generation service
- VectorSearchTool: Privacy-compliant semantic search tool
- VectorSearchResult: Individual search result with metadata
- VectorSearchResponse: Complete search response container
- SearchMetadata: Rich metadata container for analysis
"""

from .embedder import Embedder
from .vector_search_tool import VectorSearchTool
from .search_result import (
    VectorSearchResult,
    VectorSearchResponse, 
    SearchMetadata,
    RelevanceCategory
)

__all__ = [
    "Embedder"
]
