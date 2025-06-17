"""
Vector Search Module for RAG System

This module provides vector search capabilities including embedding generation,
text chunking, semantic retrieval, and vector indexing for the RAG system.

Current Implementation:
- embedder.py: Clean async embedding generation interface

Planned Components:
- chunk_processor.py: Text chunking strategies
- retriever.py: Semantic retrieval
- indexer.py: Vector indexing management

Classes:
- Embedder: Main embedding generation service
"""

from .embedder import Embedder

__all__ = [
    "Embedder"
]
