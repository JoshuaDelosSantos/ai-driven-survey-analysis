"""
Embeddings Manager for RAG System

This module provides async operations for managing vector embeddings in the RAG system.
It handles embedding generation, storage, retrieval, and batch processing with support
for multiple embedding providers and models.

Key Features:
- Async batch processing for efficient embedding generation
- Support for multiple embedding providers (OpenAI, Sentence Transformers)
- Configurable vector dimensions and chunking strategies
- Rich metadata storage and filtering capabilities
- Connection pooling and error handling
- Model versioning support for future upgrades

Classes:
- EmbeddingsManager: Main class for embedding operations
- EmbeddingProvider: Abstract base for embedding providers
- OpenAIEmbeddingProvider: OpenAI embedding implementation
- SentenceTransformerProvider: Local sentence transformer implementation

Usage:
    from src.rag.data.embeddings_manager import EmbeddingsManager
    
    manager = EmbeddingsManager()
    await manager.initialize()
    
    # Store embeddings
    await manager.store_embeddings(
        response_id=123,
        field_name="general_feedback",
        text_chunks=["chunk1", "chunk2"],
        metadata={"user_level": 5, "agency": "ATO"}
    )
    
    # Search embeddings
    results = await manager.search_similar(
        query_text="course feedback",
        field_name="general_feedback",
        limit=10
    )
"""

import asyncio
import logging
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

from ..config.settings import RAGSettings
from ..utils.logging_utils import get_logger


logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension for this provider."""
        pass
    
    @abstractmethod
    def get_model_version(self) -> str:
        """Get the model version identifier."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using their API."""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self._dimension = 1536 if "ada-002" in model_name else 1536  # Default, can be made dynamic
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def get_model_version(self) -> str:
        return f"openai-{self.model_name}-v1"


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence transformer embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = None
        
    async def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            # Run in thread pool to avoid blocking async event loop
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, SentenceTransformer, self.model_name
            )
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with dimension: {self._dimension}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        await self._load_model()
        
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.model.encode, texts
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Sentence transformer embedding generation failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Model not loaded yet. Call generate_embeddings first.")
        return self._dimension
    
    def get_model_version(self) -> str:
        return f"sentence-transformers-{self.model_name}-v1"


class EmbeddingsManager:
    """
    Main embeddings manager for the RAG system.
    
    Handles embedding generation, storage, and retrieval with async operations,
    connection pooling, and support for multiple embedding providers.
    """
    
    def __init__(self, settings: Optional[RAGSettings] = None):
        self.settings = settings or RAGSettings()
        self.db_pool: Optional[asyncpg.Pool] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the embeddings manager with database connection and embedding provider."""
        if self._initialized:
            return
            
        logger.info("Initializing EmbeddingsManager")
        
        # Initialize database connection pool
        await self._init_db_pool()
        
        # Initialize embedding provider
        await self._init_embedding_provider()
        
        # Verify table structure
        await self._verify_table_structure()
        
        self._initialized = True
        logger.info("EmbeddingsManager initialized successfully")
        
    async def _init_db_pool(self):
        """Initialize the database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.settings.rag_database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise
            
    async def _init_embedding_provider(self):
        """Initialize the embedding provider based on configuration."""
        provider = self.settings.embedding_provider.lower()
        
        if provider == "openai":
            api_key = self.settings.embedding_api_key or self.settings.llm_api_key
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embedding provider")
                
            self.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model_name=self.settings.embedding_model_name
            )
            logger.info(f"Initialized OpenAI embedding provider with model: {self.settings.embedding_model_name}")
            
        elif provider == "sentence_transformers":
            self.embedding_provider = SentenceTransformerProvider(
                model_name=self.settings.embedding_model_name
            )
            logger.info(f"Initialized Sentence Transformer provider with model: {self.settings.embedding_model_name}")
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
    async def _verify_table_structure(self):
        """Verify that the rag_embeddings table exists and has the correct structure."""
        async with self.db_pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'rag_embeddings'
                )
            """)
            
            if not table_exists:
                raise RuntimeError(
                    "rag_embeddings table does not exist. "
                    "Please run src/db/create_rag_embeddings_table.py first."
                )
                
            # Verify vector dimension matches configuration
            dimension_info = await conn.fetchval("""
                SELECT atttypmod 
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                WHERE c.relname = 'rag_embeddings' 
                AND a.attname = 'embedding'
            """)
            
            if dimension_info and dimension_info != self.settings.embedding_dimension:
                logger.warning(
                    f"Table vector dimension ({dimension_info}) differs from "
                    f"configured dimension ({self.settings.embedding_dimension})"
                )
                
    async def store_embeddings(
        self,
        response_id: int,
        field_name: str,
        text_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Store embeddings for text chunks from an evaluation response.
        
        Args:
            response_id: ID of the evaluation response
            field_name: Name of the field (e.g., 'general_feedback')
            text_chunks: List of text chunks to embed
            metadata: Additional metadata to store with embeddings
            
        Returns:
            List of embedding IDs that were created
        """
        if not self._initialized:
            await self.initialize()
            
        if not text_chunks:
            logger.warning(f"No text chunks provided for response_id {response_id}")
            return []
            
        logger.info(f"Storing embeddings for response_id {response_id}, field {field_name}, {len(text_chunks)} chunks")
        
        try:
            # Generate embeddings
            embeddings = await self.embedding_provider.generate_embeddings(text_chunks)
            model_version = self.embedding_provider.get_model_version()
            
            # Store in database
            embedding_ids = []
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
                        # Convert embedding list to pgvector format
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        embedding_id = await conn.fetchval("""
                            INSERT INTO rag_embeddings (
                                response_id, field_name, chunk_text, chunk_index,
                                embedding, model_version, metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (response_id, field_name, chunk_index) 
                            DO UPDATE SET
                                chunk_text = EXCLUDED.chunk_text,
                                embedding = EXCLUDED.embedding,
                                model_version = EXCLUDED.model_version,
                                metadata = EXCLUDED.metadata,
                                created_at = CURRENT_TIMESTAMP
                            RETURNING embedding_id
                        """, response_id, field_name, chunk_text, i, 
                           embedding_str, model_version, json.dumps(metadata) if metadata else None)
                        
                        embedding_ids.append(embedding_id)
                        
            logger.info(f"Successfully stored {len(embedding_ids)} embeddings")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
            
    async def search_similar(
        self,
        query_text: str,
        field_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using vector similarity.
        
        Args:
            query_text: Text to search for
            field_name: Filter by specific field name
            metadata_filter: Filter by metadata properties
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"Searching for similar text: '{query_text[:50]}...'")
        
        try:
            # Generate embedding for query
            query_embeddings = await self.embedding_provider.generate_embeddings([query_text])
            query_embedding = query_embeddings[0]
            
            # Convert query embedding to pgvector format
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build search query
            where_conditions = ["1 - (embedding <=> $1) >= $2"]
            params = [query_embedding_str, similarity_threshold]
            param_index = 3
            
            if field_name:
                where_conditions.append(f"field_name = ${param_index}")
                params.append(field_name)
                param_index += 1
                
            if metadata_filter:
                for key, value in metadata_filter.items():
                    where_conditions.append(f"metadata ->> ${param_index} = ${param_index + 1}")
                    params.extend([key, str(value)])
                    param_index += 2
                    
            where_clause = " AND ".join(where_conditions)
            
            search_query = f"""
                SELECT 
                    embedding_id,
                    response_id,
                    field_name,
                    chunk_text,
                    chunk_index,
                    model_version,
                    metadata,
                    created_at,
                    1 - (embedding <=> $1) as similarity_score
                FROM rag_embeddings
                WHERE {where_clause}
                ORDER BY similarity_score DESC
                LIMIT ${param_index}
            """
            
            params.append(limit)
            
            async with self.db_pool.acquire() as conn:
                results = await conn.fetch(search_query, *params)
                
            # Convert results to dictionaries
            search_results = []
            for row in results:
                result = {
                    'embedding_id': row['embedding_id'],
                    'response_id': row['response_id'],
                    'field_name': row['field_name'],
                    'chunk_text': row['chunk_text'],
                    'chunk_index': row['chunk_index'],
                    'model_version': row['model_version'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                    'created_at': row['created_at'],
                    'similarity_score': float(row['similarity_score'])
                }
                search_results.append(result)
                
            logger.info(f"Found {len(search_results)} similar results")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise
            
    async def delete_embeddings(
        self,
        response_id: Optional[int] = None,
        field_name: Optional[str] = None,
        embedding_ids: Optional[List[int]] = None
    ) -> int:
        """
        Delete embeddings based on various criteria.
        
        Args:
            response_id: Delete all embeddings for this response
            field_name: Delete embeddings for this field (requires response_id)
            embedding_ids: Delete specific embedding IDs
            
        Returns:
            Number of embeddings deleted
        """
        if not self._initialized:
            await self.initialize()
            
        if not any([response_id, embedding_ids]):
            raise ValueError("Must specify either response_id or embedding_ids")
            
        try:
            async with self.db_pool.acquire() as conn:
                if embedding_ids:
                    # Delete specific embeddings
                    result = await conn.execute(
                        "DELETE FROM rag_embeddings WHERE embedding_id = ANY($1)",
                        embedding_ids
                    )
                elif response_id and field_name:
                    # Delete embeddings for specific response and field
                    result = await conn.execute(
                        "DELETE FROM rag_embeddings WHERE response_id = $1 AND field_name = $2",
                        response_id, field_name
                    )
                elif response_id:
                    # Delete all embeddings for response
                    result = await conn.execute(
                        "DELETE FROM rag_embeddings WHERE response_id = $1",
                        response_id
                    )
                    
                deleted_count = int(result.split()[-1])
                logger.info(f"Deleted {deleted_count} embeddings")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
            
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_embeddings,
                        COUNT(DISTINCT response_id) as unique_responses,
                        COUNT(DISTINCT field_name) as unique_fields,
                        COUNT(DISTINCT model_version) as unique_models
                    FROM rag_embeddings
                """)
                
                field_breakdown = await conn.fetch("""
                    SELECT field_name, COUNT(*) as count
                    FROM rag_embeddings
                    GROUP BY field_name
                    ORDER BY count DESC
                """)
                
                model_breakdown = await conn.fetch("""
                    SELECT model_version, COUNT(*) as count
                    FROM rag_embeddings
                    GROUP BY model_version
                    ORDER BY count DESC
                """)
                
                return {
                    'total_embeddings': stats['total_embeddings'],
                    'unique_responses': stats['unique_responses'],
                    'unique_fields': stats['unique_fields'],
                    'unique_models': stats['unique_models'],
                    'field_breakdown': {row['field_name']: row['count'] for row in field_breakdown},
                    'model_breakdown': {row['model_version']: row['count'] for row in model_breakdown}
                }
                
        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            raise
            
    async def close(self):
        """Close database connections and clean up resources."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")
            
        self._initialized = False


# Convenience function for quick access
async def get_embeddings_manager(settings: Optional[RAGSettings] = None) -> EmbeddingsManager:
    """
    Get an initialized embeddings manager instance.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        Initialized EmbeddingsManager instance
    """
    manager = EmbeddingsManager(settings)
    await manager.initialize()
    return manager
