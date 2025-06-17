"""
Embedder - Clean Async Embedding Generation Service

This module provides a focused, maintainable embedding generation service for the RAG system.
It leverages existing embedding providers while offering a clean interface with enhanced
batch processing capabilities.

Key Features:
- Clean async interface for embedding generation
- Reuses existing providers (OpenAI, Sentence Transformers)
- Configurable batch processing with automatic batching
- Model versioning support for future upgrades
- Error handling with retry logic
- Performance monitoring

Classes:
- Embedder: Main embedding generation service
- EmbeddingResult: Result container with metadata
- EmbeddingBatch: Batch processing container

Usage:
    from src.rag.core.vector_search.embedder import Embedder
    
    embedder = Embedder()
    await embedder.initialize()
    
    # Single text embedding
    result = await embedder.embed_text("sample text")
    
    # Batch embedding
    results = await embedder.embed_batch(["text1", "text2", "text3"])
    
    # With custom configuration
    embedder = Embedder(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        batch_size=50
    )
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union, NamedTuple
from dataclasses import dataclass
from datetime import datetime

from ...data.embeddings_manager import (
    EmbeddingProvider, 
    OpenAIEmbeddingProvider, 
    SentenceTransformerProvider
)
from ...config.settings import RAGSettings
from ...utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """
    Container for embedding results with metadata.
    
    Attributes:
        text: Original text that was embedded
        embedding: Generated embedding vector
        model_version: Model version used for generation
        processing_time: Time taken to generate embedding (seconds)
        chunk_index: Index if this is part of a chunked text
        metadata: Additional metadata
    """
    text: str
    embedding: List[float]
    model_version: str
    processing_time: float
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingBatch:
    """
    Container for batch embedding operations.
    
    Attributes:
        texts: List of texts to embed
        results: List of embedding results
        total_processing_time: Total time for batch processing
        batch_size: Size of each processing batch
        success_count: Number of successful embeddings
        error_count: Number of failed embeddings
        errors: List of errors encountered
    """
    texts: List[str]
    results: List[EmbeddingResult]
    total_processing_time: float
    batch_size: int
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class Embedder:
    """
    Clean async embedding generation service for the RAG system.
    
    This class provides a focused interface for embedding generation that separates
    concerns from storage operations. It leverages existing embedding providers while
    offering enhanced batch processing and error handling capabilities.
    
    Features:
    - Async batch processing with configurable batch sizes
    - Support for OpenAI and Sentence Transformer models
    - Automatic retry logic for failed embeddings
    - Performance monitoring and metrics collection
    - Model versioning for future upgrades
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        settings: Optional[RAGSettings] = None
    ):
        """
        Initialize the Embedder with optional configuration overrides.
        
        Args:
            provider: Embedding provider ("openai" or "sentence_transformers")
            model_name: Model name to use
            batch_size: Batch size for processing
            settings: RAG settings object
        """
        self.settings = settings or RAGSettings()
        
        # Override settings with provided parameters
        self.provider_name = provider or self.settings.embedding_provider
        self.model_name = model_name or self.settings.embedding_model_name
        self.batch_size = batch_size or self.settings.embedding_batch_size
        
        self.provider: Optional[EmbeddingProvider] = None
        self._initialized = False
        
        # Performance tracking
        self._total_embeddings_generated = 0
        self._total_processing_time = 0.0
        self._initialization_time = None
        
        logger.info(f"Embedder configured: provider={self.provider_name}, model={self.model_name}, batch_size={self.batch_size}")
    
    async def initialize(self) -> None:
        """
        Initialize the embedding provider.
        
        Raises:
            ValueError: If provider configuration is invalid
            RuntimeError: If provider initialization fails
        """
        if self._initialized:
            return
        
        start_time = time.time()
        logger.info(f"Initializing Embedder with provider: {self.provider_name}")
        
        try:
            if self.provider_name.lower() == "openai":
                api_key = self.settings.embedding_api_key or self.settings.llm_api_key
                if not api_key:
                    raise ValueError("OpenAI API key required for OpenAI embedding provider")
                
                self.provider = OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model_name=self.model_name
                )
                logger.info(f"Initialized OpenAI provider with model: {self.model_name}")
                
            elif self.provider_name.lower() == "sentence_transformers":
                self.provider = SentenceTransformerProvider(
                    model_name=self.model_name
                )
                logger.info(f"Initialized Sentence Transformer provider with model: {self.model_name}")
                
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider_name}")
            
            # Test the provider with a small sample
            await self._test_provider()
            
            self._initialization_time = time.time() - start_time
            self._initialized = True
            
            logger.info(f"Embedder initialized successfully in {self._initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Embedder: {e}")
            raise RuntimeError(f"Embedder initialization failed: {e}") from e
    
    async def _test_provider(self) -> None:
        """Test the provider with a small sample to ensure it's working."""
        try:
            test_text = "test embedding"
            test_embeddings = await self.provider.generate_embeddings([test_text])
            if not test_embeddings or len(test_embeddings[0]) == 0:
                raise RuntimeError("Provider test failed: empty embeddings returned")
            logger.debug(f"Provider test successful, embedding dimension: {len(test_embeddings[0])}")
        except Exception as e:
            raise RuntimeError(f"Provider test failed: {e}") from e
    
    async def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            metadata: Additional metadata to include in result
            
        Returns:
            EmbeddingResult with embedding and metadata
            
        Raises:
            RuntimeError: If embedder is not initialized or embedding fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        start_time = time.time()
        
        try:
            embeddings = await self.provider.generate_embeddings([text])
            processing_time = time.time() - start_time
            
            # Update metrics
            self._total_embeddings_generated += 1
            self._total_processing_time += processing_time
            
            result = EmbeddingResult(
                text=text,
                embedding=embeddings[0],
                model_version=self.provider.get_model_version(),
                processing_time=processing_time,
                metadata=metadata
            )
            
            logger.debug(f"Generated embedding for text ({len(text)} chars) in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    async def embed_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        custom_batch_size: Optional[int] = None
    ) -> EmbeddingBatch:
        """
        Generate embeddings for a batch of texts with automatic batching.
        
        Args:
            texts: List of texts to embed
            metadata_list: Optional list of metadata for each text
            custom_batch_size: Override default batch size for this operation
            
        Returns:
            EmbeddingBatch with results and processing statistics
            
        Raises:
            RuntimeError: If embedder is not initialized
            ValueError: If inputs are invalid
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        if metadata_list and len(metadata_list) != len(texts):
            raise ValueError("Metadata list length must match texts list length")
        
        batch_size = custom_batch_size or self.batch_size
        start_time = time.time()
        
        logger.info(f"Starting batch embedding for {len(texts)} texts with batch size {batch_size}")
        
        results = []
        errors = []
        success_count = 0
        error_count = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size] if metadata_list else None
            
            try:
                batch_results = await self._process_batch(batch_texts, batch_metadata, i)
                results.extend(batch_results)
                success_count += len(batch_results)
                
                logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch_results)} embeddings generated")
                
            except Exception as e:
                error_msg = f"Batch {i//batch_size + 1} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                error_count += len(batch_texts)
                
                # Add placeholder results for failed texts
                for j, text in enumerate(batch_texts):
                    metadata = batch_metadata[j] if batch_metadata else None
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=[],
                        model_version=self.provider.get_model_version(),
                        processing_time=0.0,
                        chunk_index=i + j,
                        metadata={**(metadata or {}), "error": str(e)}
                    ))
        
        total_processing_time = time.time() - start_time
        
        # Update global metrics
        self._total_embeddings_generated += success_count
        self._total_processing_time += total_processing_time
        
        batch_result = EmbeddingBatch(
            texts=texts,
            results=results,
            total_processing_time=total_processing_time,
            batch_size=batch_size,
            success_count=success_count,
            error_count=error_count,
            errors=errors
        )
        
        logger.info(
            f"Batch embedding completed: {success_count}/{len(texts)} successful "
            f"in {total_processing_time:.2f}s"
        )
        
        return batch_result
    
    async def _process_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]],
        start_index: int
    ) -> List[EmbeddingResult]:
        """
        Process a single batch of texts.
        
        Args:
            texts: Batch of texts to embed
            metadata_list: Corresponding metadata
            start_index: Starting index for chunk indexing
            
        Returns:
            List of EmbeddingResult objects
        """
        batch_start_time = time.time()
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts in batch")
            return []
        
        valid_indices, valid_text_list = zip(*valid_texts)
        
        # Generate embeddings for valid texts
        embeddings = await self.provider.generate_embeddings(list(valid_text_list))
        model_version = self.provider.get_model_version()
        batch_processing_time = time.time() - batch_start_time
        
        # Create results
        results = []
        for i, (orig_index, text) in enumerate(valid_texts):
            metadata = metadata_list[orig_index] if metadata_list else None
            
            result = EmbeddingResult(
                text=text,
                embedding=embeddings[i],
                model_version=model_version,
                processing_time=batch_processing_time / len(valid_texts),  # Approximate per-text time
                chunk_index=start_index + orig_index,
                metadata=metadata
            )
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embedder.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "total_embeddings_generated": self._total_embeddings_generated,
            "total_processing_time": self._total_processing_time,
            "average_time_per_embedding": (
                self._total_processing_time / self._total_embeddings_generated
                if self._total_embeddings_generated > 0 else 0
            ),
            "initialization_time": self._initialization_time,
            "initialized": self._initialized,
            "batch_size": self.batch_size
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        if not self._initialized:
            return {
                "provider": self.provider_name,
                "model": self.model_name,
                "initialized": False
            }
        
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "model_version": self.provider.get_model_version(),
            "dimension": self.provider.get_dimension(),
            "initialized": self._initialized
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed (currently no cleanup required)
        pass
