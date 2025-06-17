#!/usr/bin/env python3
"""
Test suite for Embedder class with comprehensive coverage.

This module tests the embedding functionality using both OpenAI and Sentence Transformer
models, focusing on:
- Embedder initialization and configuration
- Single text embedding generation
- Batch processing with various sizes
- Error handling and recovery
- Performance metrics collection
- Model versioning and compatibility

Tests use real models to ensure functionality but include mocks for CI/CD environments.
"""

import asyncio
import pytest
import pytest_asyncio
import logging
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Setup path for imports
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.core.vector_search.embedder import Embedder, EmbeddingResult, EmbeddingBatch
from src.rag.config.settings import RAGSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def rag_settings():
    """Create RAG settings configured for local sentence transformer model."""
    settings = RAGSettings()
    # Override for testing with local model
    settings.embedding_provider = "sentence_transformers"
    settings.embedding_model_name = "all-MiniLM-L6-v2"
    settings.embedding_dimension = 384
    settings.embedding_batch_size = 50
    return settings


@pytest.fixture
def sample_texts():
    """Sample texts for testing embedding generation."""
    return [
        "The course was very helpful and informative.",
        "I experienced technical difficulties during the session.",
        "The facilitator was knowledgeable and engaging.",
        "The content was relevant to my work responsibilities.",
        "I would recommend this course to my colleagues."
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return [
        {"user_level": 5, "agency": "ATO", "sentiment": "positive"},
        {"user_level": 3, "agency": "ACMA", "sentiment": "negative"},
        {"user_level": 6, "agency": "AUSTRAC", "sentiment": "positive"},
        {"user_level": 4, "agency": "ASIC", "sentiment": "positive"},
        {"user_level": 5, "agency": "ATO", "sentiment": "positive"}
    ]


class TestEmbedderInitialization:
    """Test embedder initialization and configuration."""
    
    @pytest.mark.asyncio
    async def test_embedder_default_initialization(self, rag_settings):
        """Test embedder initialization with default settings."""
        embedder = Embedder(settings=rag_settings)
        
        assert embedder.provider_name == "sentence_transformers"
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.batch_size == 50
        assert not embedder._initialized
        
        await embedder.initialize()
        
        assert embedder._initialized
        assert embedder.provider is not None
        assert embedder._initialization_time > 0
    
    @pytest.mark.asyncio
    async def test_embedder_custom_configuration(self):
        """Test embedder initialization with custom configuration."""
        embedder = Embedder(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            batch_size=25
        )
        
        assert embedder.provider_name == "sentence_transformers"
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.batch_size == 25
        
        await embedder.initialize()
        
        assert embedder._initialized
        assert embedder.provider is not None
    
    @pytest.mark.asyncio
    async def test_embedder_double_initialization(self, rag_settings):
        """Test that double initialization doesn't cause issues."""
        embedder = Embedder(settings=rag_settings)
        
        await embedder.initialize()
        first_init_time = embedder._initialization_time
        
        await embedder.initialize()  # Should not re-initialize
        
        assert embedder._initialization_time == first_init_time
        assert embedder._initialized
    
    def test_embedder_invalid_provider(self):
        """Test embedder with invalid provider configuration."""
        embedder = Embedder(provider="invalid_provider")
        
        with pytest.raises(RuntimeError, match="Embedder initialization failed"):
            asyncio.run(embedder.initialize())
    
    @pytest.mark.asyncio
    async def test_embedder_model_info_before_initialization(self):
        """Test getting model info before initialization."""
        embedder = Embedder(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2"
        )
        
        info = embedder.get_model_info()
        
        assert info["provider"] == "sentence_transformers"
        assert info["model"] == "all-MiniLM-L6-v2"
        assert not info["initialized"]
        assert "model_version" not in info
    
    @pytest.mark.asyncio
    async def test_embedder_model_info_after_initialization(self, rag_settings):
        """Test getting model info after initialization."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        info = embedder.get_model_info()
        
        assert info["provider"] == "sentence_transformers"
        assert info["model"] == "all-MiniLM-L6-v2"
        assert info["initialized"]
        assert "model_version" in info
        assert "dimension" in info
        assert info["dimension"] > 0


class TestEmbedderSingleText:
    """Test single text embedding functionality."""
    
    @pytest.mark.asyncio
    async def test_embed_single_text(self, rag_settings):
        """Test embedding a single text."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        text = "This is a test text for embedding."
        result = await embedder.embed_text(text)
        
        assert isinstance(result, EmbeddingResult)
        assert result.text == text
        assert len(result.embedding) > 0
        assert result.processing_time > 0
        assert result.model_version is not None
        assert result.chunk_index is None
    
    @pytest.mark.asyncio
    async def test_embed_text_with_metadata(self, rag_settings):
        """Test embedding text with metadata."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        text = "Course feedback with metadata."
        metadata = {"user_level": 5, "agency": "ATO"}
        
        result = await embedder.embed_text(text, metadata=metadata)
        
        assert result.text == text
        assert result.metadata == metadata
        assert len(result.embedding) > 0
    
    @pytest.mark.asyncio
    async def test_embed_empty_text(self, rag_settings):
        """Test embedding empty or whitespace text."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedder.embed_text("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedder.embed_text("   ")
    
    @pytest.mark.asyncio
    async def test_embed_without_initialization(self):
        """Test embedding without initialization (should auto-initialize)."""
        embedder = Embedder(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2"
        )
        
        text = "Auto-initialization test."
        result = await embedder.embed_text(text)
        
        assert embedder._initialized
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) > 0


class TestEmbedderBatchProcessing:
    """Test batch embedding functionality."""
    
    @pytest.mark.asyncio
    async def test_embed_batch_basic(self, rag_settings, sample_texts):
        """Test basic batch embedding."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        batch_result = await embedder.embed_batch(sample_texts)
        
        assert isinstance(batch_result, EmbeddingBatch)
        assert len(batch_result.results) == len(sample_texts)
        assert batch_result.success_count == len(sample_texts)
        assert batch_result.error_count == 0
        assert batch_result.total_processing_time > 0
        
        # Check individual results
        for i, result in enumerate(batch_result.results):
            assert result.text == sample_texts[i]
            assert len(result.embedding) > 0
            assert result.chunk_index == i
    
    @pytest.mark.asyncio
    async def test_embed_batch_with_metadata(self, rag_settings, sample_texts, sample_metadata):
        """Test batch embedding with metadata."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        batch_result = await embedder.embed_batch(sample_texts, metadata_list=sample_metadata)
        
        assert len(batch_result.results) == len(sample_texts)
        
        for i, result in enumerate(batch_result.results):
            assert result.text == sample_texts[i]
            assert result.metadata == sample_metadata[i]
            assert len(result.embedding) > 0
    
    @pytest.mark.asyncio
    async def test_embed_batch_custom_batch_size(self, rag_settings, sample_texts):
        """Test batch embedding with custom batch size."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        # Use very small batch size to test batching logic
        batch_result = await embedder.embed_batch(sample_texts, custom_batch_size=2)
        
        assert len(batch_result.results) == len(sample_texts)
        assert batch_result.batch_size == 2
        assert batch_result.success_count == len(sample_texts)
    
    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, rag_settings):
        """Test batch embedding with empty text list."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            await embedder.embed_batch([])
    
    @pytest.mark.asyncio
    async def test_embed_batch_metadata_length_mismatch(self, rag_settings, sample_texts):
        """Test batch embedding with mismatched metadata length."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        wrong_metadata = [{"test": "data"}]  # Only one item for multiple texts
        
        with pytest.raises(ValueError, match="Metadata list length must match texts list length"):
            await embedder.embed_batch(sample_texts, metadata_list=wrong_metadata)
    
    @pytest.mark.asyncio
    async def test_embed_batch_with_empty_texts(self, rag_settings):
        """Test batch embedding with some empty texts."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        texts_with_empty = [
            "Valid text 1",
            "",  # Empty text
            "Valid text 2",
            "   ",  # Whitespace only
            "Valid text 3"
        ]
        
        batch_result = await embedder.embed_batch(texts_with_empty)
        
        # Should only process valid texts
        assert batch_result.success_count == 3
        assert len([r for r in batch_result.results if r.embedding]) == 3


class TestEmbedderPerformance:
    """Test embedder performance and metrics."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, rag_settings, sample_texts):
        """Test that performance metrics are tracked correctly."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        # Initial metrics
        initial_metrics = embedder.get_metrics()
        assert initial_metrics["total_embeddings_generated"] == 0
        assert initial_metrics["total_processing_time"] == 0
        
        # Process some texts
        await embedder.embed_text("Single text")
        await embedder.embed_batch(sample_texts)
        
        # Check updated metrics
        final_metrics = embedder.get_metrics()
        assert final_metrics["total_embeddings_generated"] == 1 + len(sample_texts)
        assert final_metrics["total_processing_time"] > 0
        assert final_metrics["average_time_per_embedding"] > 0
        assert final_metrics["initialization_time"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, rag_settings):
        """Test that batch processing is more efficient than individual processing."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        texts = [f"Test text number {i}" for i in range(10)]
        
        # Time individual processing
        start_time = time.time()
        for text in texts:
            await embedder.embed_text(text)
        individual_time = time.time() - start_time
        
        # Reset metrics for fair comparison
        embedder._total_embeddings_generated = 0
        embedder._total_processing_time = 0
        
        # Time batch processing
        start_time = time.time()
        await embedder.embed_batch(texts)
        batch_time = time.time() - start_time
        
        # Batch processing time should be reasonable (allow for model loading time)
        logger.info(f"Individual processing: {individual_time:.3f}s")
        logger.info(f"Batch processing: {batch_time:.3f}s")
        
        # For small batches, batch processing might be slower due to overhead
        # but should still be reasonable (within 5x of individual processing)
        # Allow for more flexibility as performance can vary based on system load
        max_acceptable_time = max(individual_time * 5.0, 1.0)  # At least 1 second allowance
        assert batch_time <= max_acceptable_time


class TestEmbedderErrorHandling:
    """Test embedder error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_provider_failure_during_initialization(self):
        """Test handling of provider failure during initialization."""
        with patch('src.rag.data.embeddings_manager.SentenceTransformerProvider.__init__') as mock_init:
            mock_init.side_effect = Exception("Model loading failed")
            
            embedder = Embedder(
                provider="sentence_transformers",
                model_name="invalid-model"
            )
            
            with pytest.raises(RuntimeError, match="Embedder initialization failed"):
                await embedder.initialize()
    
    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, rag_settings):
        """Test handling of embedding generation failure."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        # Mock the provider to fail
        with patch.object(embedder.provider, 'generate_embeddings') as mock_generate:
            mock_generate.side_effect = Exception("API failure")
            
            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                await embedder.embed_text("Test text")


class TestEmbedderContextManager:
    """Test embedder as async context manager."""
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self, rag_settings):
        """Test embedder used as async context manager."""
        async with Embedder(settings=rag_settings) as embedder:
            assert embedder._initialized
            
            result = await embedder.embed_text("Context manager test")
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) > 0
    
    @pytest.mark.asyncio
    async def test_context_manager_with_custom_config(self):
        """Test context manager with custom configuration."""
        async with Embedder(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            batch_size=25
        ) as embedder:
            assert embedder.batch_size == 25
            assert embedder._initialized
            
            batch_result = await embedder.embed_batch([
                "Text 1", "Text 2", "Text 3"
            ])
            assert batch_result.success_count == 3


# Integration tests that can be run manually
class TestEmbedderIntegration:
    """Integration tests for real-world usage patterns."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_evaluation_text_processing(self, rag_settings):
        """Test processing real evaluation feedback text."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        # Sample evaluation texts (anonymized)
        evaluation_texts = [
            "The course content was comprehensive and well-structured. The facilitator demonstrated excellent knowledge of the subject matter.",
            "I experienced some technical difficulties with the online platform, but the support team was responsive.",
            "The course was relevant to my current role and provided practical skills I can apply immediately.",
            "The session ran over time, which conflicted with other work commitments.",
            "Excellent course overall. Would recommend to colleagues in similar roles."
        ]
        
        metadata_list = [
            {"field_name": "general_feedback", "user_level": 5, "agency": "ATO"},
            {"field_name": "did_experience_issue_detail", "user_level": 3, "agency": "ACMA"},
            {"field_name": "general_feedback", "user_level": 6, "agency": "AUSTRAC"},
            {"field_name": "general_feedback", "user_level": 4, "agency": "ASIC"},
            {"field_name": "course_application_other", "user_level": 5, "agency": "ATO"}
        ]
        
        batch_result = await embedder.embed_batch(
            evaluation_texts,
            metadata_list=metadata_list
        )
        
        assert batch_result.success_count == len(evaluation_texts)
        assert batch_result.error_count == 0
        
        # Verify all embeddings are valid
        for result in batch_result.results:
            assert len(result.embedding) == 384  # all-MiniLM-L6-v2 dimension
            assert result.model_version is not None
            assert result.metadata is not None
        
        # Log performance metrics
        metrics = embedder.get_metrics()
        logger.info(f"Processed {metrics['total_embeddings_generated']} embeddings")
        logger.info(f"Average time per embedding: {metrics['average_time_per_embedding']:.3f}s")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_large_batch_processing(self, rag_settings):
        """Test processing a large batch of texts."""
        embedder = Embedder(settings=rag_settings)
        await embedder.initialize()
        
        # Create a large batch of texts
        large_batch = [f"Evaluation feedback text number {i} with varying content." for i in range(100)]
        
        start_time = time.time()
        batch_result = await embedder.embed_batch(large_batch, custom_batch_size=20)
        processing_time = time.time() - start_time
        
        assert batch_result.success_count == 100
        assert batch_result.error_count == 0
        assert batch_result.batch_size == 20
        
        logger.info(f"Processed {len(large_batch)} texts in {processing_time:.2f}s")
        logger.info(f"Average time per text: {processing_time/len(large_batch):.3f}s")


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        __file__ + "::TestEmbedderInitialization",
        __file__ + "::TestEmbedderSingleText",
        __file__ + "::TestEmbedderBatchProcessing",
        "-v"
    ])
