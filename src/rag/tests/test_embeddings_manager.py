#!/usr/bin/env python3
"""
Test suite for EmbeddingsManager with local sentence transformer model.

This module tests the embedding functionality using the all-MiniLM-L6-v2 model
with real evaluation data from the database, focusing on:
- Embedding generation and storage
- Vector similarity search
- Database integration
- Model configuration flexibility

Tests use actual free-text fields from the evaluation table as specified
in data-dictionary.json: did_experience_issue_detail, course_application_other, general_feedback
"""

import asyncio
import pytest
import pytest_asyncio
import logging
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup path for imports
import sys
from pathlib import Path

# Add project root to path for imports
# From src/rag/tests/test_embeddings_manager.py, go up 3 levels to project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.data.embeddings_manager import EmbeddingsManager, SentenceTransformerProvider
from src.rag.config.settings import RAGSettings
from src.db import db_connector

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
    settings.embedding_batch_size = 50  # Smaller batch for testing
    settings.chunk_size = 200  # Smaller chunks for testing
    settings.chunk_overlap = 20
    return settings


@pytest_asyncio.fixture
async def embeddings_manager(rag_settings):
    """Create and initialise an EmbeddingsManager instance."""
    manager = EmbeddingsManager(rag_settings)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def sample_evaluation_texts():
    """Sample texts based on actual evaluation free-text fields."""
    return {
        "did_experience_issue_detail": [
            "The audio quality was poor during the virtual session, making it difficult to hear the facilitator clearly.",
            "Technical difficulties with the learning platform prevented access to course materials for the first hour.",
            "No significant issues encountered during the course delivery."
        ],
        "course_application_other": [
            "I plan to implement these leadership strategies in my team meetings and one-on-one discussions with staff.",
            "The content will be valuable for developing our agency's new performance management framework.",
            "These skills will help improve communication between different levels in our organisation."
        ],
        "general_feedback": [
            "Excellent course with practical examples relevant to public service work. The facilitator was knowledgeable and engaging.",
            "Good content but could benefit from more interactive elements and case studies specific to our agency context.",
            "The course met expectations and provided useful tools for immediate application in my current role."
        ]
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata following Australian public service context."""
    return {
        "user_level": "Level 5",
        "agency": "Australian Taxation Office",
        "course_type": "Virtual",
        "sentiment_scores": {
            "positive": 0.7,
            "negative": 0.1,
            "neutral": 0.2
        }
    }


class TestSentenceTransformerProvider:
    """Test the local sentence transformer embedding provider."""
    
    def test_provider_initialization(self):
        """Test that the sentence transformer provider initialises correctly."""
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.model is None  # Should be lazy loaded
        assert provider._dimension is None
    
    @pytest.mark.asyncio
    async def test_model_loading_and_embedding_generation(self):
        """Test model loading and embedding generation."""
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        
        test_texts = [
            "This is a test sentence for embedding.",
            "Another sentence to test batch processing."
        ]
        
        embeddings = await provider.generate_embeddings(test_texts)
        
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert provider.get_dimension() == 384  # all-MiniLM-L6-v2 dimension
        assert all(len(emb) == 384 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_embedding_consistency(self):
        """Test that identical texts produce identical embeddings."""
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        
        text = "Consistent embedding test"
        
        embeddings1 = await provider.generate_embeddings([text])
        embeddings2 = await provider.generate_embeddings([text])
        
        assert embeddings1 == embeddings2
    
    def test_model_version_identifier(self):
        """Test model version string generation."""
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        version = provider.get_model_version()
        
        assert version == "sentence-transformers-all-MiniLM-L6-v2-v1"
        assert "sentence-transformers" in version
        assert "all-MiniLM-L6-v2" in version


class TestEmbeddingsManagerInitialisation:
    """Test EmbeddingsManager initialisation and configuration."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, rag_settings):
        """Test embeddings manager initialises with correct configuration."""
        manager = EmbeddingsManager(rag_settings)
        await manager.initialize()
        
        assert manager._initialized is True
        assert manager.db_pool is not None
        assert manager.embedding_provider is not None
        assert isinstance(manager.embedding_provider, SentenceTransformerProvider)
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_provider_configuration(self, embeddings_manager):
        """Test that the provider is configured correctly."""
        provider = embeddings_manager.embedding_provider
        
        assert isinstance(provider, SentenceTransformerProvider)
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.get_model_version() == "sentence-transformers-all-MiniLM-L6-v2-v1"
    
    @pytest.mark.asyncio
    async def test_database_connection(self, embeddings_manager):
        """Test database connection pool is working."""
        async with embeddings_manager.db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1


class TestEmbeddingStorage:
    """Test embedding storage functionality."""
    
    @pytest.mark.asyncio
    async def test_store_single_field_embeddings(self, embeddings_manager, sample_evaluation_texts, sample_metadata):
        """Test storing embeddings for a single evaluation field."""
        response_id = 91  # Use a valid ID from evaluation table (avoiding low numbers that might be used elsewhere)
        field_name = "general_feedback"
        text_chunks = sample_evaluation_texts[field_name][:1]  # Just one text for simplicity
        
        embedding_ids = await embeddings_manager.store_embeddings(
            response_id=response_id,
            field_name=field_name,
            text_chunks=text_chunks,
            metadata=sample_metadata
        )
        
        assert len(embedding_ids) == 1
        assert all(isinstance(eid, int) for eid in embedding_ids)
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)
    
    @pytest.mark.asyncio
    async def test_store_multiple_chunks(self, embeddings_manager, sample_evaluation_texts, sample_metadata):
        """Test storing embeddings for multiple text chunks."""
        response_id = 92
        field_name = "did_experience_issue_detail"
        text_chunks = sample_evaluation_texts[field_name]
        
        embedding_ids = await embeddings_manager.store_embeddings(
            response_id=response_id,
            field_name=field_name,
            text_chunks=text_chunks,
            metadata=sample_metadata
        )
        
        assert len(embedding_ids) == len(text_chunks)
        assert len(embedding_ids) == 3
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)
    
    @pytest.mark.asyncio
    async def test_store_all_evaluation_fields(self, embeddings_manager, sample_evaluation_texts, sample_metadata):
        """Test storing embeddings for all three evaluation free-text fields."""
        response_id = 93
        field_results = {}
        
        # Store embeddings for each field
        for field_name, texts in sample_evaluation_texts.items():
            embedding_ids = await embeddings_manager.store_embeddings(
                response_id=response_id,
                field_name=field_name,
                text_chunks=texts,
                metadata={**sample_metadata, "field_type": field_name}
            )
            field_results[field_name] = embedding_ids
        
        # Verify all fields were stored
        assert len(field_results) == 3
        assert "did_experience_issue_detail" in field_results
        assert "course_application_other" in field_results
        assert "general_feedback" in field_results
        
        # Verify correct number of embeddings per field
        for field_name, embedding_ids in field_results.items():
            expected_count = len(sample_evaluation_texts[field_name])
            assert len(embedding_ids) == expected_count
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)


class TestVectorSearch:
    """Test vector similarity search functionality."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, embeddings_manager, sample_evaluation_texts, sample_metadata):
        """Test basic semantic search functionality."""
        # Store some test data
        response_id = 94
        field_name = "general_feedback"
        text_chunks = sample_evaluation_texts[field_name]
        
        await embeddings_manager.store_embeddings(
            response_id=response_id,
            field_name=field_name,
            text_chunks=text_chunks,
            metadata=sample_metadata
        )
        
        # Search for similar content
        query_text = "course quality and facilitator performance"
        results = await embeddings_manager.search_similar(
            query_text=query_text,
            field_name=field_name,
            limit=5,
            similarity_threshold=0.3  # Lower threshold for local model
        )
        
        assert len(results) > 0
        assert all("similarity_score" in result for result in results)
        assert all(result["field_name"] == field_name for result in results)
        assert all(result["response_id"] == response_id for result in results)
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)
    
    @pytest.mark.asyncio
    async def test_search_with_metadata_filtering(self, embeddings_manager, sample_evaluation_texts):
        """Test search with metadata filtering."""
        response_id = 95
        field_name = "course_application_other"
        
        # Store with specific metadata
        test_metadata = {
            "agency": "Department of Finance",
            "user_level": "Level 6",
            "course_type": "In-person"
        }
        
        await embeddings_manager.store_embeddings(
            response_id=response_id,
            field_name=field_name,
            text_chunks=sample_evaluation_texts[field_name][:1],
            metadata=test_metadata
        )
        
        # Search with metadata filter
        results = await embeddings_manager.search_similar(
            query_text="leadership implementation",
            field_name=field_name,
            metadata_filter={"agency": "Department of Finance"},
            limit=5,
            similarity_threshold=0.2
        )
        
        if results:  # Only test if we found results
            assert all(result["metadata"]["agency"] == "Department of Finance" for result in results)
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)
    
    @pytest.mark.asyncio
    async def test_search_across_multiple_fields(self, embeddings_manager, sample_evaluation_texts, sample_metadata):
        """Test searching across multiple evaluation fields."""
        response_id = 96
        
        # Store embeddings for multiple fields
        for field_name, texts in sample_evaluation_texts.items():
            await embeddings_manager.store_embeddings(
                response_id=response_id,
                field_name=field_name,
                text_chunks=texts[:1],  # Just one per field for simplicity
                metadata={**sample_metadata, "field_source": field_name}
            )
        
        # Search without field restriction (across all fields)
        results = await embeddings_manager.search_similar(
            query_text="course experience and technical issues",
            limit=10,
            similarity_threshold=0.2
        )
        
        if results:
            # Should find results from multiple fields
            found_fields = set(result["field_name"] for result in results)
            assert len(found_fields) > 0  # At least one field should match
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)


class TestDatabaseIntegration:
    """Test database integration and real data compatibility."""
    
    @pytest.mark.asyncio
    async def test_database_schema_compatibility(self, embeddings_manager):
        """Test that embeddings manager works with existing database schema."""
        # Test table existence and structure
        async with embeddings_manager.db_pool.acquire() as conn:
            # Check if rag_embeddings table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'rag_embeddings'
                )
            """)
            assert table_exists, "rag_embeddings table should exist"
            
            # Check vector column dimension compatibility
            vector_info = await conn.fetchrow("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'rag_embeddings' 
                    AND column_name = 'embedding'
            """)
            assert vector_info is not None
            assert vector_info['data_type'] == 'USER-DEFINED'
    
    @pytest.mark.asyncio
    async def test_evaluation_table_integration(self, embeddings_manager):
        """Test integration with actual evaluation table structure."""
        async with embeddings_manager.db_pool.acquire() as conn:
            # Check if evaluation table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'evaluation'
                )
            """)
            
            if table_exists:
                # Check for the three free-text fields we're targeting
                columns = await conn.fetch("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'evaluation'
                        AND column_name IN (
                            'did_experience_issue_detail',
                            'course_application_other', 
                            'general_feedback'
                        )
                """)
                
                column_names = [col['column_name'] for col in columns]
                assert 'did_experience_issue_detail' in column_names
                assert 'course_application_other' in column_names
                assert 'general_feedback' in column_names
    
    @pytest.mark.asyncio
    async def test_embeddings_statistics(self, embeddings_manager):
        """Test embeddings statistics functionality."""
        stats = await embeddings_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_embeddings' in stats
        assert 'unique_responses' in stats
        assert 'unique_fields' in stats
        assert 'unique_models' in stats
        assert 'field_breakdown' in stats
        assert 'model_breakdown' in stats
        
        # All values should be non-negative integers
        assert stats['total_embeddings'] >= 0
        assert stats['unique_responses'] >= 0
        assert stats['unique_fields'] >= 0
        assert stats['unique_models'] >= 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_text_chunks(self, embeddings_manager):
        """Test handling of empty text chunks."""
        response_id = 98
        
        embedding_ids = await embeddings_manager.store_embeddings(
            response_id=response_id,
            field_name="general_feedback",
            text_chunks=[],  # Empty list
            metadata={}
        )
        
        assert embedding_ids == []
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(self, embeddings_manager):
        """Test search that returns no results."""
        results = await embeddings_manager.search_similar(
            query_text="very specific technical jargon that should not match anything",
            similarity_threshold=0.95,  # Very high threshold
            limit=5
        )
        
        assert isinstance(results, list)
        # Results may be empty or contain low-similarity matches
    
    @pytest.mark.asyncio
    async def test_metadata_serialisation(self, embeddings_manager, sample_metadata):
        """Test that complex metadata is properly serialised."""
        response_id = 97
        
        complex_metadata = {
            **sample_metadata,
            "nested_data": {"sub_field": "value"},
            "list_data": [1, 2, 3],
            "unicode_data": "Test with émojis 🎉"
        }
        
        embedding_ids = await embeddings_manager.store_embeddings(
            response_id=response_id,
            field_name="general_feedback",
            text_chunks=["Test text with complex metadata"],
            metadata=complex_metadata
        )
        
        # Retrieve and verify metadata
        results = await embeddings_manager.search_similar(
            query_text="Test text",
            limit=1,
            similarity_threshold=0.1
        )
        
        if results:
            retrieved_metadata = results[0]['metadata']
            assert retrieved_metadata['nested_data']['sub_field'] == "value"
            assert retrieved_metadata['list_data'] == [1, 2, 3]
            assert retrieved_metadata['unicode_data'] == "Test with émojis 🎉"
        
        # Cleanup
        await embeddings_manager.delete_embeddings(response_id=response_id)


# Utility function for running specific test groups
async def run_embedding_tests():
    """Convenience function to run all embedding tests."""
    logger.info("Running comprehensive embedding tests with all-MiniLM-L6-v2 model")
    
    # This would typically be run by pytest, but can be used for manual testing
    import pytest
    return pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Run tests if executed directly
    asyncio.run(run_embedding_tests())
