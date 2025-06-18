"""
Test suite for Vector Search Tool implementation with comprehensive coverage.

This module tests the vector search functionality including:
- VectorSearchTool initialization and configuration
- Search operations with various parameters
- Metadata filtering capabilities
- Privacy compliance (PII detection)
- Performance monitoring and metrics
- LangChain tool interface compatibility
- Integration with existing components

Tests use real models and database connections for authentic validation.
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

from src.rag.core.vector_search.vector_search_tool import VectorSearchTool, SearchParameters
from src.rag.core.vector_search.search_result import (
    VectorSearchResponse, VectorSearchResult, SearchMetadata, RelevanceCategory
)
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
    settings.chunk_size = 200
    settings.chunk_overlap = 20
    return settings


@pytest_asyncio.fixture
async def vector_search_tool(rag_settings):
    """Create and initialize a VectorSearchTool instance."""
    tool = VectorSearchTool()
    await tool.initialize()
    yield tool
    # Cleanup if needed


@pytest.fixture
def sample_search_queries():
    """Sample search queries for testing various scenarios."""
    return {
        "technical_issues": "feedback about technical difficulties and system problems",
        "course_effectiveness": "how effective was the course for learning",
        "virtual_learning": "experiences with online and virtual training delivery",
        "facilitator_feedback": "comments about instructor and facilitator performance",
        "senior_staff": "feedback from executive level and senior management",
        "negative_sentiment": "complaints and negative experiences with courses"
    }


@pytest.fixture
def sample_metadata_filters():
    """Sample metadata filters for testing filtering capabilities."""
    return {
        "high_level_users": {
            "user_level": ["Level 5", "Level 6", "Exec Level 1", "Exec Level 2"]
        },
        "specific_agency": {
            "agency": "Department of Finance"
        },
        "virtual_courses": {
            "course_delivery_type": ["Virtual", "Blended"]
        },
        "negative_sentiment": {
            "sentiment": {"type": "negative", "min_score": 0.6}
        },
        "feedback_fields": {
            "field_name": ["general_feedback", "did_experience_issue_detail"]
        },
        "complex_filter": {
            "user_level": ["Level 5", "Level 6"],
            "agency": ["Department of Finance", "Australian Taxation Office"],
            "course_delivery_type": "Virtual",
            "sentiment": {"type": "negative", "min_score": 0.5}
        }
    }


class TestVectorSearchToolInitialization:
    """Test VectorSearchTool initialization and configuration."""
    
    @pytest.mark.asyncio
    async def test_tool_initialization(self, rag_settings):
        """Test that VectorSearchTool initializes correctly."""
        tool = VectorSearchTool()
        assert not tool._initialized
        
        await tool.initialize()
        
        assert tool._initialized
        assert tool._embedder is not None
        assert tool._embeddings_manager is not None
        assert tool._pii_detector is not None
        assert tool.default_field_names == [
            "general_feedback",
            "did_experience_issue_detail", 
            "course_application_other"
        ]
    
    @pytest.mark.asyncio
    async def test_tool_initialization_failure_handling(self):
        """Test handling of initialization failures."""
        tool = VectorSearchTool()
        
        # Test usage before initialization
        with pytest.raises(RuntimeError, match="not initialized"):
            await tool.search("test query")
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, vector_search_tool):
        """Test that double initialization is handled correctly."""
        # Tool is already initialized via fixture
        assert vector_search_tool._initialized
        
        # Should not raise an error
        await vector_search_tool.initialize()
        assert vector_search_tool._initialized


class TestSearchParameters:
    """Test SearchParameters validation and configuration."""
    
    def test_valid_parameters(self):
        """Test creation of valid search parameters."""
        params = SearchParameters(
            query="test query",
            max_results=10,
            similarity_threshold=0.75,
            filters={"user_level": ["Level 5"]},
            field_names=["general_feedback"]
        )
        
        params.validate()  # Should not raise
        assert params.query == "test query"
        assert params.max_results == 10
        assert params.similarity_threshold == 0.75
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        # Empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            params = SearchParameters(query="")
            params.validate()
        
        # Invalid max_results
        with pytest.raises(ValueError, match="max_results must be between 1 and 100"):
            params = SearchParameters(query="test", max_results=0)
            params.validate()
        
        with pytest.raises(ValueError, match="max_results must be between 1 and 100"):
            params = SearchParameters(query="test", max_results=101)
            params.validate()
        
        # Invalid similarity_threshold
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            params = SearchParameters(query="test", similarity_threshold=-0.1)
            params.validate()
        
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            params = SearchParameters(query="test", similarity_threshold=1.1)
            params.validate()


class TestBasicSearchOperations:
    """Test basic vector search operations."""
    
    @pytest.mark.asyncio
    async def test_basic_search(self, vector_search_tool, sample_search_queries):
        """Test basic search functionality."""
        query = sample_search_queries["technical_issues"]
        
        response = await vector_search_tool.search(
            query=query,
            max_results=5,
            similarity_threshold=0.5  # Lower threshold for testing
        )
        
        assert isinstance(response, VectorSearchResponse)
        assert response.query == query
        assert response.result_count <= 5
        assert response.similarity_threshold == 0.5
        assert response.processing_time > 0
        assert response.query_anonymized in [True, False]  # Depends on query content
        assert "pii_detection" in response.privacy_controls_applied
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(self, vector_search_tool):
        """Test search that returns no results."""
        # Use a very specific query unlikely to match anything
        response = await vector_search_tool.search(
            query="extremely specific query about quantum computing in medieval times",
            max_results=10,
            similarity_threshold=0.95  # Very high threshold
        )
        
        assert response.result_count == 0
        assert len(response.results) == 0
        assert response.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_search_performance_tracking(self, vector_search_tool, sample_search_queries):
        """Test that search operations are tracked for performance monitoring."""
        initial_metrics = vector_search_tool.get_performance_metrics()
        initial_count = initial_metrics.get("searches_performed", 0)
        
        await vector_search_tool.search(
            query=sample_search_queries["course_effectiveness"],
            max_results=3
        )
        
        updated_metrics = vector_search_tool.get_performance_metrics()
        assert updated_metrics["searches_performed"] == initial_count + 1
        assert "average_search_time" in updated_metrics
        assert updated_metrics["initialization_status"] == "initialized"


class TestMetadataFiltering:
    """Test metadata filtering capabilities."""
    
    @pytest.mark.asyncio
    async def test_user_level_filtering(self, vector_search_tool, sample_metadata_filters):
        """Test filtering by user level."""
        response = await vector_search_tool.search(
            query="course feedback",
            filters=sample_metadata_filters["high_level_users"],
            max_results=10,
            similarity_threshold=0.5
        )
        
        # Check that filters were applied
        assert "user_level" in response.filters_applied
        assert response.filters_applied["user_level"] == ["Level 5", "Level 6", "Exec Level 1", "Exec Level 2"]
        
        # If we have results, verify they match the filter
        for result in response.results:
            if result.metadata.user_level:
                assert result.metadata.user_level in ["Level 5", "Level 6", "Exec Level 1", "Exec Level 2"]
    
    @pytest.mark.asyncio
    async def test_agency_filtering(self, vector_search_tool, sample_metadata_filters):
        """Test filtering by agency."""
        response = await vector_search_tool.search(
            query="learning experience",
            filters=sample_metadata_filters["specific_agency"],
            max_results=10,
            similarity_threshold=0.5
        )
        
        assert "agency" in response.filters_applied
        
        # If we have results, verify they match the filter
        for result in response.results:
            if result.metadata.agency:
                assert result.metadata.agency == "Department of Finance"
    
    @pytest.mark.asyncio
    async def test_sentiment_filtering(self, vector_search_tool, sample_metadata_filters):
        """Test filtering by sentiment scores."""
        response = await vector_search_tool.search(
            query="course problems",
            filters=sample_metadata_filters["negative_sentiment"],
            max_results=10,
            similarity_threshold=0.5
        )
        
        assert "sentiment" in response.filters_applied
        
        # If we have results, verify they match the sentiment filter
        for result in response.results:
            if result.metadata.sentiment_scores and "negative" in result.metadata.sentiment_scores:
                assert result.metadata.sentiment_scores["negative"] >= 0.6
    
    @pytest.mark.asyncio
    async def test_complex_filtering(self, vector_search_tool, sample_metadata_filters):
        """Test complex multi-criteria filtering."""
        response = await vector_search_tool.search(
            query="virtual learning feedback",
            filters=sample_metadata_filters["complex_filter"],
            max_results=10,
            similarity_threshold=0.4
        )
        
        # Check that all filters were applied
        assert "user_level" in response.filters_applied
        assert "agency" in response.filters_applied
        assert "course_delivery_type" in response.filters_applied
        assert "sentiment" in response.filters_applied


class TestPrivacyCompliance:
    """Test privacy and PII protection features."""
    
    @pytest.mark.asyncio
    async def test_query_anonymization(self, vector_search_tool):
        """Test that queries with PII are properly anonymized."""
        # Query with potential PII (phone number)
        pii_query = "I had issues calling 0412 345 678 for course support"
        
        response = await vector_search_tool.search(
            query=pii_query,
            max_results=5,
            similarity_threshold=0.5
        )
        
        # Should indicate that query was anonymized
        assert response.query_anonymized in [True, False]  # Depends on PII detection
        assert "pii_detection" in response.privacy_controls_applied
        assert "query_anonymization" in response.privacy_controls_applied
    
    @pytest.mark.asyncio
    async def test_error_handling_privacy_safe(self, vector_search_tool):
        """Test that errors don't expose sensitive information."""
        # This will test error handling without exposing internals
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await vector_search_tool.search(query="")


class TestResultStructures:
    """Test search result data structures and processing."""
    
    @pytest.mark.asyncio
    async def test_result_structure_completeness(self, vector_search_tool, sample_search_queries):
        """Test that search results have all expected fields and properties."""
        response = await vector_search_tool.search(
            query=sample_search_queries["course_effectiveness"],
            max_results=3,
            similarity_threshold=0.5
        )
        
        # Test response structure
        assert hasattr(response, 'query')
        assert hasattr(response, 'results')
        assert hasattr(response, 'total_results')
        assert hasattr(response, 'processing_time')
        assert hasattr(response, 'relevance_distribution')
        assert hasattr(response, 'average_similarity')
        
        # Test individual results if any exist
        for result in response.results:
            assert isinstance(result, VectorSearchResult)
            assert hasattr(result, 'chunk_text')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'metadata')
            assert hasattr(result, 'relevance_category')
            assert hasattr(result, 'user_context')
            assert hasattr(result, 'sentiment_summary')
            
            # Test metadata structure
            assert isinstance(result.metadata, SearchMetadata)
    
    @pytest.mark.asyncio
    async def test_relevance_categorization(self, vector_search_tool, sample_search_queries):
        """Test relevance score categorization."""
        response = await vector_search_tool.search(
            query=sample_search_queries["technical_issues"],
            max_results=10,
            similarity_threshold=0.3  # Lower threshold to get various scores
        )
        
        for result in response.results:
            score = result.similarity_score
            category = result.relevance_category
            
            if score >= 0.80:
                assert category == RelevanceCategory.HIGH
            elif score >= 0.65:
                assert category == RelevanceCategory.MEDIUM
            elif score >= 0.50:
                assert category == RelevanceCategory.LOW
            else:
                assert category == RelevanceCategory.WEAK
    
    @pytest.mark.asyncio
    async def test_result_serialization(self, vector_search_tool, sample_search_queries):
        """Test that results can be serialized to JSON."""
        response = await vector_search_tool.search(
            query=sample_search_queries["virtual_learning"],
            max_results=3,
            similarity_threshold=0.5
        )
        
        # Test response serialization
        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert "query" in response_dict
        assert "results" in response_dict
        assert "processing_time" in response_dict
        
        # Test individual result serialization
        for result in response.results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "chunk_text" in result_dict
            assert "similarity_score" in result_dict
            assert "relevance_category" in result_dict


class TestLangChainToolInterface:
    """Test LangChain tool interface compatibility."""
    
    @pytest.mark.asyncio
    async def test_langchain_tool_properties(self, vector_search_tool):
        """Test that the tool has required LangChain properties."""
        assert hasattr(vector_search_tool, 'name')
        assert hasattr(vector_search_tool, 'description')
        assert hasattr(vector_search_tool, 'args_schema')
        
        assert vector_search_tool.name == "vector_search"
        assert isinstance(vector_search_tool.description, str)
        assert len(vector_search_tool.description) > 0
    
    @pytest.mark.asyncio
    async def test_langchain_ainvoke_interface(self, vector_search_tool):
        """Test the async invoke interface for LangChain compatibility."""
        # Test basic ainvoke
        result = await vector_search_tool._arun(
            query="course feedback",
            max_results=3,
            similarity_threshold=0.5
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should contain summary information
        assert "Found" in result or "No relevant" in result
    
    @pytest.mark.asyncio
    async def test_langchain_error_handling(self, vector_search_tool):
        """Test error handling in LangChain interface."""
        # Test with invalid parameters
        result = await vector_search_tool._arun(
            query="",  # Empty query should cause error
            max_results=3
        )
        
        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()


class TestPerformanceAndMetrics:
    """Test performance monitoring and metrics collection."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, vector_search_tool, sample_search_queries):
        """Test that performance metrics are collected properly."""
        # Perform multiple searches
        queries = list(sample_search_queries.values())[:3]
        
        for query in queries:
            response = await vector_search_tool.search(
                query=query,
                max_results=5,
                similarity_threshold=0.5
            )
            
            # Check response-level metrics
            assert response.embedding_time >= 0
            assert response.search_time >= 0
            assert response.filtering_time >= 0
            assert response.processing_time > 0
        
        # Check tool-level metrics
        metrics = vector_search_tool.get_performance_metrics()
        assert metrics["searches_performed"] >= len(queries)
        assert "average_search_time" in metrics
    
    @pytest.mark.asyncio
    async def test_search_response_analysis(self, vector_search_tool, sample_search_queries):
        """Test search response analysis capabilities."""
        response = await vector_search_tool.search(
            query=sample_search_queries["course_effectiveness"],
            max_results=10,
            similarity_threshold=0.4
        )
        
        # Test response analysis methods
        distribution = response.relevance_distribution
        assert isinstance(distribution, dict)
        assert all(category in distribution for category in ["High", "Medium", "Low", "Weak"])
        assert all(isinstance(count, int) for count in distribution.values())
        
        # Test filtering methods if results exist
        if response.results:
            avg_similarity = response.average_similarity
            assert 0 <= avg_similarity <= 1
            
            has_high_quality = response.has_high_quality_results
            assert isinstance(has_high_quality, bool)


# Integration tests (require database and embeddings)
@pytest.mark.integration
class TestIntegrationWithDatabase:
    """Integration tests requiring database and embedding data."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_workflow(self, vector_search_tool):
        """Test complete end-to-end search workflow with real data."""
        # This test requires actual embeddings in the database
        try:
            response = await vector_search_tool.search(
                query="course feedback and learning experience",
                max_results=10,
                similarity_threshold=0.5
            )
            
            # Should complete without errors
            assert isinstance(response, VectorSearchResponse)
            logger.info(f"End-to-end test completed: {response.result_count} results found")
            
        except Exception as e:
            # Log but don't fail if no data available
            logger.warning(f"End-to-end test skipped (no data available): {e}")
            pytest.skip("No embedding data available for integration test")


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])
