"""
Test suite for Search Result data structures.

This module tests the search result data structures including:
- SearchMetadata container functionality
- VectorSearchResult properties and methods
- VectorSearchResponse analysis capabilities
- Relevance categorization logic
- Serialization and deserialization
- Data structure integrity

Tests focus on data structure behavior and utility methods.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any

# Setup path for imports
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.core.vector_search.search_result import (
    SearchMetadata,
    VectorSearchResult,
    VectorSearchResponse,
    RelevanceCategory
)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return SearchMetadata(
        user_level="Level 5",
        agency="Department of Finance",
        knowledge_level_prior="Intermediate",
        course_delivery_type="Virtual",
        course_end_date="2025-06-15",
        field_name="general_feedback",
        response_id=123,
        chunk_index=0,
        sentiment_scores={
            "positive": 0.7,
            "negative": 0.1,
            "neutral": 0.2
        },
        facilitator_skills=["Displayed strong knowledge", "Communicated clearly"],
        had_guest_speakers="Yes",
        model_version="sentence-transformers-all-MiniLM-L6-v2-v1",
        created_at=datetime(2025, 6, 18, 10, 30, 0)
    )


@pytest.fixture
def sample_search_result(sample_metadata):
    """Create sample search result for testing."""
    return VectorSearchResult(
        chunk_text="The course was excellent and provided valuable insights for my role.",
        similarity_score=0.85,
        metadata=sample_metadata,
        embedding_id=456
    )


@pytest.fixture
def sample_search_results(sample_metadata):
    """Create multiple search results with different scores."""
    results = []
    
    # High relevance result
    results.append(VectorSearchResult(
        chunk_text="Excellent course with practical applications.",
        similarity_score=0.92,
        metadata=sample_metadata,
        embedding_id=1
    ))
    
    # Medium relevance result
    metadata_medium = SearchMetadata(
        user_level="Level 3",
        agency="Australian Taxation Office",
        field_name="general_feedback",
        response_id=124,
        sentiment_scores={"positive": 0.5, "negative": 0.3, "neutral": 0.2}
    )
    results.append(VectorSearchResult(
        chunk_text="Good course but could be improved.",
        similarity_score=0.72,
        metadata=metadata_medium,
        embedding_id=2
    ))
    
    # Low relevance result
    metadata_low = SearchMetadata(
        user_level="Level 1",
        agency="Department of Health",
        field_name="did_experience_issue_detail",
        response_id=125,
        sentiment_scores={"positive": 0.2, "negative": 0.8, "neutral": 0.0}
    )
    results.append(VectorSearchResult(
        chunk_text="Some technical issues occurred.",
        similarity_score=0.58,
        metadata=metadata_low,
        embedding_id=3
    ))
    
    # Weak relevance result
    metadata_weak = SearchMetadata(
        user_level="Level 2",
        agency="Department of Education",
        field_name="course_application_other",
        response_id=126
    )
    results.append(VectorSearchResult(
        chunk_text="Not particularly relevant content.",
        similarity_score=0.45,
        metadata=metadata_weak,
        embedding_id=4
    ))
    
    return results


class TestSearchMetadata:
    """Test SearchMetadata functionality."""
    
    def test_metadata_creation(self, sample_metadata):
        """Test basic metadata creation and attribute access."""
        assert sample_metadata.user_level == "Level 5"
        assert sample_metadata.agency == "Department of Finance"
        assert sample_metadata.field_name == "general_feedback"
        assert sample_metadata.response_id == 123
        assert sample_metadata.chunk_index == 0
        assert isinstance(sample_metadata.sentiment_scores, dict)
        assert sample_metadata.sentiment_scores["positive"] == 0.7
    
    def test_metadata_to_dict(self, sample_metadata):
        """Test metadata dictionary conversion."""
        metadata_dict = sample_metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert "user_context" in metadata_dict
        assert "course_context" in metadata_dict
        assert "evaluation_context" in metadata_dict
        assert "analytics" in metadata_dict
        assert "technical" in metadata_dict
        
        # Check nested structure
        assert metadata_dict["user_context"]["user_level"] == "Level 5"
        assert metadata_dict["user_context"]["agency"] == "Department of Finance"
        assert metadata_dict["analytics"]["sentiment_scores"]["positive"] == 0.7
        assert metadata_dict["technical"]["model_version"] == "sentence-transformers-all-MiniLM-L6-v2-v1"
    
    def test_metadata_with_minimal_data(self):
        """Test metadata creation with minimal required data."""
        minimal_metadata = SearchMetadata(
            field_name="general_feedback",
            model_version="test-model-v1"
        )
        
        assert minimal_metadata.field_name == "general_feedback"
        assert minimal_metadata.model_version == "test-model-v1"
        assert minimal_metadata.user_level is None
        assert minimal_metadata.agency is None
        
        # Should still be convertible to dict
        metadata_dict = minimal_metadata.to_dict()
        assert isinstance(metadata_dict, dict)


class TestVectorSearchResult:
    """Test VectorSearchResult functionality."""
    
    def test_result_creation(self, sample_search_result):
        """Test basic result creation and attribute access."""
        assert sample_search_result.chunk_text == "The course was excellent and provided valuable insights for my role."
        assert sample_search_result.similarity_score == 0.85
        assert isinstance(sample_search_result.metadata, SearchMetadata)
        assert sample_search_result.embedding_id == 456
    
    def test_relevance_categorization(self):
        """Test relevance category assignment based on similarity scores."""
        # Test high relevance
        high_result = VectorSearchResult(
            chunk_text="Test text",
            similarity_score=0.85,
            metadata=SearchMetadata(field_name="test"),
            embedding_id=1
        )
        assert high_result.relevance_category == RelevanceCategory.HIGH
        
        # Test medium relevance
        medium_result = VectorSearchResult(
            chunk_text="Test text",
            similarity_score=0.70,
            metadata=SearchMetadata(field_name="test"),
            embedding_id=2
        )
        assert medium_result.relevance_category == RelevanceCategory.MEDIUM
        
        # Test low relevance
        low_result = VectorSearchResult(
            chunk_text="Test text",
            similarity_score=0.55,
            metadata=SearchMetadata(field_name="test"),
            embedding_id=3
        )
        assert low_result.relevance_category == RelevanceCategory.LOW
        
        # Test weak relevance
        weak_result = VectorSearchResult(
            chunk_text="Test text",
            similarity_score=0.45,
            metadata=SearchMetadata(field_name="test"),
            embedding_id=4
        )
        assert weak_result.relevance_category == RelevanceCategory.WEAK
    
    def test_user_context_property(self, sample_search_result):
        """Test user context summary generation."""
        user_context = sample_search_result.user_context
        
        assert isinstance(user_context, str)
        assert "Level 5" in user_context
        assert "Department of Finance" in user_context
        assert "staff from" in user_context
    
    def test_user_context_with_missing_data(self):
        """Test user context with missing metadata."""
        metadata = SearchMetadata(field_name="test")
        result = VectorSearchResult(
            chunk_text="Test",
            similarity_score=0.5,
            metadata=metadata,
            embedding_id=1
        )
        
        user_context = result.user_context
        assert "Unknown Level" in user_context
        assert "Unknown Agency" in user_context
    
    def test_sentiment_summary_property(self, sample_search_result):
        """Test sentiment summary generation."""
        sentiment_summary = sample_search_result.sentiment_summary
        
        assert isinstance(sentiment_summary, str)
        assert "Positive" in sentiment_summary
        assert "0.70" in sentiment_summary
    
    def test_sentiment_summary_with_no_data(self):
        """Test sentiment summary when no sentiment data available."""
        metadata = SearchMetadata(field_name="test")
        result = VectorSearchResult(
            chunk_text="Test",
            similarity_score=0.5,
            metadata=metadata,
            embedding_id=1
        )
        
        sentiment_summary = result.sentiment_summary
        assert sentiment_summary == "No sentiment data"
    
    def test_result_to_dict(self, sample_search_result):
        """Test result dictionary conversion."""
        result_dict = sample_search_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "chunk_text" in result_dict
        assert "similarity_score" in result_dict
        assert "relevance_category" in result_dict
        assert "embedding_id" in result_dict
        assert "user_context" in result_dict
        assert "sentiment_summary" in result_dict
        assert "metadata" in result_dict
        
        # Check values
        assert result_dict["chunk_text"] == sample_search_result.chunk_text
        assert result_dict["similarity_score"] == 0.85
        assert result_dict["relevance_category"] == "High"
        assert result_dict["embedding_id"] == 456


class TestVectorSearchResponse:
    """Test VectorSearchResponse functionality."""
    
    def test_response_creation(self, sample_search_results):
        """Test basic response creation."""
        response = VectorSearchResponse(
            query="test query",
            results=sample_search_results,
            total_results=10,
            processing_time=1.23,
            similarity_threshold=0.75,
            max_results=10
        )
        
        assert response.query == "test query"
        assert len(response.results) == 4
        assert response.total_results == 10
        assert response.processing_time == 1.23
        assert response.similarity_threshold == 0.75
        assert response.max_results == 10
    
    def test_result_count_property(self, sample_search_results):
        """Test result count property."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        assert response.result_count == 4
        assert response.result_count == len(sample_search_results)
    
    def test_relevance_distribution(self, sample_search_results):
        """Test relevance distribution calculation."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        distribution = response.relevance_distribution
        
        assert isinstance(distribution, dict)
        assert distribution["High"] == 1  # One result with score 0.92
        assert distribution["Medium"] == 1  # One result with score 0.72
        assert distribution["Low"] == 1  # One result with score 0.58
        assert distribution["Weak"] == 1  # One result with score 0.45
    
    def test_average_similarity(self, sample_search_results):
        """Test average similarity calculation."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        # Expected average: (0.92 + 0.72 + 0.58 + 0.45) / 4 = 0.6675
        avg_similarity = response.average_similarity
        assert abs(avg_similarity - 0.6675) < 0.001
    
    def test_average_similarity_empty_results(self):
        """Test average similarity with no results."""
        response = VectorSearchResponse(
            query="test",
            results=[],
            total_results=0,
            processing_time=1.0
        )
        
        assert response.average_similarity == 0.0
    
    def test_has_high_quality_results(self, sample_search_results):
        """Test high quality results detection."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        assert response.has_high_quality_results is True  # Has one result with score 0.92
        
        # Test with no high quality results
        low_quality_results = [r for r in sample_search_results if r.similarity_score < 0.80]
        response_low = VectorSearchResponse(
            query="test",
            results=low_quality_results,
            total_results=5,
            processing_time=1.0
        )
        
        assert response_low.has_high_quality_results is False
    
    def test_get_results_by_agency(self, sample_search_results):
        """Test filtering results by agency."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        finance_results = response.get_results_by_agency("Department of Finance")
        assert len(finance_results) == 1
        assert finance_results[0].metadata.agency == "Department of Finance"
        
        ato_results = response.get_results_by_agency("Australian Taxation Office")
        assert len(ato_results) == 1
        assert ato_results[0].metadata.agency == "Australian Taxation Office"
        
        nonexistent_results = response.get_results_by_agency("Nonexistent Agency")
        assert len(nonexistent_results) == 0
    
    def test_get_results_by_sentiment(self, sample_search_results):
        """Test filtering results by sentiment."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        # Test positive sentiment filtering
        positive_results = response.get_results_by_sentiment("positive", min_score=0.6)
        assert len(positive_results) == 1  # Only first result has positive > 0.6
        
        # Test negative sentiment filtering
        negative_results = response.get_results_by_sentiment("negative", min_score=0.7)
        assert len(negative_results) == 1  # Only third result has negative > 0.7
        
        # Test with high threshold
        high_positive_results = response.get_results_by_sentiment("positive", min_score=0.9)
        assert len(high_positive_results) == 0
    
    def test_response_to_dict(self, sample_search_results):
        """Test response dictionary conversion."""
        response = VectorSearchResponse(
            query="test query",
            results=sample_search_results,
            total_results=10,
            processing_time=1.23,
            similarity_threshold=0.75,
            max_results=10,
            filters_applied={"user_level": ["Level 5"]},
            embedding_time=0.1,
            search_time=0.8,
            filtering_time=0.33,
            query_anonymized=True,
            privacy_controls_applied=["pii_detection", "query_anonymization"]
        )
        
        response_dict = response.to_dict()
        
        assert isinstance(response_dict, dict)
        assert response_dict["query"] == "test query"
        assert response_dict["result_count"] == 4
        assert response_dict["total_results"] == 10
        assert response_dict["processing_time"] == 1.23
        assert response_dict["has_high_quality_results"] is True
        
        # Check nested structures
        assert "search_parameters" in response_dict
        assert "performance_metrics" in response_dict
        assert "privacy_info" in response_dict
        assert "results" in response_dict
        
        # Check specific values
        assert response_dict["search_parameters"]["similarity_threshold"] == 0.75
        assert response_dict["performance_metrics"]["embedding_time"] == 0.1
        assert response_dict["privacy_info"]["query_anonymized"] is True
        assert len(response_dict["results"]) == 4


class TestRelevanceCategory:
    """Test RelevanceCategory enum."""
    
    def test_category_values(self):
        """Test that category values are correct."""
        assert RelevanceCategory.HIGH.value == "High"
        assert RelevanceCategory.MEDIUM.value == "Medium"
        assert RelevanceCategory.LOW.value == "Low"
        assert RelevanceCategory.WEAK.value == "Weak"
    
    def test_category_comparison(self):
        """Test category enum comparison."""
        assert RelevanceCategory.HIGH != RelevanceCategory.MEDIUM
        assert RelevanceCategory.LOW == RelevanceCategory.LOW


class TestSerialization:
    """Test JSON serialization capabilities."""
    
    def test_metadata_json_serialization(self, sample_metadata):
        """Test that metadata can be JSON serialized."""
        metadata_dict = sample_metadata.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(metadata_dict, default=str)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
        assert deserialized["user_context"]["user_level"] == "Level 5"
    
    def test_result_json_serialization(self, sample_search_result):
        """Test that search results can be JSON serialized."""
        result_dict = sample_search_result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict, default=str)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
        assert deserialized["similarity_score"] == 0.85
    
    def test_response_json_serialization(self, sample_search_results):
        """Test that search responses can be JSON serialized."""
        response = VectorSearchResponse(
            query="test",
            results=sample_search_results,
            total_results=10,
            processing_time=1.0
        )
        
        response_dict = response.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(response_dict, default=str)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
        assert deserialized["result_count"] == 4


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])
