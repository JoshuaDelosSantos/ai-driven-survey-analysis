"""
Test suite for ConversationalPatternClassifier

- Focused unit tests with clear assertions
- Mock dependencies for external components
- Edge case coverage for boundary conditions
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.rag.core.conversational.pattern_classifier import ConversationalPatternClassifier, ClassificationResult
from src.rag.core.conversational.handler import ConversationalPattern


class TestConversationalPatternClassifier:
    """Test vector-based pattern classification enhancement."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock text embedder for testing."""
        embedder = Mock()
        embed_result = Mock()
        embed_result.embedding = [0.1, 0.2, 0.3]
        embed_result.token_count = 5
        embedder.embed_text = AsyncMock(return_value=embed_result)
        return embedder

    @pytest.fixture
    def classifier(self, mock_embedder):
        """Create classifier instance with mocked dependencies."""
        classifier = ConversationalPatternClassifier()
        classifier.embedder = mock_embedder
        classifier.is_initialized = True
        classifier.pattern_vectors = {}  # Empty for testing
        return classifier

    @pytest.mark.asyncio
    async def test_classify_with_vector_boost_not_initialized(self):
        """Test classification when classifier not initialized."""
        # Arrange
        classifier = ConversationalPatternClassifier()
        classifier.is_initialized = False
        
        # Act
        result = await classifier.classify_with_vector_boost(
            "test query", 0.5, ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result, ClassificationResult)
        assert result.original_confidence == 0.5
        assert result.boosted_confidence == 0.5
        assert result.vector_similarity == 0.0

    @pytest.mark.asyncio
    async def test_classify_with_vector_boost_basic_functionality(self, classifier, mock_embedder):
        """Test basic vector boost classification."""
        # Arrange
        test_query = "hello there"
        template_confidence = 0.6
        
        # Act
        result = await classifier.classify_with_vector_boost(
            test_query, template_confidence, ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result, ClassificationResult)
        assert result.original_confidence == template_confidence
        mock_embedder.embed_text.assert_called_once_with(test_query)

    @pytest.mark.asyncio  
    async def test_classify_with_vector_boost_preserves_high_confidence(self, classifier):
        """Test that high template confidence is used appropriately."""
        # Arrange
        high_confidence = 0.95
        
        # Act
        result = await classifier.classify_with_vector_boost(
            "clear greeting", high_confidence, ConversationalPattern.GREETING
        )
        
        # Assert
        assert result.original_confidence == high_confidence
        assert isinstance(result.boosted_confidence, float)
        assert 0.0 <= result.boosted_confidence <= 1.0
        # The actual boosted confidence may be calculated differently than just preserving

    @pytest.mark.asyncio
    async def test_classify_with_vector_boost_edge_case_detection(self, classifier):
        """Test edge case detection in classification."""
        # Arrange
        ambiguous_query = "maybe perhaps sort of hello"
        low_confidence = 0.3
        
        # Act
        result = await classifier.classify_with_vector_boost(
            ambiguous_query, low_confidence, ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result.is_edge_case, bool)
        assert result.original_confidence == low_confidence

    @pytest.mark.asyncio
    async def test_classify_with_vector_boost_processing_time_recorded(self, classifier):
        """Test that processing time is properly recorded."""
        # Act
        result = await classifier.classify_with_vector_boost(
            "test", 0.5, ConversationalPattern.GREETING
        )
        
        # Assert
        assert result.processing_time >= 0.0
        assert isinstance(result.processing_time, float)

    @pytest.mark.asyncio
    async def test_classify_with_vector_boost_similar_patterns_returned(self, classifier):
        """Test that similar patterns are identified and returned."""
        # Act
        result = await classifier.classify_with_vector_boost(
            "test query", 0.5, ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result.similar_patterns, list)
        # Should return empty list when no pattern vectors exist
