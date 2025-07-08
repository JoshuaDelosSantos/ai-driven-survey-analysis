"""
Test suite for ConversationalLearningIntegrator

Following 06-07-2025 testing approach:
- Focused unit tests with clear assertions
- Mock dependencies for external components
- Edge case coverage for boundary conditions
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.rag.core.conversational.learning_integrator import (
    ConversationalLearningIntegrator, 
    LearningFeedback,
    RoutingStrategy,
    LearningUpdateType
)
from src.rag.core.conversational.handler import ConversationalPattern, PatternLearningData


class TestConversationalLearningIntegrator:
    """Test learning integration with existing pattern learning system."""

    @pytest.fixture
    def mock_handler(self):
        """Mock conversational handler for testing."""
        handler = Mock()
        
        # Mock pattern learning data with correct field names
        pattern_data = PatternLearningData(
            pattern=ConversationalPattern.SYSTEM_QUESTION.value,
            frequency=100,
            success_rate=0.80,
            last_used=datetime.now(),
            feedback_scores=[0.8, 0.85, 0.75, 0.9, 0.82],
            template_effectiveness={'default': 0.8},
            context_success={'general': 0.8},
            user_satisfaction=0.80,
            llm_confidence_threshold=0.75,
            llm_effectiveness=0.85
        )
        
        handler.pattern_learning = {
            ConversationalPattern.SYSTEM_QUESTION: pattern_data
        }
        handler.update_pattern_learning = Mock()
        return handler

    @pytest.fixture
    def integrator(self, mock_handler):
        """Create learning integrator instance with mocked handler."""
        return ConversationalLearningIntegrator(mock_handler)

    @pytest.fixture
    def sample_feedback(self):
        """Create sample learning feedback for testing."""
        return LearningFeedback(
            query="How can I analyze survey data?",
            pattern_type=ConversationalPattern.SYSTEM_QUESTION,
            routing_strategy=RoutingStrategy.LLM_ENHANCED,
            llm_used=True,
            was_helpful=True,
            response_time_ms=1200.0,
            confidence_score=0.65,
            user_satisfaction=0.85,
            timestamp=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_update_learning_with_feedback_basic_functionality(self, integrator, sample_feedback):
        """Test basic learning feedback integration."""
        # Act
        await integrator.update_learning_with_feedback(sample_feedback)
        
        # Assert
        assert len(integrator.learning_history) == 1
        assert integrator.learning_history[0] == sample_feedback
        
        # Should create performance cache entry using pattern + query length format
        query_words = len(sample_feedback.query.split())
        cache_key = f"{sample_feedback.pattern_type.value}_{query_words}"
        assert cache_key in integrator.performance_cache

    @pytest.mark.asyncio
    async def test_update_learning_stores_multiple_feedback_entries(self, integrator):
        """Test that multiple feedback entries are stored correctly."""
        # Arrange
        feedback_entries = [
            LearningFeedback(
                query="Hello there",
                pattern_type=ConversationalPattern.GREETING,
                routing_strategy=RoutingStrategy.TEMPLATE_ONLY,
                llm_used=False,
                was_helpful=True,
                response_time_ms=100.0,
                confidence_score=0.95,
                user_satisfaction=0.8,
                timestamp=datetime.now()
            ),
            LearningFeedback(
                query="I need help with analysis",
                pattern_type=ConversationalPattern.HELP_REQUEST,
                routing_strategy=RoutingStrategy.LLM_ENHANCED,
                llm_used=True,
                was_helpful=True,
                response_time_ms=2100.0,
                confidence_score=0.45,
                user_satisfaction=0.9,
                timestamp=datetime.now()
            )
        ]
        
        # Act
        for feedback in feedback_entries:
            await integrator.update_learning_with_feedback(feedback)
        
        # Assert
        assert len(integrator.learning_history) == 2
        assert len(integrator.performance_cache) == 2

    @pytest.mark.asyncio
    async def test_should_prefer_llm_insufficient_samples(self, integrator):
        """Test adaptive threshold with basic pattern."""
        # Arrange
        pattern = ConversationalPattern.SYSTEM_QUESTION
        query_length = 5
        
        # Act
        result = await integrator.get_adaptive_threshold(pattern, query_length)
        
        # Assert
        assert isinstance(result, float)
        assert 0.4 <= result <= 0.9  # Should be within reasonable bounds

    @pytest.mark.asyncio
    async def test_should_prefer_llm_with_sufficient_samples_and_better_performance(self, integrator, mock_handler):
        """Test adaptive threshold when LLM preference is learned."""
        # Arrange
        pattern = ConversationalPattern.SYSTEM_QUESTION
        query_length = 5
        pattern_key = f"{pattern.value}_{query_length}"
        
        # Mock pattern data to prefer LLM
        pattern_data = PatternLearningData(
            pattern=pattern.value,
            frequency=50,
            success_rate=0.75,
            last_used=datetime.now(),
            feedback_scores=[0.8, 0.85, 0.9],
            template_effectiveness={'default': 0.75},
            context_success={'general': 0.75},
            user_satisfaction=0.8,
            template_vs_llm_preference="llm",  # Learned to prefer LLM
            llm_effectiveness=0.9
        )
        mock_handler.pattern_learning[pattern_key] = pattern_data
        
        # Act
        result = await integrator.get_adaptive_threshold(pattern, query_length)
        
        # Assert
        assert result < 0.7  # Should lower threshold to favor LLM
        assert result >= 0.5  # But not too low

    @pytest.mark.asyncio
    async def test_should_prefer_llm_with_marginal_difference(self, integrator, mock_handler):
        """Test adaptive threshold when templates are preferred."""
        # Arrange
        pattern = ConversationalPattern.GREETING
        query_length = 3
        pattern_key = f"{pattern.value}_{query_length}"
        
        # Mock pattern data to prefer templates
        pattern_data = PatternLearningData(
            pattern=pattern.value,
            frequency=100,
            success_rate=0.9,
            last_used=datetime.now(),
            feedback_scores=[0.9, 0.85, 0.95],
            template_effectiveness={'default': 0.9},
            context_success={'general': 0.9},
            user_satisfaction=0.9,
            template_vs_llm_preference="template",  # Learned to prefer templates
            llm_effectiveness=0.8
        )
        mock_handler.pattern_learning[pattern_key] = pattern_data
        
        # Act
        result = await integrator.get_adaptive_threshold(pattern, query_length)
        
        # Assert
        assert result > 0.7  # Should raise threshold to favor templates
        assert result <= 0.9  # But not too high

    @pytest.mark.asyncio
    async def test_get_learning_insights_basic_functionality(self, integrator):
        """Test learning insights generation."""
        # Arrange - add some feedback for analysis
        pattern = ConversationalPattern.SYSTEM_QUESTION
        for i in range(3):
            feedback = LearningFeedback(
                query="How does this work?",
                pattern_type=pattern,
                routing_strategy=RoutingStrategy.LLM_ENHANCED,
                llm_used=True,
                was_helpful=True,
                response_time_ms=1500.0,
                confidence_score=0.6,
                user_satisfaction=0.85,
                timestamp=datetime.now()
            )
            await integrator.update_learning_with_feedback(feedback)
        
        # Act
        insights = await integrator.get_learning_insights()
        
        # Assert
        assert isinstance(insights, dict)
        assert 'pattern_performance' in insights
        assert 'routing_effectiveness' in insights

    @pytest.mark.asyncio
    async def test_get_performance_metrics_calculates_correctly(self, integrator):
        """Test learning summary generation."""
        # Arrange - add feedback history
        feedback_data = [
            ("How to analyze?", 0.8, True, 1000.0),
            ("What is correlation?", 0.9, True, 1200.0),
            ("Help with stats", 0.7, False, 800.0)
        ]
        
        for query, satisfaction, helpful, response_time in feedback_data:
            feedback = LearningFeedback(
                query=query,
                pattern_type=ConversationalPattern.SYSTEM_QUESTION,
                routing_strategy=RoutingStrategy.LLM_ENHANCED,
                llm_used=True,
                was_helpful=helpful,
                response_time_ms=response_time,
                confidence_score=0.65,
                user_satisfaction=satisfaction,
                timestamp=datetime.now()
            )
            await integrator.update_learning_with_feedback(feedback)
        
        # Act
        summary = integrator.get_learning_summary()
        
        # Assert
        assert isinstance(summary, dict)
        assert 'total_feedback_samples' in summary
        assert summary['total_feedback_samples'] == 3

    @pytest.mark.asyncio
    async def test_get_performance_metrics_no_data_returns_none(self, integrator):
        """Test adaptive threshold with no learning data."""
        # Act - request threshold for pattern with no data
        result = await integrator.get_adaptive_threshold(
            ConversationalPattern.OFF_TOPIC, 
            5
        )
        
        # Assert
        assert result == 0.7  # Should return default threshold

    @pytest.mark.asyncio
    async def test_learning_feedback_to_dict_serialization(self, sample_feedback):
        """Test that LearningFeedback serializes correctly to dict."""
        # Act
        feedback_dict = sample_feedback.to_dict()
        
        # Assert
        assert feedback_dict['pattern_type'] == sample_feedback.pattern_type.value
        assert feedback_dict['routing_strategy'] == sample_feedback.routing_strategy.value
        assert feedback_dict['llm_used'] == sample_feedback.llm_used
        assert feedback_dict['was_helpful'] == sample_feedback.was_helpful
        assert feedback_dict['user_satisfaction'] == sample_feedback.user_satisfaction
        assert 'timestamp' in feedback_dict
        assert 'query_length' in feedback_dict
