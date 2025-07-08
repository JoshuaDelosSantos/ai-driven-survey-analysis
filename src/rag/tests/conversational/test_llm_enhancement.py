"""
Test suite for ConversationalLLMEnhancer

Following 06-07-2025 testing approach:
- Focused unit tests with clear assertions
- Mock dependencies for external components
- Edge case coverage for boundary conditions
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.rag.core.conversational.llm_enhancer import ConversationalLLMEnhancer, EnhancedResponse
from src.rag.core.conversational.handler import ConversationalPattern
from src.rag.core.privacy.pii_detector import PIIDetectionResult


class TestConversationalLLMEnhancer:
    """Test LLM enhancement routing logic."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager for testing."""
        manager = Mock()
        
        # Create a mock response object that matches LLMResponse and passes validation
        mock_response = Mock()
        mock_response.content = "Thanks for your question, I'm happy to help with that."
        mock_response.token_count = 25
        mock_response.model_name = "test-model"
        
        manager.generate = AsyncMock(return_value=mock_response)
        return manager

    @pytest.fixture
    def mock_pii_detector(self):
        """Mock PII detector for testing."""
        detector = Mock()
        detector.initialize = AsyncMock()
        
        # Create proper PIIDetectionResult objects instead of dicts
        no_pii_result = PIIDetectionResult(
            original_text='test query',
            anonymised_text='test query',
            entities_detected=[],
            confidence_scores={},
            processing_time=0.01,
            anonymisation_applied=False
        )
        
        pii_detected_result = PIIDetectionResult(
            original_text='My phone number is 123-456-7890',
            anonymised_text='My phone number is [PHONE_NUMBER]',
            entities_detected=[{'entity_type': 'PHONE_NUMBER', 'start': 19, 'end': 31}],
            confidence_scores={'PHONE_NUMBER': 0.95},
            processing_time=0.02,
            anonymisation_applied=True
        )
        
        detector.detect_and_anonymise = AsyncMock(return_value=no_pii_result)
        return detector

    @pytest_asyncio.fixture
    async def enhancer(self, mock_llm_manager, mock_pii_detector):
        """Create enhancer instance with mocked dependencies."""
        with patch('src.rag.core.conversational.llm_enhancer.LLMManager', return_value=mock_llm_manager), \
             patch('src.rag.core.conversational.llm_enhancer.AustralianPIIDetector', return_value=mock_pii_detector):
            enhancer = ConversationalLLMEnhancer()
            await enhancer.initialize()
            return enhancer

    @pytest.mark.asyncio
    async def test_enhance_response_bypasses_llm_when_confident(self, enhancer):
        """Test that high confidence template responses bypass LLM enhancement."""
        # Arrange
        high_confidence = 0.85
        template_response = "This is a template response"
        
        # Act
        result = await enhancer.enhance_response_if_needed(
            query="test query",
            template_response=template_response,
            confidence=high_confidence,
            pattern_type=ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result, EnhancedResponse)
        assert result.content == template_response
        assert result.confidence == high_confidence
        assert not result.enhancement_used
        assert result.pattern_type == ConversationalPattern.GREETING

    @pytest.mark.asyncio
    async def test_enhance_response_triggers_llm_when_unconfident(self, enhancer, mock_llm_manager):
        """Test that low confidence triggers LLM enhancement."""
        # Arrange
        low_confidence = 0.5
        template_response = "Basic template response"
        
        # Create a proper mock response that will pass validation
        mock_response = Mock()
        mock_response.content = "Thanks for your question, I'm happy to help with that."
        mock_response.token_count = 25
        mock_llm_manager.generate.return_value = mock_response
        
        # Mock the validation to always return True for this test
        with patch.object(enhancer, '_validate_enhanced_response', return_value=True):
            # Act
            result = await enhancer.enhance_response_if_needed(
                query="complex query",
                template_response=template_response,
                confidence=low_confidence,
                pattern_type=ConversationalPattern.SYSTEM_QUESTION
            )
        
        # Assert
        assert isinstance(result, EnhancedResponse)
        assert result.enhancement_used
        assert result.original_template == template_response
        # Note: confidence may be adjusted by the implementation
        assert result.content == "Thanks for your question, I'm happy to help with that."
        mock_llm_manager.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_response_respects_pii_detection_fallback(self, enhancer, mock_pii_detector):
        """Test that PII detection causes fallback to template."""
        # Arrange - create a PIIDetectionResult that indicates PII was detected
        pii_detected_result = PIIDetectionResult(
            original_text='My phone number is 123-456-7890',
            anonymised_text='My phone number is [PHONE_NUMBER]',
            entities_detected=[{'entity_type': 'PHONE_NUMBER', 'start': 19, 'end': 31}],
            confidence_scores={'PHONE_NUMBER': 0.95},
            processing_time=0.02,
            anonymisation_applied=True
        )
        mock_pii_detector.detect_and_anonymise.return_value = pii_detected_result
        
        low_confidence = 0.4
        template_response = "Template response"
        
        # Act
        result = await enhancer.enhance_response_if_needed(
            query="My phone number is 123-456-7890",
            template_response=template_response,
            confidence=low_confidence,
            pattern_type=ConversationalPattern.SYSTEM_QUESTION
        )
        
        # Assert
        assert isinstance(result, EnhancedResponse)
        assert result.content == template_response
        assert not result.enhancement_used
        assert not result.privacy_check_passed

    @pytest.mark.asyncio
    async def test_enhance_response_not_initialized(self):
        """Test enhancement when enhancer not initialized."""
        # Arrange
        enhancer = ConversationalLLMEnhancer()
        
        with patch.object(enhancer, 'initialize', new_callable=AsyncMock) as mock_init:
            # Act
            result = await enhancer.enhance_response_if_needed(
                query="test",
                template_response="template",
                confidence=0.5,
                pattern_type=ConversationalPattern.GREETING
            )
            
            # Assert
            mock_init.assert_called_once()
            assert isinstance(result, EnhancedResponse)

    @pytest.mark.asyncio
    async def test_enhance_response_records_processing_time(self, enhancer):
        """Test that LLM processing time is recorded."""
        # Arrange
        low_confidence = 0.3
        
        # Act
        result = await enhancer.enhance_response_if_needed(
            query="test",
            template_response="template",
            confidence=low_confidence,
            pattern_type=ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result.llm_processing_time, float)
        assert result.llm_processing_time >= 0.0

    @pytest.mark.asyncio
    async def test_enhance_response_threshold_boundary(self, enhancer):
        """Test enhancement behavior at confidence threshold boundary."""
        # Arrange - test at default threshold (0.7)
        threshold_confidence = 0.7
        template_response = "Template response"
        
        # Act
        result = await enhancer.enhance_response_if_needed(
            query="boundary test",
            template_response=template_response,
            confidence=threshold_confidence,
            pattern_type=ConversationalPattern.GREETING
        )
        
        # Assert
        assert isinstance(result, EnhancedResponse)
        assert result.content == template_response
        assert not result.enhancement_used
        # Should not enhance when exactly at threshold

    @pytest.mark.asyncio
    async def test_enhance_response_with_context(self, enhancer):
        """Test enhancement with optional context parameter."""
        # Arrange
        context = {"user_history": "returning_user", "session_id": "123"}
        
        # Act
        result = await enhancer.enhance_response_if_needed(
            query="test with context",
            template_response="template",
            confidence=0.8,
            pattern_type=ConversationalPattern.GREETING,
            context=context
        )
        
        # Assert
        assert isinstance(result, EnhancedResponse)
        # Should handle context parameter without errors
