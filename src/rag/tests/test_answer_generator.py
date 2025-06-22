#!/usr/bin/env python3
"""
Test suite for Answer Generation System with comprehensive coverage.

This module tests the intelligent answer synthesis including:
- Multi-modal synthesis strategies (Statistical, Feedback, Hybrid, Error)
- PII protection in generated responses
- Confidence scoring and quality metrics
- Source attribution and transparency
- Integration with Australian PII detection

Tests use both mocked and real components for comprehensive validation.
"""

import asyncio
import pytest
import pytest_asyncio
import logging
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Setup path for imports
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.core.synthesis.answer_generator import AnswerGenerator, AnswerType, SynthesisResult
from src.rag.core.privacy.pii_detector import AustralianPIIDetector
from src.rag.utils.llm_utils import get_llm
from src.rag.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAnswerGenerator:
    """Test intelligent answer synthesis from multiple sources."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing without API calls."""
        mock = AsyncMock()
        mock.ainvoke = AsyncMock()
        return mock
    
    @pytest_asyncio.fixture
    async def pii_detector(self):
        """Create a real PII detector for privacy testing."""
        detector = AustralianPIIDetector()
        await detector.initialize()
        return detector
    
    @pytest_asyncio.fixture
    async def generator_with_mock_llm(self, mock_llm, pii_detector):
        """Create answer generator with mock LLM for unit testing."""
        return AnswerGenerator(llm=mock_llm, pii_detector=pii_detector)
    
    @pytest_asyncio.fixture
    async def generator_with_real_llm(self, pii_detector):
        """Create answer generator with real LLM for integration testing."""
        llm = get_llm()
        return AnswerGenerator(llm=llm, pii_detector=pii_detector)
    
    @pytest.fixture
    def sample_sql_result(self):
        """Sample SQL analysis result."""
        return {
            "success": True,
            "result": [
                {"agency": "Department of Finance", "completion_rate": 87.5, "total_users": 240},
                {"agency": "Department of Health", "completion_rate": 92.1, "total_users": 180},
                {"agency": "Department of Education", "completion_rate": 78.9, "total_users": 320}
            ],
            "query": "SELECT agency, AVG(completion_rate) as completion_rate, COUNT(*) as total_users FROM user_stats GROUP BY agency"
        }
    
    @pytest.fixture
    def sample_vector_result(self):
        """Sample vector search result."""
        return {
            "results": [
                {"text": "The virtual learning platform was intuitive and easy to navigate", "score": 0.92},
                {"text": "I appreciated the flexibility to complete modules at my own pace", "score": 0.89},
                {"text": "Some technical issues with video playback on older browsers", "score": 0.76},
                {"text": "Overall satisfied with the learning experience and content quality", "score": 0.88},
                {"text": "Would recommend improvements to the mobile interface", "score": 0.71}
            ],
            "query": "virtual learning platform feedback",
            "total_results": 5
        }
    
    @pytest.fixture
    def sample_error_result(self):
        """Sample error result for testing error synthesis."""
        return {
            "success": False,
            "error": "Database connection timeout",
            "query": "SELECT * FROM users"
        }
    
    # Unit Tests - Answer Type Determination
    def test_determine_answer_type_sql_only(self, generator_with_mock_llm, sample_sql_result):
        """Test answer type determination for SQL-only results."""
        generator = generator_with_mock_llm
        
        answer_type = generator._determine_answer_type(sample_sql_result, None)
        assert answer_type == AnswerType.STATISTICAL_ONLY
    
    def test_determine_answer_type_vector_only(self, generator_with_mock_llm, sample_vector_result):
        """Test answer type determination for vector-only results."""
        generator = generator_with_mock_llm
        
        answer_type = generator._determine_answer_type(None, sample_vector_result)
        assert answer_type == AnswerType.FEEDBACK_ONLY
    
    def test_determine_answer_type_hybrid(self, generator_with_mock_llm, sample_sql_result, sample_vector_result):
        """Test answer type determination for hybrid results."""
        generator = generator_with_mock_llm
        
        answer_type = generator._determine_answer_type(sample_sql_result, sample_vector_result)
        assert answer_type == AnswerType.HYBRID_COMBINED
    
    def test_determine_answer_type_error(self, generator_with_mock_llm):
        """Test answer type determination for error state."""
        generator = generator_with_mock_llm
        
        answer_type = generator._determine_answer_type(None, None)
        assert answer_type == AnswerType.ERROR_RESPONSE
    
    # Unit Tests - Synthesis Strategies
    @pytest.mark.asyncio
    async def test_statistical_answer_synthesis(self, generator_with_mock_llm, sample_sql_result):
        """Test synthesis of SQL-only statistical answers."""
        generator = generator_with_mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Based on the database analysis, here are the course completion statistics by agency:
        
        • Department of Finance: 87.5% completion rate (240 users)
        • Department of Health: 92.1% completion rate (180 users)  
        • Department of Education: 78.9% completion rate (320 users)
        
        The Department of Health shows the highest completion rate at 92.1%, while Education has the largest user base with 320 participants.
        """
        generator.llm.ainvoke.return_value = mock_response
        
        result = await generator.synthesize_answer(
            query="What are the completion rates by agency?",
            sql_result=sample_sql_result,
            session_id="test_001"
        )
        
        assert isinstance(result, SynthesisResult)
        assert result.answer_type == AnswerType.STATISTICAL_ONLY
        assert result.confidence > 0.5
        assert "Department of Finance" in result.answer
        assert "87.5%" in result.answer
        assert "Database Analysis" in result.sources
        assert not result.pii_detected
    
    @pytest.mark.asyncio
    async def test_feedback_answer_synthesis(self, generator_with_mock_llm, sample_vector_result):
        """Test synthesis of vector search feedback answers."""
        generator = generator_with_mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Based on user feedback analysis, here are the main themes regarding the virtual learning platform:
        
        Positive Feedback (78% of responses):
        • Users appreciate the intuitive navigation and user-friendly interface
        • Flexibility to complete modules at their own pace is highly valued
        • Overall satisfaction with learning experience and content quality
        
        Areas for Improvement (22% of responses):
        • Technical issues with video playback, particularly on older browsers
        • Mobile interface could be enhanced for better accessibility
        
        The feedback indicates strong user satisfaction with the platform's core functionality and flexibility.
        """
        generator.llm.ainvoke.return_value = mock_response
        
        result = await generator.synthesize_answer(
            query="What feedback did users provide about the virtual learning platform?",
            vector_result=sample_vector_result,
            session_id="test_002"
        )
        
        assert isinstance(result, SynthesisResult)
        assert result.answer_type == AnswerType.FEEDBACK_ONLY
        assert result.confidence > 0.5
        assert "intuitive" in result.answer.lower()
        assert "flexibility" in result.answer.lower()
        assert "User Feedback" in result.sources
        assert not result.pii_detected
    
    @pytest.mark.asyncio
    async def test_hybrid_answer_synthesis(self, generator_with_mock_llm, sample_sql_result, sample_vector_result):
        """Test synthesis of combined SQL and vector results."""
        generator = generator_with_mock_llm
        
        # Mock LLM response for hybrid synthesis
        mock_response = MagicMock()
        mock_response.content = """
        Comprehensive Analysis - Platform Performance and User Satisfaction:
        
        Statistical Overview:
        • Department of Health leads with 92.1% completion rate (180 users)
        • Department of Finance follows at 87.5% completion rate (240 users)
        • Department of Education has 78.9% completion rate but largest user base (320 users)
        
        Supporting User Feedback:
        The high completion rates align with positive user sentiment. Key satisfaction drivers include:
        • Intuitive platform navigation supporting user engagement
        • Flexible self-paced learning accommodating different schedules
        • Quality content contributing to course completion
        
        Areas for Improvement:
        • Technical issues with video playback may impact completion rates
        • Mobile interface improvements could boost accessibility
        
        Conclusion: Strong performance metrics are validated by positive user feedback, with technical optimizations identified for further improvement.
        """
        generator.llm.ainvoke.return_value = mock_response
        
        result = await generator.synthesize_answer(
            query="Analyze completion rates with supporting user feedback",
            sql_result=sample_sql_result,
            vector_result=sample_vector_result,
            session_id="test_003"
        )
        
        assert isinstance(result, SynthesisResult)
        assert result.answer_type == AnswerType.HYBRID_COMBINED
        assert result.confidence > 0.7  # Higher confidence for hybrid
        assert "92.1%" in result.answer  # Statistical data
        assert "intuitive" in result.answer.lower()  # Feedback data
        assert len(result.sources) == 2  # Both SQL and vector sources
        assert "Database Analysis" in result.sources
        assert "User Feedback" in result.sources
    
    @pytest.mark.asyncio
    async def test_error_response_generation(self, generator_with_mock_llm):
        """Test generation of error responses."""
        generator = generator_with_mock_llm
        
        result = await generator.synthesize_answer(
            query="Show me user statistics",
            sql_result=None,
            vector_result=None,
            session_id="test_004"
        )
        
        assert isinstance(result, SynthesisResult)
        assert result.answer_type == AnswerType.ERROR_RESPONSE
        assert result.confidence == 0.0
        assert "wasn't able to find" in result.answer or "couldn't find" in result.answer
        assert len(result.sources) == 0
    
    # Privacy Compliance Tests
    @pytest.mark.asyncio
    async def test_pii_detection_in_generated_answers(self, generator_with_mock_llm, sample_vector_result):
        """Test PII detection in generated answers."""
        generator = generator_with_mock_llm
        
        # Mock LLM response with PII
        mock_response = MagicMock()
        mock_response.content = """
        User feedback analysis shows positive responses:
        • Contact John Smith at john.smith@agency.gov.au for training support
        • Sarah Jones (phone: 0412345678) reported excellent experience
        • Follow up with ABN 12345678901 for advanced training modules
        
        Overall satisfaction is high across all departments.
        """
        generator.llm.ainvoke.return_value = mock_response
        
        result = await generator.synthesize_answer(
            query="What feedback did users provide?",
            vector_result=sample_vector_result,
            session_id="test_005"
        )
        
        assert result.pii_detected == True
        assert "john.smith@agency.gov.au" not in result.answer
        assert "0412345678" not in result.answer
        assert "12345678901" not in result.answer
        # Should contain anonymized placeholders
        assert "[REDACTED]" in result.answer or "[EMAIL_ADDRESS]" in result.answer
    
    @pytest.mark.asyncio
    async def test_pii_anonymization_in_responses(self, generator_with_mock_llm, sample_sql_result):
        """Test complete PII anonymization in responses."""
        generator = generator_with_mock_llm
        
        # Mock LLM response with multiple PII types
        mock_response = MagicMock()
        mock_response.content = """
        Department completion analysis:
        • Finance: john.doe@finance.gov.au achieved 87.5% completion
        • Health: Contact via phone 0487654321 - 92.1% completion  
        • Education: TFN 123456789 user group - 78.9% completion
        • Medicare number 1234567890 participants showed high engagement
        """
        generator.llm.ainvoke.return_value = mock_response
        
        result = await generator.synthesize_answer(
            query="Show completion analysis",
            sql_result=sample_sql_result,
            session_id="test_006"
        )
        
        # Verify all PII types are removed
        assert "john.doe@finance.gov.au" not in result.answer
        assert "0487654321" not in result.answer
        assert "123456789" not in result.answer  # TFN
        assert "1234567890" not in result.answer  # Medicare
        
        # Verify completion percentages are preserved
        assert "87.5%" in result.answer
        assert "92.1%" in result.answer
        assert "78.9%" in result.answer
    
    def test_source_attribution_privacy_safe(self, generator_with_mock_llm):
        """Test that source attribution doesn't leak sensitive information."""
        generator = generator_with_mock_llm
        
        # Test source list building
        sources = generator._build_sources_list(
            {"success": True, "query": "SELECT * FROM users WHERE email='john@test.com'"},
            {"results": [{"text": "Contact admin@agency.gov.au"}]},
            AnswerType.HYBRID_COMBINED
        )
        
        assert "Database Analysis" in sources
        assert "User Feedback" in sources
        # Ensure no PII in source descriptions
        assert "john@test.com" not in str(sources)
        assert "admin@agency.gov.au" not in str(sources)
    
    # Quality Assessment Tests
    def test_confidence_calculation_high_quality_data(self, generator_with_mock_llm, sample_sql_result, sample_vector_result):
        """Test confidence calculation with high-quality data from both sources."""
        generator = generator_with_mock_llm
        
        confidence = generator._calculate_confidence(
            sample_sql_result, 
            sample_vector_result, 
            AnswerType.HYBRID_COMBINED
        )
        
        assert confidence >= 0.9  # Should be high for both sources
        assert confidence <= 1.0
    
    def test_confidence_calculation_partial_data(self, generator_with_mock_llm, sample_sql_result):
        """Test confidence calculation with partial data (SQL only)."""
        generator = generator_with_mock_llm
        
        confidence = generator._calculate_confidence(
            sample_sql_result, 
            None, 
            AnswerType.STATISTICAL_ONLY
        )
        
        assert confidence >= 0.7  # Good confidence for SQL data
        assert confidence < 0.9   # But not as high as hybrid
    
    def test_confidence_calculation_error_state(self, generator_with_mock_llm):
        """Test confidence calculation for error state."""
        generator = generator_with_mock_llm
        
        confidence = generator._calculate_confidence(
            None, 
            None, 
            AnswerType.ERROR_RESPONSE
        )
        
        assert confidence == 0.0
    
    # Integration Tests
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesis_with_real_llm(self, generator_with_real_llm, sample_sql_result):
        """Test synthesis with actual LLM (requires API access)."""
        generator = generator_with_real_llm
        
        result = await generator.synthesize_answer(
            query="Analyze the completion rates by agency",
            sql_result=sample_sql_result,
            session_id="integration_001"
        )
        
        assert isinstance(result, SynthesisResult)
        assert result.answer_type == AnswerType.STATISTICAL_ONLY
        assert result.confidence > 0.5
        assert len(result.answer) > 50  # Reasonable answer length
        assert result.processing_time is not None
        assert result.processing_time > 0
        assert "Database Analysis" in result.sources
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesis_with_pii_detector(self, generator_with_real_llm, sample_vector_result):
        """Test synthesis with real PII detector integration."""
        generator = generator_with_real_llm
        
        # Inject PII into vector results
        pii_vector_result = {
            "results": [
                {"text": "Contact john.smith@agency.gov.au for more information", "score": 0.9},
                {"text": "Great platform experience overall", "score": 0.85}
            ],
            "query": "user contact feedback",
            "total_results": 2
        }
        
        result = await generator.synthesize_answer(
            query="What contact information did users provide?",
            vector_result=pii_vector_result,
            session_id="integration_002"
        )
        
        # Verify PII handling
        assert "john.smith@agency.gov.au" not in result.answer
        if result.pii_detected:
            assert "[REDACTED]" in result.answer or "[EMAIL_ADDRESS]" in result.answer
    
    # Performance Tests
    @pytest.mark.asyncio
    async def test_synthesis_performance_targets(self, generator_with_mock_llm, sample_sql_result, sample_vector_result):
        """Test that synthesis meets performance targets."""
        generator = generator_with_mock_llm
        
        # Mock quick LLM response
        mock_response = MagicMock()
        mock_response.content = "Quick analysis response"
        generator.llm.ainvoke.return_value = mock_response
        
        start_time = time.time()
        result = await generator.synthesize_answer(
            query="Quick analysis",
            sql_result=sample_sql_result,
            vector_result=sample_vector_result,
            session_id="perf_001"
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Synthesis too slow: {processing_time:.3f}s"
        assert result.processing_time is not None
        assert result.processing_time < processing_time  # Internal timing should be accurate
    
    @pytest.mark.asyncio
    async def test_large_result_handling(self, generator_with_mock_llm):
        """Test handling of large result sets."""
        generator = generator_with_mock_llm
        
        # Create large vector result
        large_vector_result = {
            "results": [
                {"text": f"User feedback item {i} with detailed content", "score": 0.8 - (i * 0.01)}
                for i in range(100)
            ],
            "query": "comprehensive feedback",
            "total_results": 100
        }
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Comprehensive analysis of large dataset"
        generator.llm.ainvoke.return_value = mock_response
        
        result = await generator.synthesize_answer(
            query="Analyze comprehensive feedback",
            vector_result=large_vector_result,
            session_id="large_001"
        )
        
        assert result.answer_type == AnswerType.FEEDBACK_ONLY
        assert len(result.answer) < generator.max_answer_length
        # Should handle large input gracefully without errors
    
    # Error Handling Tests
    @pytest.mark.asyncio
    async def test_llm_failure_handling(self, generator_with_mock_llm, sample_sql_result):
        """Test graceful handling of LLM failures."""
        generator = generator_with_mock_llm
        
        # Mock LLM failure
        generator.llm.ainvoke.side_effect = Exception("LLM API error")
        
        result = await generator.synthesize_answer(
            query="Analyze data",
            sql_result=sample_sql_result,
            session_id="error_001"
        )
        
        assert isinstance(result, SynthesisResult)
        assert result.answer_type == AnswerType.ERROR_RESPONSE
        assert "encountered an issue" in result.answer
        assert result.confidence == 0.0
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_timeout_handling_in_synthesis(self, generator_with_mock_llm, sample_sql_result):
        """Test timeout handling during synthesis."""
        generator = generator_with_mock_llm
        
        # Mock slow LLM response
        async def slow_llm_response(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than reasonable timeout
            return MagicMock(content="Delayed response")
        
        generator.llm.ainvoke = slow_llm_response
        
        start_time = time.time()
        result = await generator.synthesize_answer(
            query="Analyze data",
            sql_result=sample_sql_result,
            session_id="timeout_001"
        )
        end_time = time.time()
        
        # Should timeout and return error response
        assert end_time - start_time < 8  # Should not wait full 10 seconds
        assert result.answer_type == AnswerType.ERROR_RESPONSE
    
    # Edge Cases
    @pytest.mark.asyncio
    async def test_empty_results_handling(self, generator_with_mock_llm):
        """Test handling of empty results."""
        generator = generator_with_mock_llm
        
        empty_sql = {"success": True, "result": []}
        empty_vector = {"results": [], "total_results": 0}
        
        result = await generator.synthesize_answer(
            query="Find information",
            sql_result=empty_sql,
            vector_result=empty_vector,
            session_id="empty_001"
        )
        
        assert result.answer_type == AnswerType.ERROR_RESPONSE
        assert "wasn't able to find" in result.answer or "no relevant" in result.answer
    
    @pytest.mark.asyncio
    async def test_malformed_results_handling(self, generator_with_mock_llm):
        """Test handling of malformed input results."""
        generator = generator_with_mock_llm
        
        malformed_sql = {"success": True}  # Missing 'result' key
        malformed_vector = {"results": "invalid"}  # Wrong type
        
        result = await generator.synthesize_answer(
            query="Process malformed data",
            sql_result=malformed_sql,
            vector_result=malformed_vector,
            session_id="malformed_001"
        )
        
        # Should handle gracefully and return error
        assert result.answer_type == AnswerType.ERROR_RESPONSE
        assert result.confidence == 0.0
