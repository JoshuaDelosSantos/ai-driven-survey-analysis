#!/usr/bin/env python3
"""
Test suite for Query Classification System with comprehensive coverage.

This module tests the multi-stage query classification including:
- Rule-based classification for obvious queries
- LLM-based classification with confidence scoring
- Fallback mechanisms (LLM → Rule-based → Clarification)
- PII anonymisation before LLM processing
- Error handling and timeout management
- Integration with existing components

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

from src.rag.core.routing.query_classifier import QueryClassifier
from src.rag.core.privacy.pii_detector import AustralianPIIDetector
from src.rag.utils.llm_utils import get_llm
from src.rag.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestQueryClassifier:
    """Test query classification with multi-stage logic."""
    
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
        await detector.initialise()
        return detector
    
    @pytest_asyncio.fixture
    async def classifier_with_mock_llm(self, mock_llm):
        """Create classifier with mock LLM for unit testing."""
        classifier = QueryClassifier(llm=mock_llm)
        await classifier.initialize()
        return classifier
    
    @pytest_asyncio.fixture
    async def classifier_with_real_llm(self):
        """Create classifier with real LLM for integration testing."""
        classifier = QueryClassifier()
        await classifier.initialize()
        return classifier
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for different classification types."""
        return {
            'sql_indicators': [
                "How many users completed courses in each agency?",
                "Show me the count of attendance by user level",
                "What is the average completion rate?",
                "Give me a breakdown by department",
                "List the percentage of users who finished training"
            ],
            'vector_indicators': [
                "What did users say about virtual learning?",
                "Show me feedback about the new platform?",
                "What are user experiences with mobile access?",
                "Show me comments about the course content",
                "What are people's opinions on the system?"
            ],
            'hybrid_indicators': [
                "Analyze satisfaction by agency levels",
                "Compare feedback across different departments", 
                "Show comprehensive analysis of user performance",
                "Provide detailed analysis of training outcomes",
                "Show both numbers and feedback about completion"
            ],
            'ambiguous': [
                "Tell me about the platform",
                "How is the system?",
                "Show me data",
                "What about users?",
                "Give me information"
            ]
        }
    
    # Unit Tests - Rule-based Classification
    def test_rule_based_classification_sql_indicators(self, classifier_with_mock_llm, sample_queries):
        """Test rule-based classification for clear SQL indicators."""
        classifier = classifier_with_mock_llm
        
        for query in sample_queries['sql_indicators']:
            result = classifier._rule_based_classification(query)
            assert result is not None, f"Should classify SQL query: {query}"
            assert result.classification == 'SQL', f"Should classify as SQL: {query}"
            assert result.confidence in ['HIGH', 'MEDIUM'], f"Should have good confidence: {query}"
            assert result.reasoning is not None, f"Should include reasoning: {query}"
    
    def test_rule_based_classification_vector_indicators(self, classifier_with_mock_llm, sample_queries):
        """Test rule-based classification for clear vector search indicators."""
        classifier = classifier_with_mock_llm
        
        for query in sample_queries['vector_indicators']:
            result = classifier._rule_based_classification(query)
            assert result is not None, f"Should classify vector query: {query}"
            assert result.classification == 'VECTOR', f"Should classify as VECTOR: {query}"
            assert result.confidence in ['HIGH', 'MEDIUM'], f"Should have good confidence: {query}"
            assert result.reasoning is not None, f"Should include reasoning: {query}"
    
    def test_rule_based_classification_hybrid_indicators(self, classifier_with_mock_llm, sample_queries):
        """Test rule-based classification for hybrid queries."""
        classifier = classifier_with_mock_llm
        
        for query in sample_queries['hybrid_indicators']:
            result = classifier._rule_based_classification(query)
            assert result is not None, f"Should classify hybrid query: {query}"
            assert result.classification == 'HYBRID', f"Should classify as HYBRID: {query}"
            assert result.confidence in ['HIGH', 'MEDIUM'], f"Should have good confidence: {query}"
            assert result.reasoning is not None, f"Should include reasoning: {query}"
    
    def test_rule_based_classification_unknown_query(self, classifier_with_mock_llm, sample_queries):
        """Test rule-based classification returns None for ambiguous queries."""
        classifier = classifier_with_mock_llm
        
        for query in sample_queries['ambiguous']:
            result = classifier._rule_based_classification(query)
            assert result is None, f"Should not classify ambiguous query: {query}"
    
    # Unit Tests - LLM Classification
    @pytest.mark.asyncio
    async def test_llm_classification_high_confidence(self, classifier_with_mock_llm):
        """Test LLM classification with high confidence response."""
        classifier = classifier_with_mock_llm
        
        # Mock high confidence LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Classification: SQL
        Confidence: 0.92
        Reasoning: The query explicitly asks for counting users by agency, which requires aggregation from the database.
        """
        classifier._llm.ainvoke.return_value = mock_response
        
        result = await classifier._llm_classification("How many users are in each agency?")
        
        assert result is not None
        assert result.classification == 'SQL'
        assert result.confidence == 'HIGH'
        assert 'counting users by agency' in result.reasoning.lower()
        
        # Verify LLM was called
        classifier._llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_llm_classification_low_confidence(self, classifier_with_mock_llm):
        """Test LLM classification with low confidence response."""
        classifier = classifier_with_mock_llm
        
        # Mock low confidence LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Classification: CLARIFICATION_NEEDED
        Confidence: 0.3
        Reasoning: The query is too vague and could refer to multiple types of analysis.
        """
        classifier._llm.ainvoke.return_value = mock_response
        
        result = await classifier._llm_classification("Tell me about the data")
        
        assert result is not None
        assert result.classification == 'CLARIFICATION_NEEDED'
        assert result.confidence == 'LOW'
        assert 'too vague' in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_llm_classification_failure_fallback(self, classifier_with_mock_llm):
        """Test LLM classification failure triggers fallback."""
        classifier = classifier_with_mock_llm
        
        # Mock LLM failure
        classifier._llm.ainvoke.side_effect = Exception("LLM API error")
        
        result = await classifier._llm_classification("How many users completed training?")
        
        assert result is None  # Should return None to trigger fallback
    
    @pytest.mark.asyncio
    async def test_pii_anonymization_before_llm(self, classifier_with_mock_llm):
        """Test that PII is anonymised before sending to LLM."""
        classifier = classifier_with_mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Classification: SQL
        Confidence: 0.85
        Reasoning: Query about user completion rates.
        """
        classifier._llm.ainvoke.return_value = mock_response
        
        # Query with potential PII
        query_with_pii = "Show completion rates for john.smith@agency.gov.au and phone 0412345678"
        
        result = await classifier._llm_classification(query_with_pii)
        
        # Verify LLM was called
        classifier._llm.ainvoke.assert_called_once()
        
        # Extract the actual query sent to LLM
        call_args = classifier._llm.ainvoke.call_args[0][0]
        
        # Verify PII was removed/anonymised
        assert "john.smith@agency.gov.au" not in str(call_args)
        assert "0412345678" not in str(call_args)
    
    # Integration Tests
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_classification_workflow(self, classifier_with_mock_llm, sample_queries):
        """Test complete classification workflow from query to result."""
        classifier = classifier_with_mock_llm
        
        # Mock LLM response for fallback scenario
        mock_response = MagicMock()
        mock_response.content = """
        Classification: VECTOR
        Confidence: 0.75
        Reasoning: Query asks about user feedback and experiences.
        """
        classifier._llm.ainvoke.return_value = mock_response
        
        # Test SQL query (should use rule-based)
        sql_result = await classifier.classify_query("How many users completed courses?")
        assert sql_result.classification == 'SQL'
        assert sql_result.confidence in ['HIGH', 'MEDIUM']
        assert sql_result.method_used == 'rule_based'
        
        # Test ambiguous query (should use LLM)
        ambiguous_result = await classifier.classify_query("Tell me about user experiences")
        assert ambiguous_result.classification == 'VECTOR'
        assert ambiguous_result.confidence in ['MEDIUM', 'HIGH']  # Flexible on exact confidence
        assert ambiguous_result.method_used in ['llm_based', 'rule_based']  # Could be either
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_classification_with_real_llm(self, classifier_with_real_llm, sample_queries):
        """Test classification with actual LLM (requires API access)."""
        classifier = classifier_with_real_llm
        
        # Test a clear SQL query
        result = await classifier.classify_query("How many users are there by agency?")
        
        assert result is not None
        assert result.classification in ['SQL', 'VECTOR', 'HYBRID', 'CLARIFICATION_NEEDED']
        assert result.confidence in ['HIGH', 'MEDIUM', 'LOW']
        assert result.reasoning is not None
        assert result.method_used is not None
        assert result.processing_time is not None
        
        # For a clear SQL query, we expect SQL classification
        if result.method_used == 'rule_based':
            assert result.classification == 'SQL'
            assert result.confidence in ['HIGH', 'MEDIUM']
    
    # Error Handling Tests
    @pytest.mark.asyncio
    async def test_network_failure_handling(self, classifier_with_mock_llm):
        """Test handling of network failures during LLM classification."""
        classifier = classifier_with_mock_llm
        
        # Mock network timeout
        import aiohttp
        classifier._llm.ainvoke.side_effect = aiohttp.ClientTimeoutError()
        
        # Should fall back to clarification for ambiguous query
        result = await classifier.classify_query("Tell me about the system")
        
        assert result is not None
        assert result.classification == 'CLARIFICATION_NEEDED'
        assert result.confidence == 'LOW'
        assert 'network' in result.reasoning.lower() or 'error' in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, classifier_with_mock_llm):
        """Test classification timeout handling."""
        classifier = classifier_with_mock_llm
        
        # Mock slow LLM response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than typical timeout
            return MagicMock(content="Classification: SQL\nConfidence: 0.8")
        
        classifier._llm.ainvoke = slow_response
        
        # Should timeout and fall back
        start_time = time.time()
        result = await classifier.classify_query("Show me user data")
        end_time = time.time()
        
        assert result is not None
        assert end_time - start_time < 8  # Should timeout before 10 seconds
        assert result.classification == 'CLARIFICATION_NEEDED'  # Fallback
    
    @pytest.mark.asyncio
    async def test_invalid_llm_response_handling(self, classifier_with_mock_llm):
        """Test handling of invalid/malformed LLM responses."""
        classifier = classifier_with_mock_llm
        
        # Mock invalid LLM response
        mock_response = MagicMock()
        mock_response.content = "Invalid response format without proper structure"
        classifier._llm.ainvoke.return_value = mock_response
        
        result = await classifier.classify_query("How are users performing?")
        
        assert result is not None
        assert result.classification == 'CLARIFICATION_NEEDED'  # Fallback
        assert result.confidence == 'LOW'
        assert 'parsing' in result.reasoning.lower() or 'error' in result.reasoning.lower()
    
    # Performance Tests
    @pytest.mark.asyncio
    async def test_rule_based_classification_performance(self, classifier_with_mock_llm, sample_queries):
        """Test that rule-based classification meets performance targets."""
        classifier = classifier_with_mock_llm
        
        # Test multiple queries for average performance
        all_queries = (sample_queries['sql_indicators'] + 
                      sample_queries['vector_indicators'] + 
                      sample_queries['hybrid_indicators'])
        
        start_time = time.time()
        for query in all_queries:
            classifier._rule_based_classification(query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(all_queries)
        assert avg_time < 0.1, f"Rule-based classification too slow: {avg_time:.3f}s average"
    
    @pytest.mark.asyncio
    async def test_pii_detection_performance(self, classifier_with_mock_llm):
        """Test PII detection performance doesn't impact classification speed."""
        classifier = classifier_with_mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Classification: SQL\nConfidence: 0.8\nReasoning: Test"
        classifier._llm.ainvoke.return_value = mock_response
        
        query_with_pii = ("Show completion rates for users john.smith@agency.gov.au, "
                         "jane.doe@health.gov.au and phone numbers 0412345678, 0487654321")
        
        start_time = time.time()
        await classifier.classify_query(query_with_pii)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"PII detection impacting performance: {processing_time:.3f}s"
    
    # Privacy Compliance Tests
    @pytest.mark.asyncio
    async def test_no_pii_leakage_in_logs(self, classifier_with_mock_llm, caplog):
        """Test that PII doesn't appear in log messages."""
        classifier = classifier_with_mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Classification: SQL\nConfidence: 0.8\nReasoning: User query analysis"
        classifier._llm.ainvoke.return_value = mock_response
        
        query_with_pii = "Show data for john.smith@agency.gov.au with ABN 12345678901"
        
        with caplog.at_level(logging.INFO):
            await classifier.classify_query(query_with_pii)
        
        # Check that PII doesn't appear in any log messages
        log_content = " ".join([record.message for record in caplog.records])
        assert "john.smith@agency.gov.au" not in log_content
        assert "12345678901" not in log_content
    
    @pytest.mark.asyncio
    async def test_classification_audit_logging(self, classifier_with_mock_llm, caplog):
        """Test that classification decisions are properly logged for audit."""
        classifier = classifier_with_mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Classification: HYBRID\nConfidence: 0.85\nReasoning: Analysis query"
        classifier._llm.ainvoke.return_value = mock_response
        
        with caplog.at_level(logging.INFO):
            result = await classifier.classify_query("Analyze user satisfaction with completion data")
        
        # Verify audit information is logged
        log_messages = [record.message for record in caplog.records]
        audit_logs = [msg for msg in log_messages if 'classification' in msg.lower()]
        
        assert len(audit_logs) > 0, "Should log classification decisions for audit"
        assert any('HYBRID' in log for log in audit_logs), "Should log classification result"
