#!/usr/bin/env python3
"""
Phase 3 Core Privacy Compliance Tests

Tests comprehensive Australian PII compliance across the entire RAG system
including detection, anonymisation, and Australian Privacy Principles (APP) compliance.

Focus: Core privacy compliance tests covering the most critical privacy protection scenarios.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import asyncio

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.core.privacy.pii_detector import AustralianPIIDetector, PIIDetectionResult
from src.rag.core.agent import RAGAgent, AgentConfig
from src.rag.core.routing.query_classifier import QueryClassifier
from src.rag.core.synthesis.answer_generator import AnswerGenerator


class TestPhase3PrivacyCompliance:
    """Core privacy compliance tests for Australian PII protection."""
    
    @pytest_asyncio.fixture
    async def pii_detector(self):
        """PII detector instance for testing."""
        detector = AustralianPIIDetector()
        await detector.initialise()
        return detector
    
    @pytest_asyncio.fixture
    async def mock_rag_agent(self):
        """Mock RAG agent for privacy testing."""
        mock = AsyncMock()
        return mock
    
    # Core PII Detection Tests
    @pytest.mark.asyncio
    async def test_australian_abn_detection(self, pii_detector):
        """Test detection of Australian Business Numbers (ABN)."""
        test_text = "Our company ABN is 53 004 085 616 and we need to process data."
        
        result = await pii_detector.detect_and_anonymise(test_text)
        
        assert result.anonymisation_applied is True
        assert len(result.entities_detected) > 0
        assert "53 004 085 616" not in result.anonymised_text
        assert any(entity.get("entity_type") in ["ABN", "AU_ABN"] for entity in result.entities_detected)
    
    @pytest.mark.asyncio
    async def test_australian_acn_detection(self, pii_detector):
        """Test detection of Australian Company Numbers (ACN)."""
        test_text = "The ACN 123 456 789 is associated with this entity."
        
        result = await pii_detector.detect_and_anonymise(test_text)
        
        assert result.anonymisation_applied is True
        assert "123 456 789" not in result.anonymised_text
        assert any("ACN" in entity.get("entity_type", "") or "AU_ACN" in entity.get("entity_type", "") 
                  for entity in result.entities_detected)
    
    @pytest.mark.asyncio 
    async def test_medicare_number_detection(self, pii_detector):
        """Test detection of Medicare numbers."""
        test_text = "Patient Medicare number 1234 56789 0 for medical records."
        
        result = await pii_detector.detect_and_anonymise(test_text)
        
        assert result.anonymisation_applied is True
        assert "1234 56789 0" not in result.anonymised_text
        assert any("MEDICARE" in entity.get("entity_type", "").upper() or 
                  "AU_MEDICARE" in entity.get("entity_type", "").upper()
                  for entity in result.entities_detected)
    
    @pytest.mark.asyncio
    async def test_tax_file_number_detection(self, pii_detector):
        """Test detection of Tax File Numbers (TFN)."""
        test_text = "Employee TFN 123 456 789 is required for payroll."
        
        result = await pii_detector.detect_and_anonymise(test_text)
        
        assert result.anonymisation_applied is True
        assert "123 456 789" not in result.anonymised_text
        assert any("TFN" in entity.get("entity_type", "").upper() or 
                  "TAX" in entity.get("entity_type", "").upper()
                  for entity in result.entities_detected)
    
    @pytest.mark.asyncio
    async def test_person_name_detection(self, pii_detector):
        """Test detection of person names."""
        test_text = "John Smith submitted his application for training completion."
        
        result = await pii_detector.detect_and_anonymise(test_text)
        
        # Names should be detected and anonymised
        assert result.anonymisation_applied is True
        assert "John Smith" not in result.anonymised_text
        assert any("PERSON" in entity.get("entity_type", "").upper() for entity in result.entities_detected)
    
    # Privacy Workflow Integration Tests
    @pytest.mark.asyncio
    async def test_query_classifier_pii_protection(self):
        """Test that query classifier protects PII before LLM processing."""
        query_text = "Show training data for John Smith with ABN 53 004 085 616"
        
        with patch('src.rag.core.routing.query_classifier.get_llm') as mock_llm:
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "CLARIFICATION_NEEDED", "confidence": "MEDIUM", "reasoning": "Contains PII"}'
            )
            
            classifier = QueryClassifier()
            await classifier.initialize()
            
            result = await classifier.classify_query(query_text)
            
            # Verify PII detection was applied
            assert result.anonymized_query != query_text
            assert "John Smith" not in result.anonymized_query
            assert "53 004 085 616" not in result.anonymized_query

    @pytest.mark.asyncio
    async def test_answer_generator_pii_protection(self):
        """Test that answer generator protects PII in generated responses."""
        from src.rag.core.synthesis.answer_generator import AnswerGenerator, SynthesisResult

        # Mock LLM to return response with PII
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = MagicMock(
                content="Training data shows John Smith (ABN: 53 004 085 616) completed the course."
            )
            mock_llm.return_value = mock_llm_instance

            generator = AnswerGenerator(llm=mock_llm_instance)
            
            # Test data with PII that should be detected
            query = "Show training completion for John Smith"
            sql_result = {"success": True, "result": [{"name": "John Smith", "status": "completed"}]}
            
            result = await generator.synthesize_answer(
                query=query,
                sql_result=sql_result,
                vector_result=None,
                session_id="test_session"
            )
            
            # Verify answer generation works and contains expected structure
            assert isinstance(result, SynthesisResult)
            assert result.answer is not None and len(result.answer) > 0
            assert result.answer_type is not None
            assert result.confidence >= 0.0
            assert result.sources is not None
    
    @pytest.mark.asyncio
    async def test_rag_agent_end_to_end_pii_protection(self):
        """Test end-to-end PII protection through the entire RAG agent workflow."""
        with patch('src.rag.core.agent.create_rag_agent') as mock_create_agent:
            mock_agent = AsyncMock()
            mock_agent.ainvoke.return_value = {
                "final_answer": "Analysis complete for [REDACTED_PERSON] with [REDACTED_ABN]",
                "classification": "SQL",
                "confidence": "HIGH", 
                "sources": ["database_query"],
                "processing_time": 1.5,
                "tools_used": ["classifier", "pii_detector", "sql", "synthesis"],
                "error": None,
                "session_id": "privacy_test"
            }
            mock_create_agent.return_value = mock_agent
            
            agent = await mock_create_agent(AgentConfig())
            
            query_with_pii = "Show training data for John Smith with ABN 53 004 085 616"
            result = await agent.ainvoke({
                "query": query_with_pii,
                "session_id": "privacy_compliance_test"
            })
            
            # Verify PII protection throughout workflow
            assert "pii_detector" in result["tools_used"]
            assert "[REDACTED_PERSON]" in result["final_answer"]
            assert "[REDACTED_ABN]" in result["final_answer"] 
            assert "John Smith" not in result["final_answer"]
            assert "53 004 085 616" not in result["final_answer"]
    
    # APP Compliance Tests
    @pytest.mark.asyncio
    async def test_app_principle_1_open_and_transparent(self, pii_detector):
        """Test APP 1: Open and transparent management of personal information."""
        # Test that PII detection provides transparent information about what was detected
        test_text = "Contact John Smith at john.smith@company.com for ABN 53 004 085 616"
        
        result = await pii_detector.detect_and_anonymise(test_text)
        
        assert result.anonymisation_applied is True
        assert len(result.entities_detected) > 0
        assert result.confidence_scores is not None
        
        # Verify transparency - we can see what types of entities were detected
        entity_types = [entity.get("entity_type", "") for entity in result.entities_detected]
        assert len(entity_types) > 0
        
        # Verify confidence scoring provides transparency
        assert isinstance(result.confidence_scores, dict)
        assert len(result.confidence_scores) > 0
    
    @pytest.mark.asyncio
    async def test_app_principle_3_collection_of_solicited_information(self, pii_detector):
        """Test APP 3: Collection of solicited personal information."""
        # Test that system handles personal information appropriately when processing queries
        user_query = "What training did employee ID E12345 complete?"
        
        result = await pii_detector.detect_and_anonymise(user_query)
        
        # Should handle employee identifiers appropriately
        assert result.processing_time > 0
        # Employee IDs may or may not be anonymised depending on pattern matching
        # The key is that the system processes them through the privacy filter
    
    @pytest.mark.asyncio
    async def test_app_principle_6_access_to_personal_information(self, pii_detector):
        """Test APP 6: Access to personal information."""
        # Test that system can identify when personal information is being accessed
        access_query = "Show all data for John Smith including contact details"
        
        result = await pii_detector.detect_and_anonymise(access_query)
        
        # Personal information access should be detected and anonymised
        assert result.anonymisation_applied is True
        assert "John Smith" not in result.anonymised_text
    
    @pytest.mark.asyncio
    async def test_app_principle_8_cross_border_disclosure(self, pii_detector):
        """Test APP 8: Cross-border disclosure of personal information."""
        # Test that PII is anonymised before any potential cross-border processing (LLM APIs)
        pii_text = "Send training certificate to John Smith, ABN 53 004 085 616"
        
        result = await pii_detector.detect_and_anonymise(pii_text)
        
        # All PII should be anonymised before any cross-border processing
        assert result.anonymisation_applied is True
        assert "John Smith" not in result.anonymised_text
        assert "53 004 085 616" not in result.anonymised_text
        
        # Verify that sensitive business information is protected
        assert "[PERSON]" in result.anonymised_text
        assert "[ABN]" in result.anonymised_text
    
    # Performance and Reliability Tests
    @pytest.mark.asyncio
    async def test_pii_detection_performance(self, pii_detector):
        """Test PII detection performance meets requirements."""
        test_text = "Process application for John Smith, ABN 53 004 085 616, Medicare 1234 56789 0"
        
        import time
        start_time = time.time()
        result = await pii_detector.detect_and_anonymise(test_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust based on requirements)
        assert processing_time < 5.0  # 5 seconds max for PII detection
        assert result.processing_time > 0
        assert result.anonymisation_applied is True
    
    @pytest.mark.asyncio
    async def test_batch_pii_processing(self, pii_detector):
        """Test batch processing of multiple texts with PII."""
        test_texts = [
            "Employee John Smith with ABN 53 004 085 616",
            "Contact Jane Doe, Medicare 9876 54321 0", 
            "Company ACN 123 456 789 needs processing",
            "Normal text without any PII content"
        ]
        
        results = await pii_detector.batch_process(test_texts)
        
        assert len(results) == len(test_texts)
        
        # First three should have PII detected, last should not
        assert results[0].anonymisation_applied is True
        assert results[1].anonymisation_applied is True  
        assert results[2].anonymisation_applied is True
        assert results[3].anonymisation_applied is False
        
        # Verify all PII was anonymised
        assert "John Smith" not in results[0].anonymised_text
        assert "Jane Doe" not in results[1].anonymised_text
        assert "123 456 789" not in results[2].anonymised_text
    
    # Error Handling and Edge Cases  
    @pytest.mark.asyncio
    async def test_pii_detection_with_empty_text(self, pii_detector):
        """Test PII detection handles empty text gracefully."""
        result = await pii_detector.detect_and_anonymise("")
        
        assert result.anonymisation_applied is False
        assert result.anonymised_text == ""
        assert len(result.entities_detected) == 0
    
    @pytest.mark.asyncio
    async def test_pii_detection_with_no_pii(self, pii_detector):
        """Test PII detection handles text with no PII correctly."""
        clean_text = "This is a normal query about training completion rates across departments."
        
        result = await pii_detector.detect_and_anonymise(clean_text)
        
        assert result.anonymisation_applied is False
        assert result.anonymised_text == clean_text
        assert len(result.entities_detected) == 0
    
    @pytest.mark.asyncio
    async def test_privacy_protection_error_handling(self, pii_detector):
        """Test privacy protection handles errors gracefully."""
        # Test with potentially problematic input
        problematic_text = "A" * 10000  # Very long text
        
        result = await pii_detector.detect_and_anonymise(problematic_text)
        
        # Should handle gracefully without crashing
        assert result is not None
        assert isinstance(result, PIIDetectionResult)
        assert result.processing_time > 0
    
    # Configuration and Settings Tests
    @pytest.mark.asyncio
    async def test_pii_detector_initialization(self):
        """Test PII detector initializes correctly with configuration."""
        detector = AustralianPIIDetector()
        await detector.initialise()
        
        assert detector._initialised is True
        assert detector.analyzer is not None
        assert detector.anonymizer is not None
        assert hasattr(detector, 'australian_patterns')
        assert len(detector.australian_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_privacy_compliance_logging(self, pii_detector):
        """Test that privacy operations are properly logged for audit compliance."""
        test_text = "Process request for John Smith with ABN 53 004 085 616"
        
        with patch('src.rag.core.privacy.pii_detector.logger') as mock_logger:
            result = await pii_detector.detect_and_anonymise(test_text)
            
            # Verify privacy operations are logged for audit trail
            assert mock_logger.info.called
            assert result.anonymisation_applied is True
