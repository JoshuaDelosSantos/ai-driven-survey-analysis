"""
Phase 3 Performance Tests
Tests performance characteristics and load handling of the RAG system.
Focus on essential core performance metrics only.
"""

import pytest
import asyncio
import time
import psutil
import gc
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.rag.core.agent import RAGAgent
from src.rag.core.routing.query_classifier import QueryClassifier
from src.rag.core.synthesis.answer_generator import AnswerGenerator
from src.rag.core.privacy.pii_detector import AustralianPIIDetector


@pytest.mark.asyncio
class TestPhase3Performance:
    """Core performance tests for RAG system components."""

    async def test_query_classification_performance(self):
        """Test query classification performance under normal load."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Create a proper mock response that mimics LangChain's AIMessage
            mock_response = MagicMock()
            mock_response.content = 'Classification: SQL\nConfidence: HIGH\nReasoning: Statistical query'
            
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = mock_response
            
            classifier = QueryClassifier()
            await classifier.initialize()
            
            # Test multiple classifications for performance
            queries = [
                "How many users completed training?",
                "What did people say about the course?",
                "Show completion rates by department",
                "Analyze satisfaction scores and feedback",
                "Count total participants"
            ]
            
            start_time = time.time()
            results = []
            
            for query in queries:
                result = await classifier.classify_query(query, session_id="perf_test")
                results.append(result)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(queries)
            
            # Performance assertions (allow for initialization overhead)
            assert total_time < 30.0  # Total time under 30 seconds (increased for first-time setup)
            assert avg_time < 6.0     # Average under 6 seconds per query
            assert len(results) == len(queries)
            
            # Verify all classifications completed successfully
            for result in results:
                assert result.classification is not None
                assert result.confidence is not None

    async def test_pii_detection_batch_performance(self):
        """Test PII detection performance with batch processing."""
        detector = AustralianPIIDetector()
        await detector.initialise()
        
        # Test data with various PII types
        test_texts = [
            "Contact John Smith at john@example.com",
            "Company ABN: 53 004 085 616 needs training",
            "Medicare number 2428 4567 8901 2345 on file",
            "Tax file number 123 456 789 for employee",
            "No PII in this text about course completion",
            "Multiple items: Sarah Johnson, ABN 12 345 678 901, TFN 987 654 321"
        ]
        
        start_time = time.time()
        results = []
        
        for text in test_texts:
            result = await detector.detect_and_anonymise(text)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_texts)
        
        # Performance assertions
        assert total_time < 10.0   # Total batch processing under 10 seconds
        assert avg_time < 2.0      # Average under 2 seconds per text
        assert len(results) == len(test_texts)
        
        # Verify processing quality
        pii_detected_count = sum(1 for r in results if r.anonymisation_applied)
        assert pii_detected_count >= 4  # At least 4 texts should have PII detected

    async def test_answer_synthesis_performance(self):
        """Test answer synthesis performance with different data types."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = MagicMock(
                content="Statistical analysis shows 85% completion rate with positive feedback themes."
            )
            mock_llm.return_value = mock_llm_instance
            
            generator = AnswerGenerator(llm=mock_llm_instance)
            
            # Test different synthesis scenarios
            test_scenarios = [
                {
                    "query": "Show completion statistics",
                    "sql_result": {"success": True, "result": [{"count": 150, "percentage": 85}]},
                    "vector_result": None
                },
                {
                    "query": "What feedback did users provide?",
                    "sql_result": None,
                    "vector_result": {"success": True, "results": [{"text": "Great course", "score": 0.9}]}
                },
                {
                    "query": "Analyze overall training effectiveness",
                    "sql_result": {"success": True, "result": [{"completion": 85, "satisfaction": 4.2}]},
                    "vector_result": {"success": True, "results": [{"text": "Excellent content", "score": 0.95}]}
                }
            ]
            
            start_time = time.time()
            results = []
            
            for scenario in test_scenarios:
                result = await generator.synthesize_answer(
                    query=scenario["query"],
                    sql_result=scenario["sql_result"],
                    vector_result=scenario["vector_result"],
                    session_id="perf_test"
                )
                results.append(result)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(test_scenarios)
            
            # Performance assertions
            assert total_time < 12.0   # Total synthesis under 12 seconds
            assert avg_time < 4.0      # Average under 4 seconds per synthesis
            assert len(results) == len(test_scenarios)
            
            # Verify synthesis quality
            for result in results:
                assert result.answer is not None and len(result.answer) > 0
                assert result.confidence >= 0.0
                assert result.processing_time is not None

    async def test_memory_usage_stability(self):
        """Test memory usage remains stable under repeated operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Create a proper mock response that mimics LangChain's AIMessage
            mock_response = MagicMock()
            mock_response.content = 'Classification: SQL\nConfidence: HIGH\nReasoning: Test'
            
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = mock_response
            
            classifier = QueryClassifier()
            await classifier.initialize()
            
            # Perform repeated operations
            for i in range(20):
                await classifier.classify_query(f"Test query {i}", session_id=f"mem_test_{i}")
                
                # Check memory every 5 iterations
                if i % 5 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable (under 50MB)
                    assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB at iteration {i}"
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        # Final memory check
        assert total_increase < 100, f"Total memory increase: {total_increase:.1f}MB"

    async def test_concurrent_processing_performance(self):
        """Test system performance under concurrent load."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Create a proper mock response that mimics LangChain's AIMessage
            mock_response = MagicMock()
            mock_response.content = 'Classification: VECTOR\nConfidence: HIGH\nReasoning: Feedback query'
            
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = mock_response
            
            classifier = QueryClassifier()
            await classifier.initialize()
            
            # Create concurrent classification tasks
            queries = [f"What did users think about module {i}?" for i in range(10)]
            
            start_time = time.time()
            
            # Execute all queries concurrently
            tasks = [
                classifier.classify_query(query, session_id=f"concurrent_{i}")
                for i, query in enumerate(queries)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Performance assertions for concurrent processing
            assert total_time < 10.0  # All concurrent operations under 10 seconds
            assert len(results) == len(queries)
            
            # Verify no exceptions occurred
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Exceptions in concurrent processing: {exceptions}"
            
            # Verify all results are valid
            valid_results = [r for r in results if not isinstance(r, Exception)]
            assert all(hasattr(r, 'classification') for r in valid_results)

    async def test_error_recovery_performance(self):
        """Test performance when recovering from errors."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Simulate intermittent LLM failures
            failure_count = 0
            
            def mock_ainvoke(*args, **kwargs):
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 2:  # First 2 calls fail
                    raise Exception("Simulated LLM failure")
                # Create a proper mock response that mimics LangChain's AIMessage
                mock_response = MagicMock()
                mock_response.content = 'Classification: SQL\nConfidence: MEDIUM\nReasoning: Recovered'
                return mock_response
            
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.side_effect = mock_ainvoke
            
            classifier = QueryClassifier()
            await classifier.initialize()
            
            start_time = time.time()
            
            # Test multiple queries with error recovery
            results = []
            for i in range(5):
                try:
                    result = await classifier.classify_query(
                        f"Test query {i}", 
                        session_id=f"error_test_{i}"
                    )
                    results.append(result)
                except Exception as e:
                    # Some failures are expected
                    results.append(None)
            
            total_time = time.time() - start_time
            
            # Performance assertions for error scenarios
            assert total_time < 20.0  # Error handling should not cause excessive delays
            
            # At least some queries should succeed after initial failures
            successful_results = [r for r in results if r is not None]
            assert len(successful_results) >= 2, "Should have some successful recoveries"

    async def test_system_resource_efficiency(self):
        """Test overall system resource usage efficiency."""
        # Force garbage collection before measuring
        gc.collect()
        
        initial_cpu_percent = psutil.cpu_percent(interval=1)
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        detector = AustralianPIIDetector()
        await detector.initialise()
        
        # Perform resource-intensive operations
        test_operations = [
            "Analyze training completion data for John Smith (ABN: 53 004 085 616)",
            "Review feedback from participants including Sarah Johnson (TFN: 123 456 789)",
            "Process evaluation scores and Medicare number 2428 4567 8901 2345",
            "Generate report for compliance with Australian Privacy Principles"
        ]
        
        start_time = time.time()
        
        for text in test_operations:
            await detector.detect_and_anonymise(text)
        
        processing_time = time.time() - start_time
        
        # Force garbage collection before final measurement
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_cpu_percent = psutil.cpu_percent(interval=1)
        
        # Resource efficiency assertions
        assert processing_time < 15.0  # All operations under 15 seconds
        
        memory_increase = final_memory - initial_memory
        # Allow for ML model loading (Presidio/SpaCy) - memory increase varies with system state
        assert memory_increase < 80, f"Memory increase should be reasonable: {memory_increase:.1f}MB"
        
        # CPU usage should not spike excessively
        cpu_increase = final_cpu_percent - initial_cpu_percent
        assert cpu_increase < 50, f"CPU usage increase: {cpu_increase:.1f}%"
