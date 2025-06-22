"""
Phase 3 Integration Tests
End-to-end integration tests for the complete RAG system.
Focus on essential core integration scenarios only.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.rag.core.agent import RAGAgent
from src.rag.interfaces.terminal_app import TerminalApp
from src.rag.core.privacy.pii_detector import AustralianPIIDetector


@pytest.mark.asyncio
class TestPhase3Integration:
    """Core integration tests for complete RAG system."""

    async def test_end_to_end_sql_query_workflow(self):
        """Test complete workflow for SQL-based queries."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool:
            
            # Mock LLM for classification
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "Statistical query"}'
            )
            
            # Mock SQL tool
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.return_value = {
                "success": True,
                "result": [{"count": 150, "percentage": 85.5}],
                "query": "SELECT COUNT(*) as count, AVG(completion_rate) as percentage FROM training_data",
                "explanation": "Training completion statistics"
            }
            mock_sql_tool.return_value = mock_sql_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Test SQL query workflow
            query = "How many users completed the training program?"
            initial_state = {
                "query": query,
                "session_id": "integration_test",
                "retry_count": 0,
                "tools_used": [],
                "requires_clarification": False
            }
            result = await agent.ainvoke(initial_state)
            
            # Verify end-to-end workflow
            assert result is not None
            assert "final_answer" in result
            assert result["final_answer"] is not None
            assert len(result["final_answer"]) > 0
            assert "classifier" in result["tools_used"]  # Classifier should always be used
            assert result["error"] is None or "SQL tool failed" in result.get("error", "")
            
            # Verify SQL tool was called
            mock_sql_instance.process_query.assert_called_once()

    async def test_end_to_end_vector_query_workflow(self):
        """Test complete workflow for vector search queries."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.vector_search.vector_search_tool.VectorSearchTool') as mock_vector_tool:
            
            # Mock LLM for classification
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "VECTOR", "confidence": "HIGH", "reasoning": "Feedback query"}'
            )
            
            # Mock vector search tool
            mock_vector_instance = AsyncMock()
            mock_vector_instance.initialize.return_value = None
            mock_vector_instance.search.return_value = {
                "success": True,
                "results": [
                    {"text": "The training was excellent and very helpful", "score": 0.95},
                    {"text": "Great content and well-structured modules", "score": 0.92}
                ],
                "total_results": 2
            }
            mock_vector_tool.return_value = mock_vector_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Test vector search workflow
            query = "What feedback did participants provide about the training?"
            initial_state = {
                "query": query,
                "session_id": "integration_test",
                "retry_count": 0,
                "tools_used": [],
                "requires_clarification": False
            }
            result = await agent.ainvoke(initial_state)
            
            # Verify end-to-end workflow
            assert result is not None
            assert "final_answer" in result
            assert result["final_answer"] is not None
            assert len(result["final_answer"]) > 0
            assert "classifier" in result["tools_used"]
            # Vector search might work or fail, but should attempt processing
            
            # Verify vector tool was called
            mock_vector_instance.search.assert_called_once()

    async def test_end_to_end_hybrid_query_workflow(self):
        """Test complete workflow for hybrid queries requiring both tools."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool, \
             patch('src.rag.core.vector_search.vector_search_tool.VectorSearchTool') as mock_vector_tool:
            
            # Mock LLM for classification and synthesis
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "HYBRID", "confidence": "HIGH", "reasoning": "Needs both stats and feedback"}'
            )
            
            # Mock SQL tool
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.return_value = {
                "success": True,
                "result": [{"completion_rate": 89.5, "satisfaction_score": 4.2}],
                "query": "SELECT AVG(completion_rate), AVG(satisfaction) FROM training_metrics"
            }
            mock_sql_tool.return_value = mock_sql_instance
            
            # Mock vector search tool
            mock_vector_instance = AsyncMock()
            mock_vector_instance.initialize.return_value = None
            mock_vector_instance.search.return_value = {
                "success": True,
                "results": [
                    {"text": "Training quality exceeded expectations", "score": 0.94},
                    {"text": "Comprehensive and well-organized content", "score": 0.91}
                ]
            }
            mock_vector_tool.return_value = mock_vector_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Test hybrid workflow
            query = "Analyze the training effectiveness combining completion rates with participant feedback"
            initial_state = {
                "query": query,
                "session_id": "integration_test",
                "retry_count": 0,
                "tools_used": [],
                "requires_clarification": False
            }
            result = await agent.ainvoke(initial_state)
            
            # Verify hybrid workflow
            assert result is not None
            assert "final_answer" in result
            assert result["final_answer"] is not None
            assert len(result["final_answer"]) > 0
            assert "classifier" in result["tools_used"]
            # Hybrid processing should attempt to use both tools
            
            # Verify both tools were called
            mock_sql_instance.process_query.assert_called_once()
            mock_vector_instance.search.assert_called_once()

    async def test_terminal_app_integration(self):
        """Test terminal application integration with RAG agent."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool:
            
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "Count query"}'
            )
            
            # Mock SQL tool
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.return_value = {
                "success": True,
                "result": [{"total_users": 250}],
                "query": "SELECT COUNT(*) as total_users FROM users"
            }
            mock_sql_tool.return_value = mock_sql_instance
            
            # Initialize terminal app
            terminal_app = TerminalApp()
            await terminal_app.initialize()
            
            # Process query through terminal app
            response = await terminal_app._process_with_agent("How many users are in the system?", "test_query_id")
            
            # Verify terminal app integration
            assert response is not None
            assert "final_answer" in response
            assert response["final_answer"] is not None and len(response["final_answer"]) > 0
            
            # Verify internal state
            assert terminal_app.query_count > 0
            assert terminal_app.session_id is not None

    async def test_privacy_compliance_integration(self):
        """Test privacy compliance throughout the complete workflow."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool:
            
            # Mock LLM (will receive anonymized query)
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = MagicMock(
                content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "User data query"}'
            )
            mock_llm.return_value = mock_llm_instance
            
            # Mock SQL tool
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.return_value = {
                "success": True,
                "result": [{"completion_status": "completed", "score": 95}],
                "query": "SELECT completion_status, score FROM training_data WHERE user_id = 'user123'"
            }
            mock_sql_tool.return_value = mock_sql_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Test with PII-containing query
            pii_query = "Show training results for John Smith with ABN 53 004 085 616"
            initial_state = {
                "query": pii_query,
                "session_id": "pii_test",
                "retry_count": 0,
                "tools_used": [],
                "requires_clarification": False
            }
            result = await agent.ainvoke(initial_state)
            
            # Verify privacy protection
            assert result is not None
            assert result["error"] is None  # Should not error due to PII
            
            # Verify LLM received anonymized query (check call arguments)
            llm_calls = mock_llm_instance.ainvoke.call_args_list
            assert len(llm_calls) > 0
            
            # Check that original PII was not passed to LLM
            for call in llm_calls:
                call_content = str(call)
                assert "John Smith" not in call_content
                assert "53 004 085 616" not in call_content

    async def test_error_recovery_integration(self):
        """Test error recovery across the complete system."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool:
            
            # Mock LLM to succeed
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "Database query"}'
            )
            
            # Mock SQL tool to fail initially
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.side_effect = Exception("Database connection failed")
            mock_sql_tool.return_value = mock_sql_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Test error recovery
            query = "Show user statistics"
            initial_state = {
                "query": query,
                "session_id": "error_test",
                "retry_count": 0,
                "tools_used": [],
                "requires_clarification": False
            }
            result = await agent.ainvoke(initial_state)
            
            # Verify graceful error handling
            assert result is not None
            assert result["error"] is not None  # Error should be captured
            assert "final_answer" in result
            assert result["final_answer"] is not None  # Should have fallback response
            assert len(result["final_answer"]) > 0
            assert result["tools_used"] == ["sql_failed"]

    async def test_session_management_integration(self):
        """Test session management across multiple queries."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool:
            
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "Data query"}'
            )
            
            # Mock SQL tool
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.return_value = {
                "success": True,
                "result": [{"value": 42}],
                "query": "SELECT 42 as value"
            }
            mock_sql_tool.return_value = mock_sql_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Test multiple queries in same session
            session_id = "session_test_123"
            queries = [
                "Show user count",
                "Display completion rates", 
                "Get satisfaction scores"
            ]
            
            results = []
            for query in queries:
                initial_state = {
                    "query": query,
                    "session_id": session_id,
                    "retry_count": 0,
                    "tools_used": [],
                    "requires_clarification": False
                }
                result = await agent.ainvoke(initial_state)
                results.append(result)
            
            # Verify session consistency
            assert len(results) == 3
            for result in results:
                assert result is not None
                assert result["error"] is None
                assert "final_answer" in result
                assert result["final_answer"] is not None
            
            # All queries should use the same session
            assert all(result.get("session_id") == session_id for result in results if "session_id" in result)

    async def test_concurrent_request_handling(self):
        """Test system handling of concurrent requests."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm, \
             patch('src.rag.core.text_to_sql.sql_tool.AsyncSQLTool') as mock_sql_tool:
            
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "Concurrent query"}'
            )
            
            # Mock SQL tool
            mock_sql_instance = AsyncMock()
            mock_sql_instance.initialize.return_value = None
            mock_sql_instance.process_query.return_value = {
                "success": True,
                "result": [{"concurrent_result": "success"}],
                "query": "SELECT 'success' as concurrent_result"
            }
            mock_sql_tool.return_value = mock_sql_instance
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Create concurrent queries
            concurrent_queries = [
                ("What is the user count?", "concurrent_1"),
                ("Show completion rates", "concurrent_2"),
                ("Display satisfaction data", "concurrent_3"),
                ("Get training metrics", "concurrent_4"),
                ("Analyze performance", "concurrent_5")
            ]
            
            # Execute all queries concurrently
            tasks = [
                agent.ainvoke({
                    "query": query,
                    "session_id": session_id,
                    "retry_count": 0,
                    "tools_used": [],
                    "requires_clarification": False
                })
                for query, session_id in concurrent_queries
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify concurrent processing
            assert len(results) == 5
            
            # Check for exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Concurrent processing failed: {exceptions}"
            
            # Verify all results are valid
            valid_results = [r for r in results if not isinstance(r, Exception)]
            assert len(valid_results) == 5
            
            for result in valid_results:
                assert result is not None
                assert "final_answer" in result
                assert result["final_answer"] is not None
