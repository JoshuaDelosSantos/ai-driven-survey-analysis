"""
Phase 3 Integration Tests
End-to-end integration tests for the complete RAG system.
Focus on essential core integration scenarios only.
"""

import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Mock the db module before any imports that depend on it
sys.modules['db'] = MagicMock()
sys.modules['db.db_connector'] = MagicMock()

from src.rag.core.agent import RAGAgent
from src.rag.interfaces.terminal_app import TerminalApp
from src.rag.core.privacy.pii_detector import AustralianPIIDetector
from src.rag.core.vector_search.search_result import VectorSearchResponse, VectorSearchResult
from src.rag.core.text_to_sql.sql_tool import SQLResult


@pytest.mark.asyncio
class TestPhase3Integration:
    """Core integration tests for complete RAG system."""

    async def test_end_to_end_sql_query_workflow(self):
        """Test complete workflow for SQL-based queries."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            
            # Mock LLM for classification
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Statistical query about training completion'
            )
            
            # Initialize RAG agent first
            agent = RAGAgent()
            await agent.initialize()
            
            # Now patch the agent's SQL tool instance
            mock_sql_result = SQLResult(
                query="SELECT COUNT(*) as count, AVG(completion_rate) as percentage FROM training_data",
                result=[{"count": 150, "percentage": 85.5}],
                execution_time=0.5,
                success=True,
                row_count=1
            )
            agent._sql_tool.process_question = AsyncMock(return_value=mock_sql_result)
            
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
            agent._sql_tool.process_question.assert_called_once()

    async def test_terminal_app_integration(self):
        """Test terminal application integration with RAG agent."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: User count query'
            )
            
            # Initialize terminal app
            terminal_app = TerminalApp()
            await terminal_app.initialize()
            
            # Now patch the terminal app's agent's SQL tool instance
            mock_sql_result = SQLResult(
                query="SELECT COUNT(*) as total_users FROM users",
                result=[{"total_users": 250}],
                execution_time=0.2,
                success=True,
                row_count=1
            )
            terminal_app.agent._sql_tool.process_question = AsyncMock(return_value=mock_sql_result)
            
            # Process query through terminal app
            response = await terminal_app._process_with_agent("How many users are in the system?", "test_query_id")
            
            # Verify terminal app integration
            assert response is not None
            assert "final_answer" in response
            assert response["final_answer"] is not None and len(response["final_answer"]) > 0
            
            # Verify internal state (session should be initialized)
            assert terminal_app.session_id is not None



    async def test_error_recovery_integration(self):
        """Test error recovery across the complete system."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            
            # Mock LLM to succeed
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Database statistics query'
            )
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Mock SQL tool to fail
            agent._sql_tool.process_question = AsyncMock(side_effect=Exception("Database connection failed"))
            
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
            assert result.get("error") is None  # Agent handles errors internally, doesn't expose them
            assert "final_answer" in result
            assert result["final_answer"] is not None  # Should have fallback response
            assert len(result["final_answer"]) > 0
            # Should show retry in tools_used when SQL fails
            assert any("retry" in tool for tool in result["tools_used"])

    async def test_session_management_integration(self):
        """Test session management across multiple queries."""
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Database query for session management'
            )
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Mock SQL tool
            mock_sql_result = SQLResult(
                query="SELECT 42 as value",
                result=[{"value": 42}],
                execution_time=0.1,
                success=True,
                row_count=1
            )
            agent._sql_tool.process_question = AsyncMock(return_value=mock_sql_result)
            
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
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Concurrent database query processing'
            )
            
            # Initialize RAG agent
            agent = RAGAgent()
            await agent.initialize()
            
            # Mock SQL tool
            mock_sql_result = SQLResult(
                query="SELECT 'success' as concurrent_result",
                result=[{"concurrent_result": "success"}],
                execution_time=0.05,
                success=True,
                row_count=1
            )
            agent._sql_tool.process_question = AsyncMock(return_value=mock_sql_result)
            
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
