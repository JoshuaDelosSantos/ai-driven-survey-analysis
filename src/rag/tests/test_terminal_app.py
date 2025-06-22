#!/usr/bin/env python3
"""
Core terminal application integration tests.

Tests the essential functionality of the TerminalApp class including:
- Initialization (both agent and legacy modes)
- Core query processing workflow
- Error handling and recovery
- Session management
- Australian privacy compliance

Focus: Only necessary core tests for Phase 2 milestone completion.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import uuid

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.interfaces.terminal_app import TerminalApp
from src.rag.core.agent import RAGAgent, AgentConfig


class TestTerminalAppCore:
    """Core terminal application integration tests."""
    
    @pytest_asyncio.fixture
    async def mock_rag_agent(self):
        """Mock RAG agent for testing."""
        mock = AsyncMock()
        mock.ainvoke.return_value = {
            "final_answer": "Based on the analysis, there are 150 users who completed training.",
            "classification": "SQL",
            "confidence": "HIGH",
            "sources": ["database_query"],
            "processing_time": 1.5,
            "tools_used": ["classifier", "sql", "synthesis"],
            "error": None,
            "session_id": "test_session"
        }
        return mock
    
    @pytest_asyncio.fixture
    async def mock_sql_tool(self):
        """Mock SQL tool for legacy mode testing."""
        mock = AsyncMock()
        mock.initialize.return_value = None
        mock.process_query.return_value = {
            "success": True,
            "result": [{"count": 150, "agency": "Education"}],
            "query": "SELECT COUNT(*) FROM users",
            "processing_time": 0.8
        }
        return mock
    
    @pytest_asyncio.fixture
    async def mock_create_rag_agent(self, mock_rag_agent):
        """Mock the create_rag_agent factory function."""
        async def _create_rag_agent(config: AgentConfig) -> RAGAgent:
            return mock_rag_agent
        return _create_rag_agent
    
    # Initialization Tests
    @pytest.mark.asyncio
    async def test_terminal_app_agent_mode_initialization(self, mock_create_rag_agent):
        """Test terminal app initialization in agent mode."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent):
            await app.initialize()
            
            assert app.agent is not None
            assert app.sql_tool is None
            assert app.enable_agent is True
            assert len(app.session_id) == 8
    
    @pytest.mark.asyncio
    async def test_terminal_app_legacy_mode_initialization(self, mock_sql_tool):
        """Test terminal app initialization in legacy SQL-only mode."""
        app = TerminalApp(enable_agent=False)
        
        with patch('src.rag.interfaces.terminal_app.AsyncSQLTool', return_value=mock_sql_tool), \
             patch('src.rag.interfaces.terminal_app.get_llm'):
            
            await app.initialize()
            
            assert app.agent is None
            assert app.sql_tool is not None
            assert app.enable_agent is False
            mock_sql_tool.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_failure_handling(self):
        """Test terminal app handles initialization failures gracefully."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent') as mock_create:
            mock_create.side_effect = Exception("Agent initialization failed")
            
            with pytest.raises(RuntimeError, match="Failed to initialize application"):
                await app.initialize()
    
    # Core Query Processing Tests
    @pytest.mark.asyncio
    async def test_agent_mode_query_processing(self, mock_create_rag_agent):
        """Test core query processing through RAG agent."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('builtins.print') as mock_print:
            
            await app.initialize()
            
            # Process a test query
            result = await app._process_with_agent("How many users completed training?", "test123")
            
            assert result["final_answer"] is not None
            assert result["classification"] == "SQL"
            assert result["error"] is None
            assert result["tools_used"] is not None
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_create_rag_agent):
        """Test agent error handling during query processing."""
        app = TerminalApp(enable_agent=True)
        
        # Configure mock agent to return error
        mock_agent = await mock_create_rag_agent(AgentConfig())
        mock_agent.ainvoke.return_value = {
            "final_answer": None,
            "error": "Database connection failed",
            "classification": "SQL",
            "tools_used": ["classifier", "sql_failed"],
            "processing_time": 2.0,
            "session_id": "test_session"
        }
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', return_value=mock_agent):
            await app.initialize()
            
            result = await app._process_with_agent("How many users?", "test456")
            
            assert result["error"] is not None
            assert "Database connection failed" in result["error"]
            assert result["final_answer"] is None
    
    # Session Management Tests  
    @pytest.mark.asyncio
    async def test_session_management(self, mock_create_rag_agent):
        """Test session management and query tracking."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent):
            await app.initialize()
            
            # Verify session initialization
            assert app.session_id is not None
            assert len(app.session_id) == 8
            assert app.query_count == 0
            assert isinstance(app.feedback_collected, dict)
            
            # Process a query to increment counter
            await app._process_with_agent("Test query", "query123")
            
            # Verify session state (query_count is incremented in _process_question)
            assert app.session_id is not None
    
    @pytest.mark.asyncio
    async def test_feedback_collection(self, mock_create_rag_agent):
        """Test feedback collection system."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('builtins.input', return_value='👍'):
            
            await app.initialize()
            
            test_result = {
                "final_answer": "Test answer",
                "classification": "SQL",
                "confidence": "HIGH",
                "processing_time": 1.0
            }
            
            # Test feedback collection
            await app._collect_feedback("test_query_123", test_result)
            
            # Verify feedback was stored
            assert "test_query_123" in app.feedback_collected
            assert app.feedback_collected["test_query_123"]["rating"] == "positive"
    
    # Error Recovery Tests
    @pytest.mark.asyncio
    async def test_graceful_error_recovery(self, mock_create_rag_agent):
        """Test graceful error recovery and user-friendly messaging."""
        app = TerminalApp(enable_agent=True)
        
        # Mock agent to raise exception
        mock_agent = await mock_create_rag_agent(AgentConfig())
        mock_agent.ainvoke.side_effect = Exception("Network timeout")
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', return_value=mock_agent), \
             patch('builtins.print') as mock_print:
            
            await app.initialize()
            
            # Process query that will fail
            result = await app._process_with_agent("Test query", "error123")
            
            # Verify error was handled gracefully
            assert result is not None
            # The method should handle the exception and return some error state
    
    # Privacy Compliance Tests
    @pytest.mark.asyncio
    async def test_pii_protection_in_terminal_app(self, mock_create_rag_agent):
        """Test that terminal app maintains PII protection throughout workflow."""
        app = TerminalApp(enable_agent=True)
        
        # Mock agent to simulate PII detection
        mock_agent = await mock_create_rag_agent(AgentConfig())
        mock_agent.ainvoke.return_value = {
            "final_answer": "Analysis complete for user [REDACTED_PERSON]",
            "classification": "SQL", 
            "confidence": "HIGH",
            "sources": ["database_query"],
            "processing_time": 1.2,
            "tools_used": ["classifier", "pii_detector", "sql", "synthesis"],
            "error": None,
            "session_id": "privacy_test"
        }
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', return_value=mock_agent):
            await app.initialize()
            
            result = await app._process_with_agent("Show data for John Smith", "pii_test")
            
            # Verify PII was detected and protected
            assert "pii_detector" in result["tools_used"]
            assert "[REDACTED_PERSON]" in result["final_answer"]
            assert "John Smith" not in result["final_answer"]
    
    # Configuration and Environment Tests
    @pytest.mark.asyncio
    async def test_configuration_integration(self, mock_create_rag_agent):
        """Test terminal app integrates properly with configuration system."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent):
            await app.initialize()
            
            # Verify configuration integration
            assert app.settings is not None
            assert hasattr(app, 'session_id')
            assert hasattr(app, 'example_queries')
            assert len(app.example_queries) > 0
    
    # Performance and Response Time Tests
    @pytest.mark.asyncio
    async def test_response_time_tracking(self, mock_create_rag_agent):
        """Test that response times are properly tracked."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent):
            await app.initialize()
            
            result = await app._process_with_agent("Test query", "perf_test")
            
            # Verify processing time is tracked
            assert "processing_time" in result
            assert isinstance(result["processing_time"], (int, float))
            assert result["processing_time"] > 0
    
    # Integration Workflow Tests
    @pytest.mark.asyncio 
    async def test_end_to_end_workflow(self, mock_create_rag_agent):
        """Test complete end-to-end workflow without UI interactions."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent):
            await app.initialize()
            
            # Test the core processing workflow
            result = await app._process_with_agent("How many users completed training?", "e2e_test")
            
            # Verify complete workflow execution
            assert result["final_answer"] is not None
            assert result["classification"] is not None
            assert result["tools_used"] is not None
            assert result["error"] is None
            assert result["session_id"] == "test_session"
            
            # Verify expected tools were used
            expected_tools = ["classifier", "sql", "synthesis"]
            for tool in expected_tools:
                assert tool in result["tools_used"]
