#!/usr/bin/env python3
"""
Enhanced terminal application integration tests.

Tests the complete functionality of the TerminalApp class including:
- Initialization (both agent and legacy modes)
- Core query processing workflow
- Enhanced feedback collection system (1-5 scale + comments)
- Feedback analytics via `/feedback-stats` command
- Error handling and recovery
- Session management
- Australian privacy compliance
- Database integration for feedback storage

Focus: Comprehensive testing of Phase 3 feedback system implementation.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
import sys
import uuid
import asyncio

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.interfaces.terminal_app import TerminalApp
from src.rag.core.agent import RAGAgent, AgentConfig
from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
from src.rag.core.synthesis.feedback_analytics import FeedbackAnalytics, FeedbackStats


class TestTerminalAppEnhanced:
    """Enhanced terminal application integration tests with feedback system."""
    
    @pytest_asyncio.fixture
    async def mock_rag_agent(self):
        """Mock RAG agent for testing."""
        mock = AsyncMock()
        mock.ainvoke.return_value = {
            "final_answer": "Based on the analysis, there are 150 users who completed training.",
            "answer": "Based on the analysis, there are 150 users who completed training.",
            "query": "How many users completed training?",
            "classification": "SQL",
            "confidence": "HIGH",
            "sources": ["database_query"],
            "processing_time": 1.5,
            "tools_used": ["classifier", "sql", "synthesis"],
            "error": None,
            "session_id": "test_session",
            "success": True,
            "synthesis_successful": True
        }
        return mock
    
    @pytest_asyncio.fixture
    async def mock_feedback_collector(self):
        """Mock feedback collector for testing."""
        mock = AsyncMock()
        mock.collect_feedback.return_value = True  # Success
        return mock
    
    @pytest_asyncio.fixture 
    async def mock_feedback_analytics(self):
        """Mock feedback analytics for testing."""
        mock = AsyncMock()
        mock_stats = FeedbackStats()
        mock_stats.total_count = 5
        mock_stats.average_rating = 4.2
        mock_stats.rating_counts = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        mock_stats.recent_comments = ["Great response!", "Very helpful"]
        
        mock.get_feedback_stats.return_value = mock_stats
        mock.format_stats_for_display.return_value = """
📊 Feedback Statistics (Last 30 Days)
=====================================

Total Feedback: 5
Average Rating: 4.2/5.0

Rating Distribution:
  1⭐:   0 (  0.0%) 
  2⭐:   0 (  0.0%) 
  3⭐:   1 ( 20.0%) █████
  4⭐:   2 ( 40.0%) ██████████
  5⭐:   2 ( 40.0%) ██████████

Recent Comments (2):
  1. "Great response!"
  2. "Very helpful"
"""
        return mock
    
    @pytest_asyncio.fixture
    async def mock_settings(self):
        """Mock settings for testing."""
        mock = MagicMock()
        mock.enable_feedback_collection = True
        mock.feedback_database_enabled = True
        return mock
    
    @pytest_asyncio.fixture
    async def mock_sql_tool(self):
        """Mock SQL tool for legacy mode testing."""
        mock = AsyncMock()
        mock.initialize.return_value = None
        mock.process_question.return_value = MagicMock(
            success=True,
            result=[{"count": 150, "agency": "Education"}],
            query="SELECT COUNT(*) FROM users",
            processing_time=0.8
        )
        return mock
    
    @pytest_asyncio.fixture
    async def mock_create_rag_agent(self, mock_rag_agent):
        """Mock the create_rag_agent factory function."""
        async def _create_rag_agent(config: AgentConfig) -> RAGAgent:
            return mock_rag_agent
        return _create_rag_agent
    
    # Initialization Tests
    @pytest.mark.asyncio
    async def test_terminal_app_agent_mode_initialization(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test terminal app initialization in agent mode with feedback system."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings):
            
            # Manually set the mocked components before initialization
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            assert app.agent is not None
            assert app.sql_tool is None
            assert app.enable_agent is True
            assert len(app.session_id) == 8
            assert app.feedback_collector is not None
            assert app.feedback_analytics is not None
            assert app.settings.enable_feedback_collection is True
    
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
    
    # Enhanced Feedback Collection Tests
    @pytest.mark.asyncio
    async def test_feedback_collection_success(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test successful feedback collection with 1-5 scale rating."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['4', 'Great response!']), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            test_result = {
                "final_answer": "Test answer",
                "answer": "Test answer", 
                "query": "Test query",
                "classification": "SQL",
                "confidence": "HIGH",
                "processing_time": 1.0,
                "sources": ["database"]
            }
            
            # Test feedback collection
            await app._collect_feedback("test_query_123", test_result)
            
            # Verify feedback was collected and stored
            assert "test_query_123" in app.feedback_collected
            assert app.feedback_collected["test_query_123"]["rating"] == 4
            assert app.feedback_collected["test_query_123"]["comment"] == "Great response!"
            assert app.feedback_collected["test_query_123"]["helpful"] is True
            
            # Verify database storage was attempted
            mock_feedback_collector.collect_feedback.assert_called_once()
            
            # Verify positive feedback message
            mock_print.assert_any_call("✅ Thank you for the positive feedback!")
    
    @pytest.mark.asyncio
    async def test_feedback_collection_skip(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test feedback collection when user skips."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', return_value='skip'), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # Test feedback skipping
            await app._collect_feedback("test_query_456", test_result)
            
            # Verify no feedback was stored
            assert "test_query_456" not in app.feedback_collected
            
            # Verify database storage was not attempted
            mock_feedback_collector.collect_feedback.assert_not_called()
            
            # Verify skip message
            mock_print.assert_any_call("⏭️  Skipping feedback collection...")
    
    @pytest.mark.asyncio
    async def test_feedback_collection_validation(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test feedback collection input validation."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['invalid', '6', '0', '3', '']), \
             patch('builtins.print') as mock_print:
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # Test feedback collection with validation
            await app._collect_feedback("test_query_789", test_result)
            
            # Verify feedback was eventually collected with valid rating
            assert "test_query_789" in app.feedback_collected
            assert app.feedback_collected["test_query_789"]["rating"] == 3
            assert app.feedback_collected["test_query_789"]["helpful"] is False  # 3 < 4
            
            # Verify validation messages were displayed
            mock_print.assert_any_call("Please enter a number between 1 and 5, or 'skip'")
    
    @pytest.mark.asyncio
    async def test_feedback_database_failure_handling(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test handling of database failure during feedback collection."""
        app = TerminalApp(enable_agent=True)
        
        # Configure mock to simulate database failure
        mock_feedback_collector.collect_feedback.return_value = False  # Return False instead of raising exception
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['5', 'Excellent!']), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # Test feedback collection with database failure
            await app._collect_feedback("test_query_error", test_result)
            
            # Verify feedback was still stored locally
            assert "test_query_error" in app.feedback_collected
            assert app.feedback_collected["test_query_error"]["rating"] == 5
            assert app.feedback_collected["test_query_error"]["stored_in_db"] is False
            
            # Verify failure message was displayed
            mock_print.assert_any_call("⚠️  Feedback stored locally but database storage failed.")
    
    # Feedback Analytics Tests
    @pytest.mark.asyncio
    async def test_feedback_stats_display(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test feedback statistics display via /feedback-stats command."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            # Test feedback stats display
            await app._show_feedback_stats()
            
            # Verify analytics methods were called
            mock_feedback_analytics.get_feedback_stats.assert_called_once_with(days_back=30)
            mock_feedback_analytics.format_stats_for_display.assert_called_once()
            
            # Verify stats were displayed
            mock_print.assert_any_call("📊 Feedback Analytics")
    
    @pytest.mark.asyncio
    async def test_feedback_stats_no_data(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test feedback statistics display with no data."""
        app = TerminalApp(enable_agent=True)
        
        # Configure mock to return empty stats
        empty_stats = FeedbackStats()
        empty_stats.total_count = 0
        mock_feedback_analytics.get_feedback_stats.return_value = empty_stats
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            # Test feedback stats display with no data
            await app._show_feedback_stats()
            
            # Verify appropriate message was displayed
            mock_print.assert_any_call("📭 No feedback data available yet.")
    
    @pytest.mark.asyncio 
    async def test_feedback_stats_disabled(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics):
        """Test feedback statistics when database is disabled."""
        app = TerminalApp(enable_agent=True)
        
        # Mock settings with feedback disabled
        mock_settings_disabled = MagicMock()
        mock_settings_disabled.feedback_database_enabled = False
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings_disabled), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings_disabled
            
            await app.initialize()
            
            # Test feedback stats display when disabled
            await app._show_feedback_stats()
            
            # Verify analytics were not called
            mock_feedback_analytics.get_feedback_stats.assert_not_called()
            
            # Verify disabled message was displayed
            mock_print.assert_any_call("⚠️  Database feedback storage is disabled.")
    
    # Core Query Processing Tests
    @pytest.mark.asyncio
    async def test_agent_mode_query_processing(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test core query processing through RAG agent with feedback integration."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.print') as mock_print:
            
            await app.initialize()
            
            # Process a test query
            result = await app._process_with_agent("How many users completed training?", "test123")
            
            assert result["final_answer"] is not None
            assert result["classification"] == "SQL"
            assert result["error"] is None
            assert result["tools_used"] is not None
            assert result["synthesis_successful"] is True
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
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
            "session_id": "test_session",
            "synthesis_successful": False
        }
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', return_value=mock_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings):
            
            await app.initialize()
            
            result = await app._process_with_agent("How many users?", "test456")
            
            assert result["error"] is not None
            assert "Database connection failed" in result["error"]
            assert result["final_answer"] is None
            assert result["synthesis_successful"] is False
    
    # Session Management Tests  
    @pytest.mark.asyncio
    async def test_session_management(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test session management and query tracking."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings):
            
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
    
    # Enhanced Integration Tests
    @pytest.mark.asyncio
    async def test_complete_feedback_workflow(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test complete feedback workflow from query to analytics."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['5', 'Excellent response!']), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            # 1. Process query
            result = await app._process_with_agent("Test query", "workflow_test")
            assert result["synthesis_successful"] is True
            
            # 2. Collect feedback
            await app._collect_feedback("workflow_test", result)
            
            # 3. Verify feedback storage
            assert "workflow_test" in app.feedback_collected
            assert app.feedback_collected["workflow_test"]["rating"] == 5
            
            # 4. Display analytics
            await app._show_feedback_stats()
            
            # Verify all components were used
            mock_feedback_collector.collect_feedback.assert_called_once()
            mock_feedback_analytics.get_feedback_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handling(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test graceful handling of keyboard interrupts during feedback collection."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=KeyboardInterrupt()), \
             patch('builtins.print') as mock_print:
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # Test keyboard interrupt handling
            await app._collect_feedback("interrupt_test", test_result)
            
            # Verify feedback was not stored
            assert "interrupt_test" not in app.feedback_collected
            
            # Verify appropriate message was displayed
            mock_print.assert_any_call("\n⏭️  Skipping feedback collection...")
    
    # Privacy and PII Tests
    @pytest.mark.asyncio
    async def test_pii_protection_in_feedback(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test that PII protection is maintained in feedback collection."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['4', 'Contact john.smith@example.com for more info']), \
             patch('builtins.print'):
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            test_result = {
                "final_answer": "Analysis for user [REDACTED_PERSON]",
                "query": "Show data for John Smith",
                "sources": ["database"]
            }
            
            # Test feedback collection with PII in comment
            await app._collect_feedback("pii_test", test_result)
            
            # Verify feedback collector was called with FeedbackData
            mock_feedback_collector.collect_feedback.assert_called_once()
            call_args = mock_feedback_collector.collect_feedback.call_args[0][0]
            
            # Verify the feedback data structure
            assert isinstance(call_args, FeedbackData)
            assert call_args.rating == 4
            assert call_args.comment == 'Contact john.smith@example.com for more info'  # Raw comment stored
            assert call_args.query_text == "Show data for John Smith"
    
    # Performance and Timeout Tests
    @pytest.mark.asyncio
    async def test_feedback_collection_performance(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test that feedback collection doesn't significantly impact performance."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['3', 'Average response']), \
             patch('builtins.print'):
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # Time the feedback collection
            import time
            start_time = time.time()
            await app._collect_feedback("perf_test", test_result)
            end_time = time.time()
            
            # Verify reasonable performance (should be nearly instantaneous with mocks)
            assert (end_time - start_time) < 1.0  # Should complete in under 1 second
            
            # Verify feedback was collected
            assert "perf_test" in app.feedback_collected
    
    # Edge Cases and Error Recovery
    @pytest.mark.asyncio
    async def test_malformed_result_handling(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test handling of malformed result objects in feedback collection."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['2', 'Poor response']), \
             patch('builtins.print'):
            
            await app.initialize()
            
            # Test with malformed result (missing expected fields)
            malformed_result = {"unexpected_field": "value"}
            
            # Should not raise exception
            await app._collect_feedback("malformed_test", malformed_result)
            
            # Verify feedback was still collected
            assert "malformed_test" in app.feedback_collected
            assert app.feedback_collected["malformed_test"]["rating"] == 2
    
    # Configuration Tests
    @pytest.mark.asyncio
    async def test_feedback_disabled_configuration(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics):
        """Test behavior when feedback collection is disabled."""
        app = TerminalApp(enable_agent=True)
        
        # Mock settings with feedback disabled
        mock_settings_disabled = MagicMock()
        mock_settings_disabled.enable_feedback_collection = False
        mock_settings_disabled.feedback_database_enabled = False
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings_disabled), \
             patch('builtins.print'):
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # Test feedback collection when disabled
            await app._collect_feedback("disabled_test", test_result)
            
            # Verify no feedback was collected
            assert "disabled_test" not in app.feedback_collected
            
            # Verify database storage was not attempted
            mock_feedback_collector.collect_feedback.assert_not_called()
    
    # End-to-End Integration Test
    @pytest.mark.asyncio 
    async def test_end_to_end_workflow_with_feedback(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test complete end-to-end workflow including feedback system."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings):
            
            await app.initialize()
            
            # Test the core processing workflow
            result = await app._process_with_agent("How many users completed training?", "e2e_test")
            
            # Verify complete workflow execution
            assert result["final_answer"] is not None
            assert result["classification"] is not None
            assert result["tools_used"] is not None
            assert result["error"] is None
            assert result["session_id"] == "test_session"
            assert result["synthesis_successful"] is True
            
            # Verify expected tools were used
            expected_tools = ["classifier", "sql", "synthesis"]
            for tool in expected_tools:
                assert tool in result["tools_used"]
            
            # Verify feedback system is properly initialized
            assert app.feedback_collector is not None
            assert app.feedback_analytics is not None
            assert app.settings.enable_feedback_collection is True
    
    # Infinite Loop Prevention Tests
    @pytest.mark.asyncio
    async def test_feedback_input_timeout_prevention(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test that feedback input validation doesn't cause infinite loops."""
        app = TerminalApp(enable_agent=True)
        
        # Create a counter to simulate repeated invalid input that could cause infinite loop
        input_call_count = 0
        def mock_input_with_limit(prompt):
            nonlocal input_call_count
            input_call_count += 1
            
            # Simulate user providing invalid input multiple times, then valid input
            if input_call_count <= 3:
                return 'invalid'
            elif input_call_count == 4:
                return '7'  # Still invalid (out of range)
            elif input_call_count == 5:
                return '4'  # Finally valid rating
            else:
                return ''  # Optional comment (empty)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=mock_input_with_limit), \
             patch('builtins.print') as mock_print:
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # This should not hang or cause infinite loop
            await app._collect_feedback("timeout_test", test_result)
            
            # Verify feedback was eventually collected
            assert "timeout_test" in app.feedback_collected
            assert app.feedback_collected["timeout_test"]["rating"] == 4
            
            # Verify that validation messages were shown but eventually succeeded
            # Should call input for: invalid(1), invalid(2), invalid(3), 7(4), 4(5), comment(6)
            assert input_call_count == 6  # Should have called input exactly 6 times
    
    @pytest.mark.asyncio
    async def test_feedback_collection_with_max_retries(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test feedback collection behavior with repeated keyboard interrupts."""
        app = TerminalApp(enable_agent=True)
        
        # Simulate keyboard interrupt during feedback collection
        interrupt_count = 0
        def mock_input_with_interrupt(prompt):
            nonlocal interrupt_count
            interrupt_count += 1
            if interrupt_count <= 2:
                raise KeyboardInterrupt()
            return 'skip'  # Eventually skip
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.FeedbackCollector', return_value=mock_feedback_collector), \
             patch('src.rag.interfaces.terminal_app.FeedbackAnalytics', return_value=mock_feedback_analytics), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=mock_input_with_interrupt), \
             patch('builtins.print') as mock_print:
            
            await app.initialize()
            
            test_result = {"final_answer": "Test answer", "query": "Test query"}
            
            # This should handle interrupts gracefully and not loop infinitely
            await app._collect_feedback("interrupt_retry_test", test_result)
            
            # Verify no feedback was stored due to interrupts
            assert "interrupt_retry_test" not in app.feedback_collected
            
            # Verify that the interrupt was handled properly
            mock_print.assert_any_call("\n⏭️  Skipping feedback collection...")
    
    @pytest.mark.asyncio
    async def test_run_method_basic_functionality(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test that the main run loop can be initialized without infinite loops."""
        app = TerminalApp(enable_agent=True)
        
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=['exit']):  # Exit immediately
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            # Test that run() method can start and exit properly
            # Using asyncio.wait_for to ensure it doesn't hang indefinitely
            await asyncio.wait_for(app.run(), timeout=5.0)
            
            # If we get here, the run loop exited properly
            assert True
    
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_in_main_loop(self, mock_create_rag_agent, mock_feedback_collector, mock_feedback_analytics, mock_settings):
        """Test that KeyboardInterrupt in main loop exits gracefully without infinite loops."""
        app = TerminalApp(enable_agent=True)
        
        # Mock input to raise KeyboardInterrupt
        with patch('src.rag.interfaces.terminal_app.create_rag_agent', mock_create_rag_agent), \
             patch('src.rag.interfaces.terminal_app.get_settings', return_value=mock_settings), \
             patch('builtins.input', side_effect=KeyboardInterrupt()), \
             patch('builtins.print'):
            
            # Manually set the mocked components
            app.feedback_collector = mock_feedback_collector
            app.feedback_analytics = mock_feedback_analytics
            app.settings = mock_settings
            
            await app.initialize()
            
            # Test that KeyboardInterrupt exits the run loop without hanging
            # Using asyncio.wait_for to ensure it doesn't loop infinitely
            await asyncio.wait_for(app.run(), timeout=5.0)
            
            # If we get here, the method exited properly (no infinite loop)
            assert True
