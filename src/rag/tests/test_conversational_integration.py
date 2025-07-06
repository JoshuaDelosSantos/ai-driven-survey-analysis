"""
Integration tests for Phase 3: Conversational Intelligence Integration

This module provides comprehensive tests for the integration of conversational
intelligence with the terminal application and agent workflow.

Tests cover:
- End-to-end conversational query processing
- Terminal app conversational interaction flows
- Agent routing for conversational vs data queries
- Feedback collection integration with pattern learning
- Performance and reliability of integrated system
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.interfaces.terminal_app import TerminalApp
from src.rag.core.agent import RAGAgent, AgentConfig, create_rag_agent
from src.rag.core.conversational.handler import ConversationalHandler, ConversationalPattern


@pytest.fixture
def terminal_app():
    """Create a terminal app for testing."""
    app = TerminalApp(enable_agent=True)
    # Mock the settings to avoid actual initialization
    app.settings.enable_feedback_collection = True
    app.settings.feedback_database_enabled = False  # Don't use real DB in tests
    return app


@pytest.fixture
def mock_agent():
    """Create a mock agent with conversational capabilities."""
    agent = Mock(spec=RAGAgent)
    agent._conversational_handler = ConversationalHandler()
    
    # Mock the ainvoke method to simulate agent responses
    async def mock_ainvoke(state):
        query = state.get("query", "")
        
        # Simulate different response types based on query
        if any(word in query.lower() for word in ['hello', 'hi', 'hey']):
            return {
                "query": query,
                "classification": "CONVERSATIONAL",
                "confidence": "HIGH",
                "final_answer": "G'day! I'm working well, thank you for asking. I'm here to help you analyse survey and training data from the Australian Public Service. How can I assist you today?",
                "sources": ["Conversational AI"],
                "tools_used": ["conversational"],
                "processing_time": 0.05,
                "requires_clarification": False,
                "error": None
            }
        elif any(word in query.lower() for word in ['what can you do', 'capabilities']):
            return {
                "query": query,
                "classification": "CONVERSATIONAL",
                "confidence": "HIGH",
                "final_answer": "I'm here to help you analyse survey data and training feedback from the Australian Public Service. I can:\n• Provide statistical analysis of survey responses\n• Search through user feedback and comments\n• Analyse training evaluations and learning outcomes\n• Generate insights from attendance and engagement data",
                "sources": ["Conversational AI"],
                "tools_used": ["conversational"],
                "processing_time": 0.08,
                "requires_clarification": False,
                "error": None
            }
        elif "satisfaction" in query.lower() and "users" in query.lower():
            return {
                "query": query,
                "classification": "SQL",
                "confidence": "HIGH",
                "final_answer": "Based on the analysis of 245 survey responses, the overall user satisfaction with training is 4.2/5.0. Key findings:\n• 78% of users rated their experience as 4 or 5 stars\n• Virtual learning scored 4.1/5.0 on average\n• Face-to-face training scored 4.3/5.0 on average",
                "sources": ["Database Analysis"],
                "tools_used": ["sql", "synthesis"],
                "processing_time": 1.23,
                "requires_clarification": False,
                "error": None
            }
        else:
            return {
                "query": query,
                "classification": "UNKNOWN",
                "confidence": "LOW",
                "final_answer": "I need clarification about your query. Could you please be more specific?",
                "sources": [],
                "tools_used": ["classifier"],
                "processing_time": 0.15,
                "requires_clarification": True,
                "error": None
            }
    
    agent.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    return agent


class TestConversationalIntegration:
    """Integration tests for conversational intelligence in the full system."""
    
    @pytest.mark.asyncio
    async def test_conversational_greeting_flow(self, terminal_app, mock_agent):
        """Test complete conversational greeting flow."""
        terminal_app.agent = mock_agent
        
        # Mock user input for feedback
        with patch('builtins.input', side_effect=['5', '']):  # Rating 5, no comment
            with patch('builtins.print') as mock_print:
                await terminal_app._process_question("Hello!")
        
        # Verify agent was called
        mock_agent.ainvoke.assert_called_once()
        call_args = mock_agent.ainvoke.call_args[0][0]
        assert call_args["query"] == "Hello!"
        assert call_args["session_id"] == terminal_app.session_id
        
        # Verify conversational response was displayed
        print_calls = [call.args[0] for call in mock_print.call_args_list if call.args]
        response_displayed = any("G'day!" in str(call) for call in print_calls)
        assert response_displayed, "Conversational greeting response should be displayed"
        
        # Verify conversational classification was shown
        classification_displayed = any("CONVERSATIONAL" in str(call) for call in print_calls)
        assert classification_displayed, "Conversational classification should be displayed"
    
    @pytest.mark.asyncio
    async def test_conversational_capabilities_query(self, terminal_app, mock_agent):
        """Test system capabilities query handling."""
        terminal_app.agent = mock_agent
        
        with patch('builtins.input', side_effect=['4', 'Very helpful']):  # Rating 4, with comment
            with patch('builtins.print') as mock_print:
                await terminal_app._process_question("What can you do?")
        
        # Verify agent processing
        mock_agent.ainvoke.assert_called_once()
        
        # Verify capabilities response content
        print_calls = [call.args[0] for call in mock_print.call_args_list if call.args]
        capabilities_mentioned = any("statistical analysis" in str(call).lower() for call in print_calls)
        assert capabilities_mentioned, "Capabilities should be mentioned in response"
        
        # Verify suggested queries are displayed for conversational responses
        suggestions_displayed = any("you might also be interested" in str(call).lower() for call in print_calls)
        assert suggestions_displayed, "Suggested queries should be displayed for conversational responses"
    
    @pytest.mark.asyncio
    async def test_conversational_feedback_integration(self, terminal_app, mock_agent):
        """Test that conversational feedback integrates with pattern learning."""
        terminal_app.agent = mock_agent
        
        # Test positive feedback
        with patch('builtins.input', side_effect=['5', 'Great response!']):
            await terminal_app._process_question("Hello there!")
        
        # Verify conversational handler received feedback
        # Note: In a real implementation, we'd check the pattern learning data
        # For this test, we verify the flow doesn't error
        assert terminal_app.query_count == 1
        assert len(terminal_app.feedback_collected) == 1
        
        # Test negative feedback
        with patch('builtins.input', side_effect=['2', 'Not very helpful']):
            await terminal_app._process_question("Hi!")
        
        assert terminal_app.query_count == 2
        assert len(terminal_app.feedback_collected) == 2
        
        # Verify both positive and negative feedback are tracked
        feedback_values = [f['helpful'] for f in terminal_app.feedback_collected.values()]
        assert True in feedback_values  # Positive feedback
        assert False in feedback_values  # Negative feedback
    
    @pytest.mark.asyncio
    async def test_conversational_display_formatting(self, terminal_app, mock_agent):
        """Test that conversational responses are displayed with proper formatting."""
        terminal_app.agent = mock_agent
        
        with patch('builtins.input', side_effect=['skip']):  # Skip feedback
            with patch('builtins.print') as mock_print:
                await terminal_app._process_question("Hello!")
        
        print_calls = [str(call.args[0]) if call.args else '' for call in mock_print.call_args_list]
        
        # Check for conversational-specific formatting
        conversational_header = any("💬 Conversational Response:" in call for call in print_calls)
        assert conversational_header, "Conversational responses should have special header"
        
        # Check for suggested queries
        suggestions = any("💡 You might also be interested in:" in call for call in print_calls)
        assert suggestions, "Conversational responses should include suggested queries"
        
        # Check for proper classification display
        classification = any("CONVERSATIONAL" in call for call in print_calls)
        assert classification, "Classification should be displayed"
    

class TestConversationalAgentIntegration:
    """Tests for conversational integration in the RAG agent."""
    
    @pytest.mark.asyncio
    async def test_agent_conversational_node_routing(self):
        """Test that conversational queries are routed to the conversational node."""
        # This would require a full agent initialization, so we'll use mocks
        
        # Mock the conversational handler
        mock_handler = Mock(spec=ConversationalHandler)
        mock_handler.handle_conversational_query.return_value = Mock(
            content="Test conversational response",
            confidence=0.95,
            pattern_type=ConversationalPattern.GREETING
        )
        
        # Create a simple test state
        test_state = {
            "query": "Hello!",
            "session_id": "test_session",
            "classification": "CONVERSATIONAL",
            "confidence": "HIGH",
            "tools_used": [],
            "start_time": 1234567890.0
        }
        
        # Test conversational node processing
        from src.rag.core.agent import RAGAgent
        agent = RAGAgent()
        agent._conversational_handler = mock_handler
        
        result = await agent._conversational_node(test_state)
        
        # Verify conversational processing
        assert result["classification"] == "CONVERSATIONAL"
        assert result["final_answer"] == "Test conversational response"
        assert "conversational" in result["tools_used"]
        assert result["sources"] == ["Conversational AI"]
        
        # Verify handler was called
        mock_handler.handle_conversational_query.assert_called_once_with("Hello!")
    
    @pytest.mark.asyncio
    async def test_agent_conversational_error_handling(self):
        """Test conversational node error handling."""
        # Mock handler that raises an error
        mock_handler = Mock(spec=ConversationalHandler)
        mock_handler.handle_conversational_query.side_effect = Exception("Test error")
        
        test_state = {
            "query": "Hello!",
            "session_id": "test_session",
            "classification": "CONVERSATIONAL",
            "confidence": "HIGH",
            "tools_used": [],
            "start_time": 1234567890.0
        }
        
        from src.rag.core.agent import RAGAgent
        agent = RAGAgent()
        agent._conversational_handler = mock_handler
        
        result = await agent._conversational_node(test_state)
        
        # Verify fallback response
        assert result["classification"] == "CONVERSATIONAL"
        assert "having trouble with conversational processing" in result["final_answer"]
        assert "conversational_error" in result["tools_used"]
        assert result["sources"] == ["Error Handler"]


class TestPerformanceAndReliability:
    """Tests for performance and reliability of conversational integration."""
    
    @pytest.mark.asyncio
    async def test_conversational_response_performance(self, terminal_app, mock_agent):
        """Test that conversational responses are fast."""
        terminal_app.agent = mock_agent
        
        import time
        start_time = time.time()
        
        with patch('builtins.input', side_effect=['skip']):
            await terminal_app._process_question("Hello!")
        
        processing_time = time.time() - start_time
        
        # Conversational responses should be very fast (< 1 second in test environment)
        assert processing_time < 1.0, f"Conversational response took {processing_time:.3f}s, should be < 1.0s"
    
    def test_conversational_pattern_recognition_accuracy(self):
        """Test accuracy of conversational pattern recognition."""
        handler = ConversationalHandler()
        
        test_cases = [
            ("Hello!", True, 0.8),
            ("What can you do?", True, 0.9),
            ("Thank you very much", True, 0.9),
            ("What's the weather like?", True, 0.9),
            ("How satisfied were users with training?", False, 0.0),  # Should NOT be conversational
            ("Show me completion rates by agency", False, 0.0),  # Should NOT be conversational
        ]
        
        for query, should_be_conversational, min_confidence in test_cases:
            is_conv, pattern, confidence = handler.is_conversational_query(query)
            
            if should_be_conversational:
                assert is_conv, f"Query '{query}' should be conversational"
                assert confidence >= min_confidence, f"Query '{query}' confidence {confidence} should be >= {min_confidence}"
            else:
                assert not is_conv, f"Query '{query}' should NOT be conversational"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
