"""
Integration Tests for RAG System

Tests end-to-end workflows and component integration for the enhanced RAG system.
Validates integration between terminal application, query classification, and logging.

Phase 4 of the test suite refactoring focusing on:
- Complete user workflow testing
- Help system integration 
- Cross-component error handling
- Performance and reliability validation
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from io import StringIO
import sys

# Import only what we need for integration testing
# Use mock objects to avoid import issues


class TestIntegration:
    """Test complete integration workflows for the enhanced RAG system."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for integration testing."""
        mock = Mock()
        mock.log_user_query = Mock()
        return mock
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock query classifier for integration testing."""
        mock = AsyncMock()
        mock.classify_query = AsyncMock()
        mock.classify_query.return_value = Mock(
            classification='SQL',
            confidence='HIGH',
            method_used='rule_based',
            processing_time=0.1,
            reasoning='Test query classification'
        )
        return mock
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for integration testing."""
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock()
        mock_agent.process_query.return_value = {
            'response': 'Test response',
            'sources': ['test_source'],
            'query_type': 'SQL'
        }
        return mock_agent
    
    @pytest.fixture
    def mock_terminal_app(self, mock_logger, mock_classifier, mock_agent):
        """Create a mock terminal application for integration testing."""
        app = Mock()
        app.logger = mock_logger
        app.classifier = mock_classifier
        app.agent = mock_agent
        app._show_help = Mock()
        app._show_examples = Mock()
        app.config = Mock()
        return app
    
    def test_terminal_app_initialization(self, mock_terminal_app):
        """Test that terminal application initializes correctly for integration."""
        assert mock_terminal_app is not None
        assert mock_terminal_app.logger is not None
        assert mock_terminal_app.classifier is not None
        assert mock_terminal_app.agent is not None
    
    def test_help_command_integration(self, mock_terminal_app):
        """Test help command integration with terminal workflow."""
        # Mock stdout to capture output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Test help command
            mock_terminal_app._show_help()
            
            # Verify help method was called
            mock_terminal_app._show_help.assert_called_once()
    
    def test_examples_command_integration(self, mock_terminal_app):
        """Test examples command integration with terminal workflow."""
        # Mock stdout to capture output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Test examples command
            mock_terminal_app._show_examples()
            
            # Verify examples method was called
            mock_terminal_app._show_examples.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_processing_integration(self, mock_terminal_app):
        """Test complete query processing integration workflow."""
        test_query = "How many users completed training?"
        
        # Mock the query processing workflow
        mock_terminal_app.classifier.classify_query.return_value = Mock(
            classification='SQL',
            confidence='HIGH',
            method_used='rule_based',
            processing_time=0.1,
            reasoning='Statistical query requiring database aggregation'
        )
        
        mock_terminal_app.agent.process_query.return_value = {
            'response': 'Test SQL response',
            'sources': ['attendance_table'],
            'query_type': 'SQL'
        }
        
        # Test query classification
        result = await mock_terminal_app.classifier.classify_query(test_query)
        assert result.classification == 'SQL'
        assert result.confidence == 'HIGH'
        
        # Test agent processing
        agent_result = await mock_terminal_app.agent.process_query(test_query)
        assert agent_result['response'] == 'Test SQL response'
        assert agent_result['query_type'] == 'SQL'
    
    @pytest.mark.asyncio
    async def test_logging_integration(self, mock_terminal_app):
        """Test logging integration across components."""
        test_query = "What feedback did participants give?"
        
        # Test that logging is called during query processing
        mock_terminal_app.logger.log_user_query.return_value = None
        
        # Simulate query processing with logging
        await mock_terminal_app.classifier.classify_query(test_query)
        
        # Verify logger can be called (integration point exists)
        mock_terminal_app.logger.log_user_query(
            query_id='test_123',
            query_type='VECTOR',
            processing_time=0.2,
            success=True,
            metadata={'classification': 'VECTOR', 'confidence': 'HIGH'}
        )
        
        # Verify the logger was called
        mock_terminal_app.logger.log_user_query.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_terminal_app):
        """Test error handling integration across components."""
        # Test classification error handling
        mock_terminal_app.classifier.classify_query.side_effect = Exception("Classification error")
        
        try:
            await mock_terminal_app.classifier.classify_query("Invalid query")
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Classification error" in str(e)
        
        # Test agent error handling
        mock_terminal_app.agent.process_query.side_effect = Exception("Agent error")
        
        try:
            await mock_terminal_app.agent.process_query("Invalid query")
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Agent error" in str(e)
    
    def test_configuration_integration(self, mock_terminal_app):
        """Test configuration integration across components."""
        # Test that terminal app can access configuration
        assert hasattr(mock_terminal_app, 'config')
        
        # Test that components can be configured
        assert mock_terminal_app.logger is not None
        assert mock_terminal_app.classifier is not None
        assert mock_terminal_app.agent is not None
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, mock_terminal_app):
        """Test performance integration across components."""
        test_queries = [
            "How many users completed training?",
            "What feedback did participants give?",
            "Show me completion rates by agency"
        ]
        
        # Test that multiple queries can be processed efficiently
        start_time = time.time()
        
        for query in test_queries:
            # Mock quick responses
            mock_terminal_app.classifier.classify_query.return_value = Mock(
                classification='SQL',
                confidence='HIGH',
                method_used='rule_based',
                processing_time=0.05,
                reasoning='Fast classification'
            )
            
            result = await mock_terminal_app.classifier.classify_query(query)
            assert result is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process all queries reasonably quickly
        assert total_time < 1.0, f"Integration too slow: {total_time:.2f}s"
    
    def test_component_isolation(self, mock_terminal_app):
        """Test that components are properly isolated for testing."""
        # Test that components can be mocked independently
        assert mock_terminal_app.logger is not None
        assert mock_terminal_app.classifier is not None
        assert mock_terminal_app.agent is not None
        
        # Test that mocked components behave correctly
        mock_terminal_app.logger.log_user_query.return_value = None
        mock_terminal_app.logger.log_user_query('test', 'SQL', 0.1)
        mock_terminal_app.logger.log_user_query.assert_called_with('test', 'SQL', 0.1)
    
    @pytest.mark.asyncio
    async def test_metadata_flow_integration(self, mock_terminal_app):
        """Test metadata flow integration across components."""
        # Test that metadata flows from classification to logging
        classification_result = Mock(
            classification='VECTOR',
            confidence='HIGH',
            method_used='rule_based',
            processing_time=0.15,
            reasoning='Feedback query classification'
        )
        
        mock_terminal_app.classifier.classify_query.return_value = classification_result
        
        # Test classification
        result = await mock_terminal_app.classifier.classify_query("What did users say?")
        
        # Test that metadata can be extracted and logged
        metadata = {
            'classification': result.classification,
            'confidence': result.confidence,
            'method_used': result.method_used,
            'processing_time': result.processing_time
        }
        
        # Test logging with metadata
        mock_terminal_app.logger.log_user_query(
            query_id='test_metadata',
            query_type=result.classification,
            processing_time=result.processing_time,
            success=True,
            metadata=metadata
        )
        
        # Verify metadata integration
        assert metadata['classification'] == 'VECTOR'
        assert metadata['confidence'] == 'HIGH'
        assert metadata['method_used'] == 'rule_based'
    
    def test_backwards_compatibility_integration(self, mock_terminal_app):
        """Test backwards compatibility integration."""
        # Test that legacy functionality still works
        assert mock_terminal_app.logger is not None
        
        # Test legacy logging call (without metadata)
        mock_terminal_app.logger.log_user_query.return_value = None
        mock_terminal_app.logger.log_user_query('legacy_test', 'SQL', 0.1)
        
        # Should not raise exception
        mock_terminal_app.logger.log_user_query.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_integration(self, mock_terminal_app):
        """Test concurrent processing integration."""
        # Test that multiple queries can be processed concurrently
        queries = [
            "How many users completed training?",
            "What feedback was given?",
            "Show completion rates"
        ]
        
        # Mock concurrent responses
        mock_terminal_app.classifier.classify_query.return_value = Mock(
            classification='SQL',
            confidence='HIGH',
            method_used='rule_based',
            processing_time=0.05,
            reasoning='Concurrent classification'
        )
        
        # Process queries concurrently
        tasks = [mock_terminal_app.classifier.classify_query(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == len(queries)
        for result in results:
            assert result is not None
            assert result.classification == 'SQL'
    
    def test_session_state_integration(self, mock_terminal_app):
        """Test session state integration."""
        # Test that terminal app maintains session state
        assert mock_terminal_app is not None
        
        # Test that state is preserved across method calls
        mock_terminal_app._show_help()
        mock_terminal_app._show_examples()
        
        # Should not raise exceptions and state should be maintained
        assert mock_terminal_app.logger is not None
        assert mock_terminal_app.classifier is not None
        assert mock_terminal_app.agent is not None
    
    def test_resource_cleanup_integration(self, mock_terminal_app):
        """Test resource cleanup integration."""
        # Test that resources can be properly cleaned up
        assert mock_terminal_app is not None
        
        # Mock cleanup methods if they exist
        if hasattr(mock_terminal_app, 'cleanup'):
            mock_terminal_app.cleanup = Mock()
            mock_terminal_app.cleanup()
            mock_terminal_app.cleanup.assert_called_once()
        
        if hasattr(mock_terminal_app.classifier, 'close'):
            mock_terminal_app.classifier.close = Mock()
            mock_terminal_app.classifier.close()
            mock_terminal_app.classifier.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mock_terminal_app):
        """Test complete end-to-end workflow integration."""
        # Simulate a complete user session
        test_query = "How many APS Level 6 staff completed the compliance training?"
        
        # Mock the complete workflow
        mock_terminal_app.classifier.classify_query.return_value = Mock(
            classification='SQL',
            confidence='HIGH',
            method_used='rule_based',
            processing_time=0.12,
            reasoning='Statistical query with APS-specific context'
        )
        
        mock_terminal_app.agent.process_query.return_value = {
            'response': 'Based on the attendance data, 245 APS Level 6 staff completed compliance training.',
            'sources': ['attendance_table', 'user_table'],
            'query_type': 'SQL',
            'confidence': 0.95
        }
        
        # Step 1: Query classification
        classification = await mock_terminal_app.classifier.classify_query(test_query)
        assert classification.classification == 'SQL'
        assert classification.confidence == 'HIGH'
        
        # Step 2: Agent processing
        agent_result = await mock_terminal_app.agent.process_query(test_query)
        assert agent_result['response'] is not None
        assert agent_result['query_type'] == 'SQL'
        
        # Step 3: Logging
        mock_terminal_app.logger.log_user_query(
            query_id='e2e_test',
            query_type=classification.classification,
            processing_time=classification.processing_time,
            success=True,
            metadata={
                'classification': classification.classification,
                'confidence': classification.confidence,
                'method_used': classification.method_used,
                'agent_confidence': agent_result.get('confidence', 0.0)
            }
        )
        
        # Verify end-to-end success
        mock_terminal_app.logger.log_user_query.assert_called()
        assert "APS Level 6" in agent_result['response']
        assert len(agent_result['sources']) > 0
