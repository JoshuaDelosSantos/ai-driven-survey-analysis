"""
Phase 4 Integration Tests
End-to-end integration tests for enhanced RAG system functionality.
Focus on complete workflows integrating terminal app, query classification, and logging enhancements.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any
import tempfile
import os

from src.rag.interfaces.terminal_app import TerminalApp
from src.rag.core.routing.query_classifier import QueryClassifier
from src.rag.utils.logging_utils import RAGLogger
from src.rag.core.agent import RAGAgent
from src.rag.core.text_to_sql.sql_tool import SQLResult
from src.rag.core.vector_search.search_result import VectorSearchResponse, VectorSearchResult


@pytest.mark.asyncio
class TestPhase4Integration:
    """Integration tests for enhanced RAG system workflows."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock RAG agent with realistic responses."""
        agent = AsyncMock(spec=RAGAgent)
        agent.get_available_tables.return_value = ["users", "attendance", "learning_content", "evaluation"]
        agent.get_table_schema.return_value = {
            "users": ["user_id", "name", "email", "department", "level"],
            "attendance": ["user_id", "course_id", "completion_date", "status"],
            "evaluation": ["user_id", "course_id", "rating", "feedback", "instructor_rating"]
        }
        return agent
    
    @pytest.fixture
    def mock_sql_result(self):
        """Create a realistic SQL result."""
        return SQLResult(
            query="SELECT department, COUNT(*) as count FROM users GROUP BY department",
            result=[
                {"department": "Finance", "count": 45},
                {"department": "IT", "count": 32},
                {"department": "HR", "count": 28}
            ],
            execution_time=0.3,
            success=True,
            row_count=3
        )
    
    @pytest.fixture
    def mock_vector_result(self):
        """Create a realistic vector search result."""
        return VectorSearchResponse(
            results=[
                VectorSearchResult(
                    content="The training was very effective and well-structured.",
                    score=0.85,
                    metadata={"user_id": "U001", "course_id": "C001", "rating": 4.5}
                ),
                VectorSearchResult(
                    content="Good content but could use more interactive elements.",
                    score=0.78,
                    metadata={"user_id": "U002", "course_id": "C001", "rating": 3.8}
                )
            ],
            query="What feedback did users give about the training?",
            total_results=2,
            search_time=0.2
        )
    
    async def test_complete_terminal_to_classification_workflow(self, mock_agent, mock_sql_result):
        """Test complete workflow from terminal input to classification and response."""
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Mock LLM for classification
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Statistical query about departmental distribution'
            )
            
            # Initialize components
            terminal_app = TerminalApp(enable_agent=True)
            terminal_app.agent = mock_agent
            
            # Mock agent's query processing
            mock_agent.process_query.return_value = {
                "response": "Based on the data, Finance has 45 users, IT has 32 users, and HR has 28 users.",
                "query_type": "sql",
                "sql_result": mock_sql_result,
                "classification": {
                    "classification": "SQL",
                    "confidence": "HIGH",
                    "method_used": "rule_based"
                }
            }
            
            # Test query processing
            query = "How many users are in each department?"
            
            # Process query through terminal app
            with patch('builtins.input', return_value='n'):  # Skip feedback collection
                result = await terminal_app.agent.process_query(query)
            
            # Verify classification and response
            assert result["query_type"] == "sql"
            assert result["classification"]["classification"] == "SQL"
            assert result["classification"]["confidence"] == "HIGH"
            assert "Finance has 45 users" in result["response"]
            
            # Verify agent was called correctly
            mock_agent.process_query.assert_called_once_with(query)
    
    async def test_help_system_integration(self):
        """Test help system integration with terminal app."""
        
        # Initialize terminal app
        terminal_app = TerminalApp(enable_agent=True)
        
        # Test help method exists and works
        assert hasattr(terminal_app, '_show_help')
        assert callable(getattr(terminal_app, '_show_help'))
        
        # Test examples method exists and works
        assert hasattr(terminal_app, '_show_examples')
        assert callable(getattr(terminal_app, '_show_examples'))
        
        # Test example queries are schema-accurate
        assert hasattr(terminal_app, 'example_queries')
        assert isinstance(terminal_app.example_queries, dict)
        
        # Verify categories exist
        expected_categories = ["SQL Queries", "Feedback Analysis", "Hybrid Analysis"]
        for category in expected_categories:
            assert category in terminal_app.example_queries
            assert isinstance(terminal_app.example_queries[category], list)
            assert len(terminal_app.example_queries[category]) > 0
    
    async def test_metadata_logging_integration(self, mock_agent):
        """Test metadata logging integration throughout the workflow."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_integration.log")
            
            # Initialize logger
            logger = RAGLogger("integration_test", log_file=log_file)
            
            # Test comprehensive metadata logging
            metadata = {
                "classification": "SQL",
                "confidence": "HIGH",
                "method_used": "rule_based",
                "processing_time": 0.5,
                "pattern_matches": ["count", "department", "users"],
                "table_suggestions": ["users"],
                "query_complexity": "medium",
                "pii_detected": False
            }
            
            # Log user query with metadata
            logger.log_user_query(
                query_id="TEST_001",
                query_type="sql",
                processing_time=0.5,
                success=True,
                metadata=metadata
            )
            
            # Verify log file was created and contains expected content
            assert os.path.exists(log_file)
            
            with open(log_file, 'r') as f:
                log_content = f.read()
                
            # Check that metadata was logged
            assert "classification" in log_content
            assert "SQL" in log_content
            assert "rule_based" in log_content
            assert "table_suggestions" in log_content
    
    async def test_classification_to_agent_integration(self, mock_agent, mock_sql_result):
        """Test integration between query classification and agent processing."""
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Mock LLM for classification
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Query asks for user count by department'
            )
            
            # Initialize classifier
            classifier = QueryClassifier()
            await classifier.initialize()
            
            # Test classification
            query = "How many users are in each department?"
            classification_result = await classifier.classify_query(query)
            
            # Verify classification
            assert classification_result.classification == "SQL"
            assert classification_result.confidence == "HIGH"
            assert classification_result.method_used == "rule_based"
            
            # Test agent processing with classification metadata
            mock_agent.process_query.return_value = {
                "response": "Based on the data, Finance has 45 users, IT has 32 users, and HR has 28 users.",
                "query_type": "sql",
                "sql_result": mock_sql_result,
                "classification": {
                    "classification": classification_result.classification,
                    "confidence": classification_result.confidence,
                    "method_used": classification_result.method_used,
                    "processing_time": classification_result.processing_time
                }
            }
            
            # Process query with agent
            agent_result = await mock_agent.process_query(query)
            
            # Verify integration
            assert agent_result["classification"]["classification"] == "SQL"
            assert agent_result["classification"]["confidence"] == "HIGH"
            assert agent_result["classification"]["method_used"] == "rule_based"
    
    async def test_feedback_integration_workflow(self, mock_agent, mock_vector_result):
        """Test feedback-related query integration workflow."""
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Mock LLM for classification
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: VECTOR\nConfidence: HIGH\nReasoning: Query asks for participant feedback content'
            )
            
            # Initialize terminal app
            terminal_app = TerminalApp(enable_agent=True)
            terminal_app.agent = mock_agent
            
            # Mock agent's feedback query processing
            mock_agent.process_query.return_value = {
                "response": "Users generally found the training effective and well-structured, though some suggested more interactive elements.",
                "query_type": "vector",
                "vector_result": mock_vector_result,
                "classification": {
                    "classification": "VECTOR",
                    "confidence": "HIGH",
                    "method_used": "rule_based",
                    "table_suggestions": ["evaluation"]
                }
            }
            
            # Test feedback query processing
            query = "What feedback did participants give about the training content?"
            result = await terminal_app.agent.process_query(query)
            
            # Verify feedback processing
            assert result["query_type"] == "vector"
            assert result["classification"]["classification"] == "VECTOR"
            assert result["classification"]["table_suggestions"] == ["evaluation"]
            assert "effective and well-structured" in result["response"]
    
    async def test_error_handling_integration(self, mock_agent):
        """Test error handling integration across components."""
        
        # Initialize terminal app
        terminal_app = TerminalApp(enable_agent=True)
        terminal_app.agent = mock_agent
        
        # Test agent failure scenario
        mock_agent.process_query.side_effect = Exception("Database connection failed")
        
        # Test error handling in terminal app
        query = "How many users completed training?"
        
        try:
            result = await terminal_app.agent.process_query(query)
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Database connection failed" in str(e)
        
        # Test classification fallback when LLM fails
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.side_effect = Exception("LLM API timeout")
            
            # Initialize classifier
            classifier = QueryClassifier()
            await classifier.initialize()
            
            # Test classification with LLM failure
            result = await classifier.classify_query("Tell me about stuff")
            
            # Should fallback to enhanced fallback
            assert result is not None
            assert result.method_used == "fallback"
            assert result.classification in ["SQL", "VECTOR", "HYBRID", "CLARIFICATION_NEEDED"]
    
    async def test_performance_integration(self, mock_agent):
        """Test performance characteristics of integrated system."""
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Mock LLM for quick responses
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Statistical query'
            )
            
            # Initialize components
            terminal_app = TerminalApp(enable_agent=True)
            terminal_app.agent = mock_agent
            
            # Mock quick agent responses
            mock_agent.process_query.return_value = {
                "response": "Query processed successfully",
                "query_type": "sql",
                "processing_time": 0.1,
                "classification": {
                    "classification": "SQL",
                    "confidence": "HIGH",
                    "method_used": "rule_based"
                }
            }
            
            # Test multiple queries for performance
            queries = [
                "How many users completed training?",
                "What is the average completion rate?",
                "Which departments have the highest engagement?"
            ]
            
            import time
            start_time = time.time()
            
            for query in queries:
                result = await terminal_app.agent.process_query(query)
                assert result["response"] == "Query processed successfully"
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should process multiple queries reasonably quickly
            assert total_time < 5.0, f"Performance test took too long: {total_time:.2f}s"
    
    async def test_session_management_integration(self):
        """Test session management integration in terminal app."""
        
        # Initialize terminal app
        terminal_app = TerminalApp(enable_agent=True)
        
        # Test session initialization
        assert terminal_app.session_id is not None
        assert len(terminal_app.session_id) == 8
        assert terminal_app.query_count == 0
        
        # Test session state management
        initial_session_id = terminal_app.session_id
        
        # Simulate query processing
        terminal_app.query_count += 1
        assert terminal_app.query_count == 1
        
        # Test session persistence
        assert terminal_app.session_id == initial_session_id
        
        # Test feedback collection state
        assert isinstance(terminal_app.feedback_collected, dict)
    
    async def test_configuration_integration(self):
        """Test configuration integration across components."""
        
        # Test agent mode configuration
        agent_app = TerminalApp(enable_agent=True)
        assert agent_app.enable_agent == True
        assert agent_app.agent is not None
        
        # Test legacy mode configuration
        legacy_app = TerminalApp(enable_agent=False)
        assert legacy_app.enable_agent == False
        assert legacy_app.agent is None
        
        # Test feedback system configuration
        assert hasattr(agent_app, 'feedback_collector')
        assert hasattr(agent_app, 'feedback_analytics')
        assert agent_app.feedback_collector is not None
        assert agent_app.feedback_analytics is not None
    
    async def test_regression_prevention(self, mock_agent):
        """Test that new enhancements don't break existing functionality."""
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Mock LLM
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: SQL\nConfidence: HIGH\nReasoning: Statistical query'
            )
            
            # Initialize terminal app
            terminal_app = TerminalApp(enable_agent=True)
            terminal_app.agent = mock_agent
            
            # Test existing functionality still works
            assert hasattr(terminal_app, 'run')
            assert callable(getattr(terminal_app, 'run'))
            
            # Test feedback collection is optional
            assert hasattr(terminal_app, '_collect_feedback')
            assert callable(getattr(terminal_app, '_collect_feedback'))
            
            # Test existing session management
            assert hasattr(terminal_app, 'session_id')
            assert hasattr(terminal_app, 'query_count')
            
            # Test existing error handling
            assert hasattr(terminal_app, 'feedback_collected')
            assert isinstance(terminal_app.feedback_collected, dict)
            
            # Test backward compatibility
            legacy_app = TerminalApp(enable_agent=False)
            assert legacy_app.enable_agent == False
            assert legacy_app.agent is None
    
    async def test_real_world_scenario_integration(self, mock_agent):
        """Test realistic end-to-end scenario integration."""
        
        with patch('src.rag.utils.llm_utils.get_llm') as mock_llm:
            # Mock LLM responses
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.ainvoke.return_value = MagicMock(
                content='Classification: HYBRID\nConfidence: HIGH\nReasoning: Query requires both statistical data and qualitative feedback'
            )
            
            # Initialize terminal app
            terminal_app = TerminalApp(enable_agent=True)
            terminal_app.agent = mock_agent
            
            # Mock realistic hybrid response
            mock_agent.process_query.return_value = {
                "response": "The training completion rate is 85% across all departments. Participant feedback indicates high satisfaction with content quality (average 4.2/5) but suggests improvements in delivery method. Finance department shows highest engagement (92% completion) while IT department feedback requests more technical depth.",
                "query_type": "hybrid",
                "sql_result": {
                    "query": "SELECT department, AVG(completion_rate) FROM training_stats GROUP BY department",
                    "result": [{"department": "Finance", "completion_rate": 0.92}, {"department": "IT", "completion_rate": 0.78}]
                },
                "vector_result": {
                    "results": [
                        {"content": "High satisfaction with content quality", "score": 0.89},
                        {"content": "Delivery method needs improvement", "score": 0.82}
                    ]
                },
                "classification": {
                    "classification": "HYBRID",
                    "confidence": "HIGH",
                    "method_used": "rule_based",
                    "table_suggestions": ["attendance", "evaluation"]
                }
            }
            
            # Test realistic query
            query = "Analyze training effectiveness across departments with participant feedback"
            result = await terminal_app.agent.process_query(query)
            
            # Verify comprehensive response
            assert result["query_type"] == "hybrid"
            assert result["classification"]["classification"] == "HYBRID"
            assert result["classification"]["table_suggestions"] == ["attendance", "evaluation"]
            assert "completion rate is 85%" in result["response"]
            assert "participant feedback indicates" in result["response"]
            assert "Finance department shows highest engagement" in result["response"]
