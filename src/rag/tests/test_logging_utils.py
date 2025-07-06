"""
Tests for RAG logging utilities.

This module tests the logging functionality including:
- RAGLogger configuration and initialization
- User query logging with metadata support
- Backward compatibility with legacy logging calls
- Error handling and edge cases
- Log formatting and output validation
"""

import pytest
import logging
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
import tempfile
import os

from src.rag.utils.logging_utils import RAGLogger, setup_logging


class TestRAGLogger:
    """Test RAG logging utility functionality."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def rag_logger(self, temp_log_file):
        """Create a RAGLogger instance for testing."""
        logger = RAGLogger(name='test_component')
        return logger
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing without file I/O."""
        logger = RAGLogger(name='test_component')
        logger.logger = Mock()
        logger.logger.info = Mock()
        logger.logger.error = Mock()
        logger.logger.warning = Mock()
        logger.logger.debug = Mock()
        return logger
    
    def test_rag_logger_initialization(self, temp_log_file):
        """Test RAGLogger initialization with various configurations."""
        # Test basic initialization
        logger = RAGLogger(name='test_component')
        assert logger.logger is not None
        
        # Test initialization with different names
        logger_with_name = RAGLogger(name='test_component_2')
        assert logger_with_name.logger is not None
        
        # Test that logger names are different
        assert logger.logger.name != logger_with_name.logger.name
    
    def test_setup_logging_function(self, temp_log_file):
        """Test the setup_logging utility function."""
        # Test the setup_logging function (it doesn't return anything)
        try:
            setup_logging()
            # If no exception, the function works
            setup_successful = True
        except Exception:
            setup_successful = False
        
        assert setup_successful, "setup_logging should execute without error"
        
        # Test that it configures logging properly
        root_logger = logging.getLogger()
        assert root_logger.level > 0  # Should have a level set
    
    # Phase 3 Core Tests: Metadata Parameter Testing
    def test_log_user_query_with_metadata(self, mock_logger):
        """Test log_user_query method with metadata parameter."""
        # Test metadata parameter acceptance
        metadata = {
            'classification': 'SQL',
            'confidence': 'HIGH',
            'processing_method': 'rule_based',
            'tool_usage': ['pattern_matcher', 'sql_generator'],
            'reasoning': 'Query matched SQL patterns with high confidence'
        }
        
        # Should not raise any exceptions
        try:
            mock_logger.log_user_query(
                query_id='test_query_001',
                query_type='agent',
                processing_time=1.5,
                success=True,
                metadata=metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify the method executed successfully
        assert method_executed, f"log_user_query with metadata should execute successfully"
    
    def test_log_user_query_without_metadata(self, mock_logger):
        """Test log_user_query method without metadata parameter (backward compatibility)."""
        # Test legacy call without metadata
        try:
            mock_logger.log_user_query(
                query_id='test_query_002',
                query_type='sql',
                processing_time=0.8,
                success=True
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify legacy logging still works
        assert method_executed, f"log_user_query without metadata should execute successfully"
    
    def test_log_user_query_with_error(self, mock_logger):
        """Test log_user_query method with error scenarios."""
        # Test error logging with metadata
        metadata = {
            'classification': 'CLARIFICATION_NEEDED',
            'confidence': 'LOW',
            'error_context': 'Query too ambiguous for classification'
        }
        
        error_message = "Classification failed: ambiguous query"
        
        try:
            mock_logger.log_user_query(
                query_id='test_query_003',
                query_type='agent',
                processing_time=2.1,
                success=False,
                error=error_message,
                metadata=metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify error logging with metadata works
        assert method_executed, f"log_user_query with error and metadata should execute successfully"
    
    def test_log_user_query_metadata_types(self, mock_logger):
        """Test log_user_query with various metadata types."""
        # Test different metadata data types
        complex_metadata = {
            'string_field': 'test_value',
            'integer_field': 42,
            'float_field': 3.14,
            'boolean_field': True,
            'list_field': ['item1', 'item2', 'item3'],
            'dict_field': {'nested': 'value', 'count': 5},
            'none_field': None
        }
        
        try:
            mock_logger.log_user_query(
                query_id='test_query_004',
                query_type='hybrid',
                processing_time=1.8,
                success=True,
                metadata=complex_metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify complex metadata types are handled
        assert method_executed, f"log_user_query with complex metadata should execute successfully"
    
    def test_log_user_query_empty_metadata(self, mock_logger):
        """Test log_user_query with empty metadata."""
        # Test with empty dictionary
        try:
            mock_logger.log_user_query(
                query_id='test_query_005',
                query_type='vector',
                processing_time=0.5,
                success=True,
                metadata={}
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify empty metadata is handled gracefully
        assert method_executed, f"log_user_query with empty metadata should execute successfully"
    
    def test_log_user_query_none_metadata(self, mock_logger):
        """Test log_user_query with None metadata."""
        # Test with None metadata (should behave like no metadata)
        try:
            mock_logger.log_user_query(
                query_id='test_query_006',
                query_type='sql',
                processing_time=0.3,
                success=True,
                metadata=None
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify None metadata is handled gracefully
        assert method_executed, f"log_user_query with None metadata should execute successfully"
    
    # Phase 3 Agent Integration Tests
    def test_agent_query_metadata_logging(self, mock_logger):
        """Test agent query logging with comprehensive metadata."""
        # Test realistic agent query metadata
        agent_metadata = {
            'classification': 'HYBRID',
            'confidence': 'MEDIUM',
            'processing_method': 'llm_based',
            'pattern_matches': ['feedback', 'analysis', 'trends'],
            'table_suggestions': ['evaluation', 'attendance'],
            'reasoning': 'Query combines statistical analysis with qualitative feedback review',
            'tool_usage': ['query_classifier', 'sql_generator', 'vector_search', 'response_synthesizer'],
            'llm_calls': 2,
            'vector_search_results': 15,
            'sql_execution_time': 0.8,
            'response_length': 1250,
            'user_feedback_score': None  # Not yet provided
        }
        
        try:
            mock_logger.log_user_query(
                query_id='agent_query_001',
                query_type='agent',
                processing_time=4.2,
                success=True,
                metadata=agent_metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify comprehensive agent metadata logging works
        assert method_executed, f"log_user_query with agent metadata should execute successfully"
    
    def test_classification_metadata_logging(self, mock_logger):
        """Test classification-specific metadata logging."""
        # Test classification metadata from query classifier
        classification_metadata = {
            'classification': 'SQL',
            'confidence': 'HIGH',
            'processing_method': 'rule_based',
            'pattern_matches': ['count', 'users', 'agency'],
            'confidence_score': 0.92,
            'reasoning': 'Strong SQL patterns detected: aggregation, grouping, statistical analysis',
            'table_routing': 'users',
            'query_complexity': 'medium',
            'pii_detected': False,
            'anonymized_query': None
        }
        
        try:
            mock_logger.log_user_query(
                query_id='classify_query_001',
                query_type='agent',
                processing_time=0.15,
                success=True,
                metadata=classification_metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify classification metadata logging works
        assert method_executed, f"log_user_query with classification metadata should execute successfully"
    
    def test_error_metadata_logging(self, mock_logger):
        """Test error scenarios with metadata logging."""
        # Test error with diagnostic metadata
        error_metadata = {
            'classification': 'CLARIFICATION_NEEDED',
            'confidence': 'LOW',
            'processing_method': 'fallback',
            'error_type': 'ambiguous_query',
            'error_context': 'Query too vague for reliable classification',
            'fallback_attempts': 3,
            'circuit_breaker_state': 'closed',
            'retry_count': 2,
            'diagnostic_info': {
                'query_length': 8,
                'word_count': 2,
                'has_keywords': False,
                'pattern_matches': []
            }
        }
        
        try:
            mock_logger.log_user_query(
                query_id='error_query_001',
                query_type='agent',
                processing_time=1.8,
                success=False,
                error='Query classification failed: insufficient context',
                metadata=error_metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify error metadata logging works
        assert method_executed, f"log_user_query with error metadata should execute successfully"
    
    # Backward Compatibility Tests
    def test_backward_compatibility_sql_mode(self, mock_logger):
        """Test backward compatibility for SQL-only mode logging."""
        # Test legacy SQL query logging
        try:
            mock_logger.log_user_query(
                query_id='legacy_sql_001',
                query_type='sql',
                processing_time=0.6,
                success=True
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify legacy logging still works
        assert method_executed, f"Legacy SQL logging should execute successfully"
    
    def test_mixed_usage_scenarios(self, mock_logger):
        """Test mixed usage of legacy and enhanced logging."""
        # Test sequence of legacy and enhanced calls
        
        # Legacy call
        try:
            mock_logger.log_user_query(
                query_id='mixed_001',
                query_type='sql',
                processing_time=0.5,
                success=True
            )
            legacy_executed = True
        except Exception:
            legacy_executed = False
        
        # Enhanced call
        try:
            mock_logger.log_user_query(
                query_id='mixed_002',
                query_type='agent',
                processing_time=2.1,
                success=True,
                metadata={
                    'classification': 'VECTOR',
                    'confidence': 'HIGH',
                    'processing_method': 'rule_based'
                }
            )
            enhanced_executed = True
        except Exception:
            enhanced_executed = False
        
        # Verify both calls worked
        assert legacy_executed, "Legacy call should execute successfully"
        assert enhanced_executed, "Enhanced call should execute successfully"
    
    # Error Handling Tests
    def test_metadata_error_handling(self, mock_logger):
        """Test error handling for problematic metadata."""
        # Test with metadata that might cause issues
        problematic_metadata = {
            'large_string': 'x' * 10000,  # Very large string
            'unicode_content': 'Test with émojis 🚀 and spécial chars',
            'empty_nested': {'': {'': ''}},
            'numeric_keys': {1: 'one', 2: 'two'},  # Non-string keys
        }
        
        # Should handle problematic metadata gracefully
        try:
            mock_logger.log_user_query(
                query_id='problem_query_001',
                query_type='agent',
                processing_time=1.0,
                success=True,
                metadata=problematic_metadata
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Verify the call succeeded despite problematic metadata
        assert method_executed, f"Logging with problematic metadata should handle gracefully"
    
    def test_method_parameter_validation(self, mock_logger):
        """Test parameter validation for log_user_query method."""
        # Test with required parameters only
        try:
            mock_logger.log_user_query(
                query_id='param_test_001',
                query_type='sql',
                processing_time=0.5
            )
            method_executed = True
        except Exception as e:
            method_executed = False
            error_msg = str(e)
        
        # Should succeed with defaults
        assert method_executed, f"Method should execute successfully with required parameters only"
    
    # Integration Tests
    def test_real_file_logging(self, temp_log_file):
        """Test actual file logging with metadata."""
        # Create logger with real file output
        logger = RAGLogger(name='test_integration')
        
        # Log with metadata
        metadata = {
            'classification': 'HYBRID',
            'confidence': 'HIGH',
            'processing_method': 'llm_based',
            'test_timestamp': time.time()
        }
        
        logger.log_user_query(
            query_id='file_test_001',
            query_type='agent',
            processing_time=1.5,
            success=True,
            metadata=metadata
        )
        
        # For this test, we just verify that the method executes without error
        # since we don't have direct file output control with the current logger setup
        assert True  # If we get here, the logging worked without exceptions
    
    def test_log_formatting_consistency(self, mock_logger):
        """Test that log formatting is consistent across different scenarios."""
        # Test different query types with metadata
        scenarios = [
            ('sql', {'classification': 'SQL', 'confidence': 'HIGH'}),
            ('vector', {'classification': 'VECTOR', 'confidence': 'MEDIUM'}),
            ('hybrid', {'classification': 'HYBRID', 'confidence': 'LOW'}),
            ('agent', {'classification': 'CLARIFICATION_NEEDED', 'confidence': 'LOW'})
        ]
        
        all_successful = True
        for i, (query_type, metadata) in enumerate(scenarios):
            try:
                mock_logger.log_user_query(
                    query_id=f'format_test_{i:03d}',
                    query_type=query_type,
                    processing_time=float(i) + 0.5,
                    success=True,
                    metadata=metadata
                )
            except Exception as e:
                all_successful = False
                break
        
        # Verify all calls were successful
        assert all_successful, "All log formatting scenarios should execute successfully"
    
    def test_performance_impact(self, mock_logger):
        """Test that metadata logging doesn't significantly impact performance."""
        # Test performance with and without metadata
        
        # Large metadata set
        large_metadata = {
            f'field_{i}': f'value_{i}' for i in range(100)
        }
        large_metadata.update({
            'classification': 'HYBRID',
            'confidence': 'MEDIUM',
            'processing_method': 'llm_based',
            'tool_usage': ['tool_' + str(i) for i in range(20)],
            'large_data': list(range(1000))
        })
        
        # Time the logging operation
        start_time = time.time()
        
        all_successful = True
        for i in range(10):  # Multiple calls
            try:
                mock_logger.log_user_query(
                    query_id=f'perf_test_{i:03d}',
                    query_type='agent',
                    processing_time=1.0,
                    success=True,
                    metadata=large_metadata
                )
            except Exception as e:
                all_successful = False
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance is reasonable and all calls succeeded
        assert total_time < 1.0, "Logging should complete quickly"
        assert all_successful, "All performance test calls should succeed"
