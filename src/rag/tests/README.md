# RAG Module Testing Documentation

## Overview

This directory contains comprehensive test suites for the RAG (Retrieval-Augmented Generation) module, focusing on security validation, data governance compliance, user feedback system testing, and functionality validation. All tests are designed with Australian privacy principles, data sovereignty requirements, and mandatory PII protection in mind.

## Current Test Status: **Phase 3 Complete** ✅

**Total Coverage**: 150+ Tests Passing  
- **Automated Tests**: 135+ passing  
- **Manual Tests**: 15+ passing  
- **Integration**: Complete Phase 1-3 hybrid system with feedback analytics  
- **Coverage**: All components, feedback system, privacy compliance, and performance testing
- **New Phase 3**: Feedback collection, agent orchestration, and enhanced terminal interface testing ✅

## Enhanced Test Modules (Phase 3)

### Feedback System Testing ✅ NEW (Phase 3)
#### `test_feedback_integration.py` - 20 Tests
**Purpose**: Validates integrated user feedback collection and analytics
- 1-5 scale rating validation with boundary testing
- Anonymous comment collection with PII protection
- Real-time feedback analytics and trend analysis
- Privacy-compliant feedback storage and retrieval
- Integration with terminal interface and agent workflow

#### `test_feedback_collector.py` - 15 Tests  
**Purpose**: Validates core feedback collection functionality
- Database integration with feedback table operations
- PII anonymisation in user comments
- Validation and error handling for rating input
- Session correlation and audit trail maintenance

#### `test_feedback_analytics.py` - 12 Tests
**Purpose**: Validates feedback analytics and reporting
- Real-time statistics calculation and aggregation
- Sentiment analysis integration with feedback data
- Privacy-safe analytics with anonymised data processing
- Performance monitoring and trend analysis

### Agent Orchestration Testing ✅ NEW (Phase 3)
#### `test_rag_agent.py` - 25 Tests
**Purpose**: Validates LangGraph agent orchestration and workflow
- Intelligent query routing with confidence scoring
- Hybrid processing coordination between SQL and vector search
- Error handling and recovery mechanisms with graceful degradation
- Session management and state persistence
- Integration with feedback collection workflow

#### `test_query_classifier.py` - 18 Tests
**Purpose**: Validates enhanced query classification system
- Pattern matching with optimised APS-specific patterns
- LLM-based classification with confidence calibration
- Fallback mechanisms and circuit breaker patterns
- Privacy protection with mandatory query anonymisation

### Enhanced Terminal Interface Testing ✅ (Phase 3)
#### `test_terminal_app.py` - 24 Tests (Enhanced)
**Purpose**: Validates enhanced terminal interface with feedback integration
- Natural language query processing with hybrid routing
- Feedback collection workflow with user choice and privacy protection
- System commands (`/feedback-stats`, `/help`) with analytics integration
- Error handling and session management with graceful recovery
- PII protection across all user interactions

### Core Infrastructure Testing (Enhanced Phase 3)
#### `test_phase1_refactoring.py` - 26 Tests (Enhanced)
**Purpose**: Validates async-first architecture and core RAG functionality with feedback integration
- Configuration management with security controls and feedback settings
- Database operations with read-only enforcement and feedback table support  
- LLM integration with multi-provider support and feedback context
- Error handling with information sanitisation and feedback privacy protection
- Terminal interface functionality with feedback workflow integration

#### `test_config.py` - 12 Tests (Enhanced)
**Purpose**: Validates secure configuration management with feedback system configuration
- Environment variable loading and validation including feedback settings
- Sensitive data masking in representations with feedback privacy protection
- Database URI construction with credential protection and feedback table access
- Input validation and error handling with feedback system compliance

### Privacy and Security Testing (Enhanced Phase 3)
#### `test_pii_detection.py` - 18 Tests (Enhanced)
**Purpose**: Validates mandatory Australian PII anonymisation across all system components
- ABN, ACN, TFN, Medicare number detection in queries and feedback
- Anonymisation accuracy and coverage with feedback comment protection
- Integration with Microsoft Presidio with enhanced Australian entity recognition
- Fallback mechanisms and error handling with feedback privacy compliance

### Data Processing and Vector Search Testing (Enhanced Phase 3)
#### `test_embeddings_manager.py` - 30 Tests (Enhanced)
**Purpose**: Validates vector embedding operations with advanced metadata search and feedback integration
- Multi-provider support (OpenAI, Sentence Transformers) with feedback context
- Database integration with pgvector and feedback table correlation
- **Enhanced**: Advanced metadata filtering with feedback sentiment correlation
- **Enhanced**: Complex filter combinations (user level, agency, sentiment, feedback ratings)
- **Enhanced**: Sentiment-based filtering with configurable thresholds and feedback integration
- Batch processing and async operations with feedback processing efficiency
- Error handling with foreign key validation and feedback table constraints

#### `test_content_processor.py` - 10 Tests (Enhanced)
**Purpose**: Validates unified text processing pipeline with feedback integration
- Six-stage processing pipeline with feedback data integration
- Privacy-compliant processing with feedback PII protection
- Sentiment analysis integration with feedback correlation
- Error handling and validation with feedback system compliance

### Performance and Integration Testing ✅ NEW (Phase 3)
#### `test_phase3_performance.py` - 8 Tests
**Purpose**: Validates system performance with feedback analytics integration
- Resource efficiency monitoring with feedback processing overhead
- Memory usage optimization with feedback data management
- Concurrent processing capabilities with feedback collection
- Performance benchmarking with feedback analytics integration

#### `test_phase3_e2e.py` - 15 Tests
**Purpose**: End-to-end integration testing with complete feedback workflow
- Complete user journey from query to feedback collection
- Hybrid processing workflow with feedback integration
- Privacy compliance across entire system including feedback processing
- Error recovery and system resilience with feedback system reliability
- Five-stage processing pipeline integration
- Real sentiment analysis with transformer models
- Text chunking with configurable strategies  
- Component integration and error resilience

### Vector Search Tool (`test_vector_search_tool.py`) - 40+ Tests ✅ **Phase 2 Task 2.5 Complete**
**Purpose**: Validates complete vector search functionality with privacy protection
- VectorSearchTool initialization and configuration
- Basic and advanced search operations with automatic query anonymisation
- Rich metadata filtering capabilities (user level, agency, sentiment)
- Privacy compliance and Australian PII protection validation
- Performance monitoring and metrics collection
- LangChain tool interface compatibility for agent workflows
- Error handling and edge cases with privacy-safe messaging

### Search Result Structures (`test_search_result.py`) - 25+ Tests ✅ **Phase 2 Task 2.5 Complete**
**Purpose**: Validates search result data structures with Australian compliance
- SearchMetadata container functionality with privacy-protected fields
- VectorSearchResult properties and methods with relevance categorisation
- VectorSearchResponse analysis capabilities with performance metrics
- Relevance categorisation logic (High/Medium/Low/Weak classifications)
- JSON serialisation with automatic PII sanitisation
- Data structure integrity and utility methods

### Manual Testing & Utilities

#### Interactive Testing (`manual_test_phase1.py`)
- Comprehensive terminal-based testing suite
- Real-time validation of core functionality  
- Manual verification of security controls

#### PII Detection Testing (`manual_test_pii_detection.py`)
- Interactive Australian PII detection validation
- Manual testing of anonymisation accuracy
- Edge case verification for entity recognition

#### Embedding Test Runner (`run_embedding_tests.py`)
- Convenience script for embedding-specific tests
- Environment validation and setup verification
- Simplified test execution with readable output

#### Sentiment Analysis Testing (`test_real_sentiment.py`)
- Validates real transformer model integration
- Tests sentiment analysis accuracy with sample data
- Verifies cardiffnlp/twitter-roberta-base-sentiment model

## Security & Compliance

### Australian Privacy Principles (APP) Compliance
- **Zero PII Storage**: Test environment stores no real personal information
- **Mandatory Anonymisation**: All Australian entities (ABN, ACN, TFN, Medicare) masked
- **Read-Only Database**: All operations limited to SELECT queries
- **Audit Logging**: Comprehensive audit trail with entity masking
- **Error Sanitisation**: Production-safe error messages without information leakage

### Data Sovereignty
- **Local Processing**: Text processing and embedding generation within Australian jurisdiction
- **Anonymised Cross-Border**: Only PII-anonymised data sent to external APIs
- **Compliance Monitoring**: Automated validation of data protection measures

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Modules  
```bash
pytest tests/test_content_processor.py -v
pytest tests/test_embeddings_manager.py -v
pytest tests/test_pii_detection.py -v
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Execution

### Quick Test Execution

#### Run All Vector Search Tests (Phase 2 Task 2.5)
```bash
# From src/rag/tests directory
python run_vector_search_tests.py

# Include integration tests (requires database)
python run_vector_search_tests.py --integration

# Run only unit tests
python run_vector_search_tests.py --unit-only

# Quick development testing
python run_vector_search_tests.py --quick
```

#### Run Specific Test Categories
```bash
# Search result structures only
python run_vector_search_tests.py --category structures

# Vector search tool only
python run_vector_search_tests.py --category tool

# Updated embeddings manager only
python run_vector_search_tests.py --category manager

# Privacy compliance tests only
python run_vector_search_tests.py --category privacy

# Integration tests only
python run_vector_search_tests.py --category integration
```

#### Run Individual Test Files
```bash
# Using pytest directly
pytest test_vector_search_tool.py -v
pytest test_search_result.py -v
pytest test_embeddings_manager.py -v

# Run specific test classes
pytest test_vector_search_tool.py::TestMetadataFiltering -v
pytest test_search_result.py::TestVectorSearchResponse -v
```

### Legacy Test Execution

#### All Embedding Tests (Previous Phases)
```bash
python run_embedding_tests.py
```

#### Complete Test Suite
```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src.rag --cov-report=html -v

# Run specific markers
pytest -m "not integration" -v  # Skip integration tests
pytest -m "asyncio" -v          # Only async tests
```

## Test Development Guidelines

### Adding New Tests

1. **Follow Existing Patterns**: Use the established async fixture patterns
2. **Use Real Models**: Test with actual sentence transformers and API models
3. **Privacy Compliance**: All tests must include PII protection validation
4. **Performance Tracking**: Include timing and metrics validation
5. **Error Handling**: Test both success and failure scenarios

### Test Structure
```python
@pytest.mark.asyncio
async def test_feature_name(fixture_name):
    """Test description with specific focus."""
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = await component.method(setup_data)
    
    # Assert
    assert result.expected_property == expected_value
    assert isinstance(result, ExpectedType)
    
    # Cleanup (if needed)
    await cleanup_test_data()
```

### Privacy Testing Requirements
- All user input must be tested for PII detection
- Error messages must be validated for information leakage
- Audit logging must be verified for anonymization
- Cross-border data transmission must be monitored

### Performance Testing
- Measure and validate response times
- Test with realistic data volumes
- Monitor memory usage for large embeddings
- Validate async operation efficiency

---

**Last Updated**: 17 June 2025  
**Phase Status**: Phase 2 Task 2.3 Complete ✅  
**Security Clearance**: Production deployment ready with Australian PII protection  
**Compliance**: Australian Privacy Principles (APP) aligned
