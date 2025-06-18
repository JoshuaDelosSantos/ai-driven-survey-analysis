# RAG Module Testing Documentation

## Overview

This directory contains comprehensive test suites for the RAG (Retrieval-Augmented Generation) module, focusing on security validation, data governance compliance, and functionality testing. All tests are designed with Australian privacy principles, data sovereignty requirements, and mandatory PII protection in mind.

## Current Test Status: **Phase 2 Complete** ✅

**Total Coverage**: 106+ Tests Passing  
- **Automated Tests**: 95+ passing  
- **Manual Tests**: 9+ passing  
- **Integration**: Complete Phase 1 + Phase 2 infrastructure  
- **Coverage**: All components, privacy compliance, and performance testing

## Test Modules

### Core Testing (`test_phase1_refactoring.py`) - 26 Tests
**Purpose**: Validates async-first architecture and core RAG functionality
- Configuration management with security controls
- Database operations with read-only enforcement  
- LLM integration with multi-provider support
- Error handling with information sanitisation
- Terminal interface functionality

### Configuration Testing (`test_config.py`) - 8 Tests  
**Purpose**: Validates secure configuration management
- Environment variable loading and validation
- Sensitive data masking in representations
- Database URI construction with credential protection
- Input validation and error handling

### Australian PII Detection (`test_pii_detection.py`) - 13 Tests
**Purpose**: Validates mandatory Australian PII anonymisation  
- ABN, ACN, TFN, Medicare number detection
- Anonymisation accuracy and coverage
- Integration with Microsoft Presidio
- Fallback mechanisms and error handling

### Enhanced Embeddings Management (`test_embeddings_manager.py`) - 25 Tests ✅ **Phase 2 Enhanced**
**Purpose**: Validates vector embedding operations with advanced metadata search
- Multi-provider support (OpenAI, Sentence Transformers)
- Database integration with pgvector  
- **Enhanced**: Advanced metadata filtering with `search_similar_with_metadata`
- **Enhanced**: Complex filter combinations (user_level, agency, sentiment)
- **Enhanced**: Sentiment-based filtering with configurable thresholds
- Batch processing and async operations
- Error handling with foreign key validation

### Content Processing (`test_content_processor.py`) - 6 Tests ✅ **Phase 2 Task 2.3 Complete**
**Purpose**: Validates unified text processing pipeline
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
