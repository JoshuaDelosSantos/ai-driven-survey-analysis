# RAG Module Testing Documentation

## Overview

This directory contains comprehensive test suites for the RAG (Retrieval-Augmented Generation) module, focusing on security validation, data governance compliance, and functionality testing. All tests are designed with Australian privacy principles, data sovereignty requirements, and mandatory PII protection in mind.

## Current Test Status: **Phase 2 Task 2.3 Complete** ✅

**Total Coverage**: 81/81 Tests Passing  
- **Automated Tests**: 72 passing  
- **Manual Tests**: 9 passing  
- **Integration**: Phase 1 + Phase 2 complete  

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

### Embeddings Management (`test_embeddings_manager.py`) - 19 Tests
**Purpose**: Validates vector embedding operations
- Multi-provider support (OpenAI, Sentence Transformers)
- Database integration with pgvector  
- Metadata handling and search functionality
- Batch processing and async operations

### Content Processing (`test_content_processor.py`) - 6 Tests ✅ **NEW**
**Purpose**: Validates unified text processing pipeline (Phase 2 Task 2.3)
- Five-stage processing pipeline integration
- Real sentiment analysis with transformer models
- Text chunking with configurable strategies  
- Component integration and error resilience

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

## Configuration Requirements

### Database Configuration
```bash
RAG_DB_HOST=localhost
RAG_DB_PORT=5432
RAG_DB_NAME=csi-db
RAG_DB_USER=rag_user_readonly
RAG_DB_PASSWORD=your_secure_password
```

### LLM Configuration
```bash
LLM_API_KEY=your_api_key
LLM_MODEL_NAME=gemini-2.0-flash
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
```

### Embedding Configuration
```bash
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### Sentiment Analysis Configuration
```bash
SENTIMENT_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment
FREE_TEXT_COLUMNS=did_experience_issue_detail,course_application_other,general_feedback
SCORE_COLUMNS=neg,neu,pos
```

---

**Last Updated**: 17 June 2025  
**Phase Status**: Phase 2 Task 2.3 Complete ✅  
**Security Clearance**: Production deployment ready with Australian PII protection  
**Compliance**: Australian Privacy Principles (APP) aligned
