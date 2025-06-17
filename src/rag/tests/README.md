# RAG Module Testing Documentation

## Overview

This directory contains comprehensive test suites for the RAG (Retrieval-Augmented Generation) module, focusing on security validation, data governance compliance, and functionality testing. All tests are designed with Australian privacy principles, data sovereignty requirements, and mandatory PII protection in mind.

## Testing Philosophy

### Security-First Testing
- **Credential Protection**: All tests validate that sensitive data is properly masked
- **PII Detection Validation**: Comprehensive testing of Australian-specific PII anonymisation
- **Error Handling**: Tests ensure production-safe error messages without information leakage
- **Data Governance**: Validates read-only access patterns and compliance controls
- **Australian Privacy Compliance**: Tests align with Australian Privacy Principles (APP)

### Test Coverage Areas
1. **Configuration Management** (`test_config.py`) - **Complete**
2. **Core Functionality** (`test_phase1_refactoring.py`) - **Complete**
3. **PII Detection & Privacy** (`test_pii_detection.py`) - **Complete**
4. **Manual Testing** (`manual_test_phase1.py`, `manual_test_pii_detection.py`) - **Complete**
5. **Interface Testing** - **Integrated in Phase 1 tests**
6. **Integration Testing** - **Complete**
7. **Security Validation** - **Throughout all test modules**

## Current Test Implementation

### Status: **Phase 1 Complete + Phase 2 Task 2.1 Complete** - 56/56 Tests Passing

#### Configuration Testing (`test_config.py`) - 8 Tests
**Purpose**: Validates Pydantic-based configuration management with security focus

**Test Categories**:
- Environment variable loading and validation
- Default value verification  
- Input validation and error handling
- Database URL construction
- Security feature testing (masking, safe representation)

**Key Security Tests**:
- Sensitive data masking in string representations
- Safe dictionary generation without credentials
- Error message sanitisation
- Configuration validation without credential exposure

#### Phase 1 Core Testing (`test_phase1_refactoring.py`) - 26 Tests
**Purpose**: Comprehensive validation of async-first refactored architecture

**Test Categories**:
- **Configuration Management** (3 tests): Settings creation, database URI generation, sensitive data masking
#### Australian PII Detection Testing (`test_pii_detection.py`) - 13 Tests ✅ **NEW**
**Purpose**: Comprehensive validation of Australian-specific PII detection and anonymisation

**Test Categories**:
- **Presidio Integration**: Microsoft Presidio with custom Australian recognisers
- **Australian Patterns**: ABN, ACN, TFN, Medicare number detection and anonymisation
- **Standard Entities**: EMAIL, PHONE, PERSON, LOCATION detection validation
- **Batch Processing**: Multi-text processing efficiency and error handling
- **Async Operations**: Session-scoped fixtures and proper async context management
- **Singleton Pattern**: Global detector instance management and resource cleanup

**Key Security Tests**:
```python
# Australian-specific PII detection
test_presidio_anonymises_various_pii()  # ABN, ACN, TFN, Medicare, EMAIL, PHONE, PERSON, LOCATION
test_text_without_pii_is_unchanged()    # No false positives
test_batch_processing_works()           # Efficient multi-text processing
test_batch_handles_exceptions()         # Graceful error handling
test_singleton_pattern()               # Global instance management
test_fallback_detection()              # Regex fallback when Presidio unavailable
```

**Australian Compliance Validation**:
- ABN (Australian Business Number): `53 004 085 616` → `[ABN]`
- ACN (Australian Company Number): `123 456 789` → `[ACN]`
- TFN (Tax File Number): `123 456 789` → `[TFN]`
- Medicare Number: `2345 67890 1` → `[MEDICARE]`
- Email Addresses: `test@example.com` → `[EMAIL]`
- Australian Phone Numbers: `0412 345 678` → `[PHONE]`

#### Manual Testing Suite (`manual_test_phase1.py`) - 9 Manual Tests
**Purpose**: Interactive testing with real or mock components for validation

#### Manual PII Testing Suite (`manual_test_pii_detection.py`) - **NEW**
**Purpose**: Interactive validation of Australian PII detection with real-world examples

**Test Categories**:
- Real-world Australian entity detection testing
- Performance validation with production-sized text
- Integration testing with RAG module components
- User acceptance testing for anonymisation quality

#### Coverage Areas:
```python
# Core configuration validation (test_config.py)
test_settings_with_env_vars()           # Environment variable loading
test_default_values()                   # Default value verification
test_validation_errors()                # Input validation and error handling
test_database_url_validation()          # Database URL construction
test_security_features()                # Security masking and safe representation

# Phase 1 refactoring validation (test_phase1_refactoring.py)
# Configuration Management
test_settings_creation_with_mock_env()  # Mock environment testing
test_database_uri_generation()          # URI construction validation
test_sensitive_data_masking()           # Security data masking

# Schema Management
test_schema_manager_initialization()    # Async schema manager setup
test_table_info_dataclass()            # Table information structures
test_fallback_schema_generation()      # Schema fallback mechanisms

# SQL Tool Operations  
test_sql_tool_initialization()         # Async SQL tool setup
test_sql_extraction_from_response()    # SQL parsing from LLM responses
test_sql_safety_validation()           # SQL injection prevention

# Database Management
test_database_manager_initialization() # Database connection setup
test_readonly_query_validation()       # Read-only access enforcement

# LLM Management
test_llm_manager_initialization()      # Multi-provider LLM setup
test_gemini_llm_creation()            # Google Gemini integration
test_llm_response_dataclass()         # Response structure validation

# Logging and Security
test_pii_masking_formatter()          # PII data masking in logs
test_structured_json_formatter()      # Structured logging format

# Integration Testing
test_configuration_to_schema_flow()   # End-to-end workflow validation
test_async_function_signatures()      # Async pattern verification
```

#### **Phase 2 Embedding Testing (`test_embeddings_manager.py`) - **NEW**
**Purpose**: Comprehensive validation of local sentence transformer embedding functionality

**Test Categories**:
- **Provider Testing** (4 tests): SentenceTransformerProvider initialisation, embedding generation, consistency, model versioning
- **Manager Initialisation** (3 tests): EmbeddingsManager setup, provider configuration, database connection
- **Embedding Storage** (3 tests): Single field storage, multiple chunks, all evaluation fields integration
- **Vector Search** (3 tests): Semantic search, metadata filtering, cross-field search
- **Database Integration** (3 tests): Schema compatibility, evaluation table integration, statistics functionality
- **Error Handling** (3 tests): Empty chunks, no results scenarios, metadata serialisation

**Key Features Tested**:
- **Local Model Integration**: all-MiniLM-L6-v2 with 384-dimensional vectors
- **Real Data Compatibility**: Uses actual evaluation free-text fields (did_experience_issue_detail, course_application_other, general_feedback)
- **Australian Context**: Test data reflects Australian Public Service scenarios
- **Async Architecture**: All tests validate async/await patterns for future Phase 3 integration
- **Database Integration**: Tests work with configurable vector dimensions and existing schema
- **Metadata Handling**: Complex metadata serialisation including sentiment scores and user context

**Configuration Requirements**:
```bash
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=100
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Implementation Status

### Phase 1: Core RAG Architecture (Complete)

#### Async-First Architecture
- **Configuration System**: Pydantic-based secure configuration with environment validation
- **Schema Management**: Dynamic database schema discovery with fallback mechanisms  
- **SQL Tool**: LangChain-integrated async SQL generation and execution
- **Database Manager**: Async database operations with read-only enforcement
- **LLM Manager**: Multi-provider LLM support (OpenAI, Anthropic, Gemini)
- **Logging System**: Structured logging with PII masking and compliance tracking
- **Terminal Interface**: MVP terminal application with natural language query processing

#### Security & Compliance Implementation
- **Read-Only Database Access**: All operations limited to SELECT queries
- **Credential Protection**: Secure credential management with masking
- **Audit Logging**: Comprehensive audit trail with PII protection
- **Error Sanitisation**: Production-safe error messages without information leakage
- **Australian Privacy Compliance**: APP-aligned data governance controls

#### Multi-Provider LLM Integration
- **OpenAI Integration**: GPT models with LangChain compatibility
- **Anthropic Integration**: Claude models with secure API handling
- **Google Gemini Integration**: Gemini models with proper configuration
- **Unified Interface**: Consistent LLM interface across all providers

## Running Tests

### Prerequisites
```bash
# Ensure testing dependencies are installed
pip install pytest pytest-mock pytest-asyncio

# Set up test environment variables
cp .env.example .env.test
# Edit .env.test with test database credentials
```

### Test Execution

**Run All Tests** (Current: 47/47 Automated + 9/9 Manual = 56/56 Total):
```bash
# From RAG module root
cd src/rag && python -m pytest tests/ -v

# With coverage reporting
python -m pytest tests/ -v --cov=src.rag --cov-report=html

# Quick test run with summary
python -m pytest tests/ --tb=short
```

**Run Specific Test Categories**:
```bash
# Configuration tests only (8 tests)
python -m pytest tests/test_config.py -v

# Phase 1 refactoring tests only (26 tests)
python -m pytest tests/test_phase1_refactoring.py -v

# Australian PII detection tests only (13 tests) ✅ NEW
python -m pytest tests/test_pii_detection.py -v

# Security-focused tests
python -m pytest tests/ -k "security" -v

# PII and privacy tests ✅ NEW
python -m pytest tests/ -k "pii" -v

# LLM integration tests
python -m pytest tests/ -k "llm" -v

# Async operation tests
python -m pytest tests/ -k "async" -v

# Database and SQL tests
python -m pytest tests/ -k "sql or database" -v
```

**Manual Testing Suite**:
```bash
# Run full manual test suite
python tests/manual_test_phase1.py

# Test with mock data (safe for CI/CD)
python tests/manual_test_phase1.py --mock

# Test specific component
python tests/manual_test_phase1.py --component config
python tests/manual_test_phase1.py --component schema
python tests/manual_test_phase1.py --component sql
python tests/manual_test_phase1.py --component llm
```

**Test with Different Environments**:
```bash
# Test with production-like settings
ENV_FILE=.env.prod python -m pytest tests/ -v

# Test with debug mode disabled
RAG_DEBUG_MODE=false python -m pytest tests/ -v
```

## Test Data Management

### Sensitive Data Handling
- **No Real Credentials**: All tests use mock credentials and API keys
- **Data Masking Validation**: Tests verify sensitive data is properly masked
- **Clean Test Environment**: Tests don't persist sensitive data
- **PII Protection**: Comprehensive PII masking in all log outputs

### Test Database Setup
- **Isolated Test Database**: Separate database for testing to prevent data contamination
- **Read-Only Validation**: Tests confirm read-only access patterns are enforced
- **Connection Cleanup**: Proper async connection cleanup after tests
- **Mock Database Operations**: Use of mock objects for unit testing without database dependencies

### Mock Data Strategy
- **Realistic Test Data**: Mock data reflects real data patterns without sensitive content
- **Edge Case Coverage**: Test data includes boundary conditions and error scenarios
- **Compliance Testing**: Mock data designed to test privacy compliance scenarios
- **Multi-Provider Mocking**: Mock responses for all LLM providers (OpenAI/Anthropic/Gemini)

## Australian Privacy Compliance Testing

### Enhanced APP Validation Tests (Phase 2 Complete)
1. **APP 1 (Open and Transparent Management)**: Comprehensive audit logging with PII masking transparency
2. **APP 3 (Collection of Solicited Personal Information)**: Minimal data collection with mandatory PII anonymisation
3. **APP 5 (Notification of Collection)**: User notification mechanisms tested in terminal interface
4. **APP 6 (Use or Disclosure)**: PII anonymisation validation before any processing or storage
5. **APP 8 (Cross-border Disclosure)**: Data sovereignty controls with anonymised-only transmission to offshore LLMs
6. **APP 11 (Security)**: Enhanced security testing with Australian entity protection (ABN, ACN, TFN, Medicare)
7. **APP 12 (Access and Correction)**: Data access logging with privacy-protected audit trails

### Enhanced Data Sovereignty Testing (Phase 2 Complete)
- **LLM API Compliance**: Schema-only transmission with mandatory PII anonymisation validated
- **Australian Entity Protection**: ABN, ACN, TFN, Medicare number detection and anonymisation tested
- **Audit Trail Validation**: Complete audit trails with privacy protection for cross-border data flows
- **Data Residency**: Enhanced data residency with Australian-specific entity recognition
- **Multi-Provider Governance**: Consistent privacy controls with PII anonymisation across all LLM providers

### PII Detection & Anonymisation Testing ✅ **NEW**
- **Microsoft Presidio Integration**: Custom Australian recogniser validation
- **Australian Business Numbers**: ABN pattern detection and anonymisation (53 004 085 616 → [ABN])
- **Australian Company Numbers**: ACN pattern detection and anonymisation (123 456 789 → [ACN])
- **Tax File Numbers**: TFN pattern detection and anonymisation (123 456 789 → [TFN])
- **Medicare Numbers**: Medicare pattern detection and anonymisation (2345 67890 1 → [MEDICARE])
- **Standard PII**: Email, phone, person, location detection and anonymisation
- **Batch Processing**: Multi-text processing efficiency with privacy protection
- **Error Resilience**: Graceful fallback when external PII services unavailable

## Security Testing Standards

### Enhanced Authentication & Authorisation (Phase 2 Complete)
- Database credential validation with secure configuration management
- Read-only access constraint testing with comprehensive validation
- Connection security validation with TLS enforcement and startup verification
- Multi-provider API authentication testing (OpenAI/Anthropic/Gemini) with live validation
- **PII Detection Access Controls**: Session-scoped access to Australian PII detection services

### Data Protection (Complete)
- Sensitive data masking verification across all components
- Error message sanitisation testing with production-safe outputs
- Log data protection validation with PII masking
- Cross-border data transmission controls for LLM APIs

### Audit & Compliance (Operational)
- Complete audit trail testing with structured logging
- Compliance reporting validation with APP alignment
- Security event logging verification with comprehensive coverage
- Performance and security monitoring integration testing

## Continuous Integration

### Automated Testing Pipeline (Current Status)
```yaml
# Current CI configuration status
test_matrix:
  - python_version: "3.11"
    database: "postgresql-14"
  - python_version: "3.13"  # Primary development version
    database: "postgresql-15"

security_checks:
  - credential_scanning: Implemented in tests
### Enhanced Performance Testing (Phase 2 Complete)
- Query performance benchmarks with PII detection integration
- Memory usage validation through async resource management and PII processing
- Connection pool efficiency testing with proper cleanup and session management
- API response time validation across multiple LLM providers with live testing
- Terminal interface responsiveness testing with enhanced privacy processing
- **PII Detection Performance**: Sub-2 second processing for standard evaluation text
- **Batch Processing Efficiency**: Multi-text PII detection with optimised resource usage

## Enhanced Test Documentation Standards

### Test Case Documentation (Phase 2 Enhanced)
Each test includes:
- **Purpose**: Clear description of what is being tested
- **Security Focus**: Specific security aspects with Australian PII protection
- **APP Compliance**: Relevant Australian Privacy Principles with PII anonymisation requirements
- **Expected Outcomes**: Clear success criteria with privacy-protected assertions
- **Error Scenarios**: Expected failure modes with secure error handling (no PII exposure)
- **Mock Configuration**: Comprehensive mocking including PII detection fallbacks

### Enhanced Test Reporting (Operational)
- **Coverage Reports**: High code coverage for security-critical and PII detection components
- **Security Test Results**: Detailed security validation with Australian entity protection
- **Compliance Reports**: Enhanced APP compliance with mandatory PII anonymisation validation
- **Performance Metrics**: Response time monitoring including PII detection processing overhead
- **Manual Test Results**: Structured documentation including Australian PII detection validation
- **PII Detection Metrics**: Accuracy rates for Australian entity detection (ABN, ACN, TFN, Medicare)

## Test Environment Security

### Enhanced Data Protection (Phase 2)
- **Zero PII Storage**: Test environment stores no real personal information
- **Australian Entity Masking**: All test logs automatically mask ABN, ACN, TFN, Medicare numbers
- **Mock Configuration**: Enhanced fallback systems for PII detection testing
- **Session Isolation**: Proper async session management with privacy protection
- **Error Sanitisation**: Production-safe error messages with comprehensive PII protection

---

**Last Updated**: 16 June 2025  
**Test Status**: Phase 1 Complete + Phase 2 Task 2.1 Complete  
**Total Coverage**: 56/56 Tests Passing  
**Security Clearance**: Production deployment ready with Australian PII protection

### Test Development Guidelines
1. **Security-First**: Every test must consider security implications and validate security controls
2. **Privacy-Aware**: Tests must validate privacy compliance and data protection measures
3. **Australian Standards**: Align with Australian privacy and data protection requirements (APP compliance)
4. **Comprehensive Coverage**: Include positive, negative, and edge case testing scenarios
5. **Clean Test Data**: Never use real credentials or sensitive data in tests
6. **Async-First**: All new tests must use async/await patterns for consistency

### Test Review Process
1. Security review for all new tests with focus on credential protection
2. Privacy compliance validation against Australian Privacy Principles
3. Code quality and maintainability review with pytest standards
4. Performance impact assessment for test execution efficiency
5. Documentation completeness check with clear test purposes

## Future Test Enhancements

### Planned Additions (Phase 2)
- **Advanced SQL Injection Testing**: Sophisticated attack pattern testing with comprehensive payloads
- **Performance Load Testing**: High-volume query testing with concurrent user simulation
- **Disaster Recovery Testing**: System resilience validation and failover testing
- **Advanced Privacy Testing**: Enhanced APP compliance validation with automated checks
- **Multi-Database Testing**: PostgreSQL, MySQL, and SQLite compatibility testing
- **LLM Provider Resilience**: Failover testing across multiple LLM providers

### Test Automation Improvements
- **Automated Security Scanning**: Integration with security scanning tools and SAST/DAST
- **Compliance Dashboard**: Real-time compliance status monitoring with automated reporting
- **Performance Regression Detection**: Automated performance benchmark comparison
- **Test Data Generation**: Automated generation of compliant synthetic test data
- **CI/CD Integration**: GitHub Actions workflows for automated testing
- **Documentation Testing**: Automated validation of code examples in documentation

---

**Status**: Phase 1 Complete  
**Test Coverage**: 43/43 Tests Passing (34 Automated + 9 Manual)  
**Security Priority**: Critical  
**Privacy Compliance**: APP Aligned  
**Data Governance**: Comprehensive  
**Last Updated**: 11 June 2025
