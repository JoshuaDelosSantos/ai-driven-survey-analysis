# RAG Module Testing Documentation

## Overview

This directory contains comprehensive test suites for the RAG (Retrieval-Augmented Generation) module, focusing on security validation, data governance compliance, and functionality testing. All tests are designed with Australian privacy principles and data sovereignty requirements in mind.

## Testing Philosophy

### Security-First Testing
- **Credential Protection**: All tests validate that sensitive data is properly masked
- **Error Handling**: Tests ensure production-safe error messages without information leakage
- **Data Governance**: Validates read-only access patterns and compliance controls
- **Australian Privacy Compliance**: Tests align with Australian Privacy Principles (APP)

### Test Coverage Areas
1. **Configuration Management** (`test_config.py`) - **Complete**
2. **Core Functionality** (`test_phase1_refactoring.py`) - **Complete**
3. **Manual Testing** (`manual_test_phase1.py`) - **Complete**
4. **Interface Testing** - **Integrated in Phase 1 tests**
5. **Integration Testing** - **Complete**
6. **Security Validation** - **Throughout all test modules**

## Current Test Implementation

### Status: **Phase 1 Complete** - 43/43 Tests Passing

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
- **Schema Management** (4 tests): Initialisation, table info handling, fallback schema generation
- **SQL Tool Testing** (5 tests): Async SQL tool operations, safety validation, result handling
- **Database Management** (3 tests): Connection handling, read-only enforcement, resource cleanup
- **LLM Management** (4 tests): Multi-provider LLM support (OpenAI/Anthropic/Gemini), response handling
- **Logging Utilities** (3 tests): PII masking, structured logging, JSON formatting
- **Integration Testing** (2 tests): End-to-end workflow validation, async function signatures
- **Mock Configuration** (2 tests): Mock LLM responses, debug mode configuration

#### Manual Testing Suite (`manual_test_phase1.py`) - 9 Manual Tests
**Purpose**: Interactive testing with real or mock components for validation

**Test Categories**:
- Configuration loading and validation
- Schema manager database connectivity
- SQL tool query processing
- Database manager operations
- LLM manager multi-provider testing
- Logging utilities validation
- Terminal application interface
- Full integration workflow testing
- Performance and reliability testing

**Coverage Areas**:
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
pip install pytest pytest-mock

# Set up test environment variables
cp .env.example .env.test
# Edit .env.test with test database credentials
```

### Test Execution

**Run All Tests** (Current: 34/34 Automated + 9/9 Manual = 43/43 Total):
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

# Security-focused tests
python -m pytest tests/ -k "security" -v

# LLM integration tests
python -m pytest tests/ -k "llm" -v

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

### APP Validation Tests (Implemented)
1. **APP 1 (Open and Transparent Management)**: Comprehensive audit logging and transparency features tested
2. **APP 3 (Collection of Solicited Personal Information)**: Minimal data collection validation implemented
3. **APP 5 (Notification of Collection)**: User notification mechanisms tested in terminal interface
4. **APP 8 (Cross-border Disclosure)**: Data sovereignty controls validated for LLM API calls
5. **APP 11 (Security)**: Comprehensive security testing across all modules
6. **APP 12 (Access and Correction)**: Data access logging and correction mechanisms tested

### Data Sovereignty Testing (Complete)
- **LLM API Compliance**: Schema-only transmission validated for offshore LLM providers
- **Audit Trail Validation**: Complete audit trails implemented and tested for cross-border data flows
- **Data Residency**: Data residency requirements and controls tested and validated
- **Multi-Provider Governance**: Consistent privacy controls across OpenAI, Anthropic, and Gemini APIs

## Security Testing Standards

### Authentication & Authorisation (Implemented)
- Database credential validation with secure configuration management
- Read-only access constraint testing with comprehensive validation
- Connection security validation with TLS enforcement
- Multi-provider API authentication testing (OpenAI/Anthropic/Gemini)

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
  - dependency_vulnerability_check: Available
  - code_quality_analysis: pytest with coverage
  - pii_masking_validation: Comprehensive testing
```

### Performance Testing (Implemented)
- Query performance benchmarks in manual testing suite
- Memory usage validation through async resource management
- Connection pool efficiency testing with proper cleanup
- API response time validation across multiple LLM providers
- Terminal interface responsiveness testing

## Test Documentation Standards

### Test Case Documentation (Current Implementation)
Each test includes:
- **Purpose**: Clear description of what is being tested
- **Security Focus**: Specific security aspects being validated
- **APP Compliance**: Relevant Australian Privacy Principles
- **Expected Outcomes**: Clear success criteria with assertions
- **Error Scenarios**: Expected failure modes and proper handling
- **Mock Configuration**: Comprehensive mocking for external dependencies

### Test Reporting (Operational)
- **Coverage Reports**: Achieving high code coverage for security-critical components
- **Security Test Results**: Detailed security validation outcomes documented
- **Compliance Reports**: APP compliance validation results tracked
- **Performance Metrics**: Response time and resource usage monitoring implemented
- **Manual Test Results**: Structured documentation of manual testing outcomes

## Contributing to Tests

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
