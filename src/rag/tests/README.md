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
1. **Configuration Management** (`test_config.py`)
2. **Core Functionality** (planned for Task 1.4-1.5)
3. **Interface Testing** (planned for Task 1.5)
4. **Integration Testing** (planned for Task 1.6)
5. **Security Validation** (throughout all test modules)

## Current Test Implementation

### Configuration Testing (`test_config.py`)

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

**Coverage Areas**:
```python
# Core configuration validation
test_settings_with_env_vars()
test_default_values()
test_validation_errors()

# Security-focused tests
test_security_features()
test_get_settings_error_handling()
test_mask_sensitive_value()

# Database connection testing
test_database_url_validation()
```

## Planned Test Implementation

### Core Module Testing (Task 1.4-1.5)

**Text-to-SQL Testing**:
- SQL generation validation
- Query complexity scoring
- Injection prevention testing
- Result set limitation validation

**LLM Interface Testing**:
- API call validation
- Response parsing
- Error handling for external services
- Data sovereignty compliance for cross-border API calls

**Security Testing**:
- Read-only database access validation
- SQL sanitisation effectiveness
- Audit logging verification
- Error message security

### Interface Testing (Task 1.5)

**Database Interface**:
- Connection pooling validation
- Read-only constraint testing
- Query timeout handling
- Error recovery mechanisms

**External API Interface**:
- LLM provider integration testing
- Rate limiting compliance
- Data sovereignty validation for offshore API calls
- Fallback mechanism testing

**Monitoring Interface**:
- Audit log generation
- Performance metric collection
- Security event logging
- Compliance reporting

### Integration Testing (Task 1.6)

**End-to-End Workflows**:
- Complete Text-to-SQL pipeline testing
- Terminal interface validation
- Error propagation testing
- Performance validation

**Security Integration**:
- Complete audit trail validation
- Data governance compliance testing
- Privacy principle adherence validation
- Cross-border data handling compliance

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

**Run All Tests**:
```bash
# From RAG module root
pytest tests/ -v

# With coverage reporting
pytest tests/ -v --cov=src.rag --cov-report=html
```

**Run Specific Test Categories**:
```bash
# Configuration tests only
pytest tests/test_config.py -v

# Security tests only (when implemented)
pytest tests/ -k "security" -v

# Integration tests only (when implemented)
pytest tests/ -k "integration" -v
```

**Test with Different Environments**:
```bash
# Test with production-like settings
ENV_FILE=.env.prod pytest tests/ -v

# Test with debug mode disabled
RAG_DEBUG_MODE=false pytest tests/ -v
```

## Test Data Management

### Sensitive Data Handling
- **No Real Credentials**: All tests use mock credentials and API keys
- **Data Masking Validation**: Tests verify sensitive data is properly masked
- **Clean Test Environment**: Tests don't persist sensitive data

### Test Database Setup
- **Isolated Test Database**: Separate database for testing to prevent data contamination
- **Read-Only Validation**: Tests confirm read-only access patterns
- **Connection Cleanup**: Proper connection cleanup after tests

### Mock Data Strategy
- **Realistic Test Data**: Mock data reflects real data patterns without sensitive content
- **Edge Case Coverage**: Test data includes boundary conditions and error scenarios
- **Compliance Testing**: Mock data designed to test privacy compliance scenarios

## Australian Privacy Compliance Testing

### APP Validation Tests
1. **APP 1 (Open and Transparent Management)**: Test audit logging and transparency features
2. **APP 3 (Collection of Solicited Personal Information)**: Validate minimal data collection
3. **APP 5 (Notification of Collection)**: Test user notification mechanisms
4. **APP 8 (Cross-border Disclosure)**: Validate data sovereignty for LLM API calls
5. **APP 11 (Security)**: Comprehensive security testing
6. **APP 12 (Access and Correction)**: Test data access logging and correction mechanisms

### Data Sovereignty Testing
- **LLM API Compliance**: Validate data handling for offshore LLM providers
- **Audit Trail Validation**: Ensure complete audit trails for cross-border data
- **Data Residency**: Test data residency requirements and controls

## Security Testing Standards

### Authentication & Authorisation
- Database credential validation
- Read-only access constraint testing
- Connection security validation

### Data Protection
- Sensitive data masking verification
- Error message sanitisation testing
- Log data protection validation

### Audit & Compliance
- Complete audit trail testing
- Compliance reporting validation
- Security event logging verification

## Continuous Integration

### Automated Testing Pipeline
```yaml
# Example CI configuration
test_matrix:
  - python_version: "3.11"
    database: "postgresql-14"
  - python_version: "3.12"
    database: "postgresql-15"

security_checks:
  - credential_scanning
  - dependency_vulnerability_check
  - code_quality_analysis
```

### Performance Testing
- Query performance benchmarks
- Memory usage validation
- Connection pool efficiency testing
- API response time validation

## Test Documentation Standards

### Test Case Documentation
Each test should include:
- **Purpose**: Clear description of what is being tested
- **Security Focus**: Specific security aspects being validated
- **APP Compliance**: Relevant Australian Privacy Principles
- **Expected Outcomes**: Clear success criteria
- **Error Scenarios**: Expected failure modes and handling

### Test Reporting
- **Coverage Reports**: Minimum 90% code coverage for security-critical components
- **Security Test Results**: Detailed security validation outcomes
- **Compliance Reports**: APP compliance validation results
- **Performance Metrics**: Response time and resource usage reports

## Contributing to Tests

### Test Development Guidelines
1. **Security-First**: Every test should consider security implications
2. **Privacy-Aware**: Tests should validate privacy compliance
3. **Australian Standards**: Align with Australian privacy and data protection requirements
4. **Comprehensive Coverage**: Include positive, negative, and edge case testing
5. **Clean Test Data**: Never use real credentials or sensitive data in tests

### Test Review Process
1. Security review for all new tests
2. Privacy compliance validation
3. Code quality and maintainability review
4. Performance impact assessment
5. Documentation completeness check

## Future Test Enhancements

### Planned Additions (Phase 2)
- **Advanced SQL Injection Testing**: Sophisticated attack pattern testing
- **Performance Load Testing**: High-volume query testing
- **Disaster Recovery Testing**: System resilience validation
- **Advanced Privacy Testing**: Enhanced APP compliance validation

### Test Automation Improvements
- **Automated Security Scanning**: Integration with security scanning tools
- **Compliance Dashboard**: Real-time compliance status monitoring
- **Performance Regression Detection**: Automated performance benchmark comparison
- **Test Data Generation**: Automated generation of compliant test data

---

**Note**: This testing framework is designed to support the RAG module's security-first approach and Australian privacy compliance requirements. All tests prioritise data protection and governance validation while ensuring functional reliability.
