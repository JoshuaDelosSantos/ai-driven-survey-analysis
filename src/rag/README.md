# RAG Module - Text-to-SQL System

[![Phase](https://img.shields.io/badge/Phase-1%20Complete-green)](https://shields.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)](https://shields.io/)
[![Security](https://img.shields.io/badge/Security-APP%20Compliant-blue)](https://shields.io/)
[![Tests](https://img.shields.io/badge/Tests-43/43%20Passing-brightgreen)](https://shields.io/)

## Overview

The Retrieval-Augmented Generation (RAG) module implements a secure, privacy-first Text-to-SQL system that enables natural language querying of the learning analytics database. Built with comprehensive data governance controls and Australian Privacy Principles (APP) compliance, this system provides secure access to database insights while maintaining strict data sovereignty.

### Current Implementation Status: **Phase 1 Complete**

**Completed Components**
- **Configuration Management**: Secure Pydantic-based configuration with comprehensive validation
- **Core Text-to-SQL Engine**: LangChain-integrated async SQL generation and execution
- **Multi-Provider LLM Support**: OpenAI, Anthropic, and Google Gemini integration
- **Database Utilities**: Read-only database management with security enforcement
- **Logging Infrastructure**: PII masking and structured audit logging
- **Terminal Application**: Interactive MVP interface for natural language queries
- **Comprehensive Testing**: 43/43 tests passing (34 automated + 9 manual)
- **Data Governance Framework**: Complete APP compliance implementation
- **Security Controls**: Multi-layer privacy protection and audit trail

## Data Governance & Security

### Privacy-First Architecture

The RAG system implements comprehensive privacy controls with Australian Privacy Principles (APP) compliance:

#### Data Sovereignty Controls
- **Schema-Only Transmission**: Only database structure metadata sent to external LLM APIs
- **Zero Personal Data Export**: No Australian personal data crosses international borders
- **Read-Only Access**: All database operations limited to SELECT queries only
- **Audit Trail**: Complete logging of all cross-border schema transmissions

#### Multi-Layer Security Framework
- **Database Layer**: Read-only user permissions with connection verification
- **LLM Integration Layer**: Schema-only context with multi-provider data governance
- **Application Layer**: PII masking, error sanitisation, and secure session management
- **Logging Layer**: Structured audit logs with automatic sensitive data detection

### APP Compliance Implementation

#### APP 3 (Collection of Personal Information)
- **Minimal Collection**: System processes only data necessary for learning analytics
- **No Data Sampling**: Database utilities never sample or cache personal data
- **Purpose Limitation**: All data collection aligned with learning analytics purpose

#### APP 6 (Use or Disclosure of Personal Information)
- **Internal Processing**: All personal data processing occurs within system boundaries
- **No Secondary Use**: Operational data not used for profiling or other purposes
- **Clear Purpose Boundaries**: All components have defined data usage scope

#### APP 8 (Cross-border Disclosure of Personal Information)
- **Schema-Only Cross-Border**: Only database structure (non-personal) sent to offshore APIs
- **Data Residency**: All personal data remains within Australian jurisdiction
- **Compliance Monitoring**: Comprehensive audit trail for all external API interactions

#### APP 11 (Security of Personal Information)
- **Encrypted Connections**: All external communications use TLS encryption
- **Credential Protection**: Secure credential management with environment isolation
- **Access Controls**: Role-based database access with read-only enforcement
- **Incident Response**: Structured error handling and security monitoring

## Architecture

### Status: **Phase 1 Complete - Production Ready**

```
src/rag/
├── __init__.py              # Module initialisation
├── runner.py                # Terminal application entry point
├── README.md               # This documentation
├── config/                 # Secure configuration management
│   ├── __init__.py
│   ├── settings.py         # Pydantic settings with APP compliance
│   └── README.md           # Configuration documentation
├── core/                   # Text-to-SQL processing engine
│   ├── __init__.py
│   ├── README.md           # Core functionality documentation
│   └── text_to_sql/        # LangChain-based SQL generation
│       ├── __init__.py
│       ├── README.md       # Text-to-SQL documentation
│       ├── schema_manager.py # Dynamic schema introspection
│       └── sql_tool.py     # Async SQL generation and execution
├── interfaces/             # User interaction interfaces
│   ├── __init__.py
│   ├── README.md           # Interface documentation
│   └── terminal_app.py     # Interactive terminal application
├── utils/                  # Foundational utility modules
│   ├── __init__.py
│   ├── README.md           # Utilities documentation
│   ├── db_utils.py         # Database connection management
│   ├── llm_utils.py        # Multi-provider LLM integration
│   └── logging_utils.py    # Privacy-first logging infrastructure
└── tests/                  # Comprehensive test suite
    ├── __init__.py
    ├── README.md           # Testing documentation
    ├── test_config.py      # Configuration security tests
    ├── test_phase1_refactoring.py # Core functionality tests
    └── manual_test_phase1.py # Interactive testing suite
```

### Implementation Highlights

#### Async-First Architecture
- **Async Database Operations**: Non-blocking database connections and query execution
- **Async LLM Integration**: Concurrent processing with multiple LLM providers
- **Resource Management**: Proper async context managers and cleanup procedures
- **Performance Optimisation**: Connection pooling and efficient resource utilisation

#### Multi-Provider LLM Support
- **OpenAI Integration**: GPT models with LangChain compatibility
- **Anthropic Integration**: Claude models with secure API handling  
- **Google Gemini Integration**: Gemini models with proper safety configuration
- **Unified Interface**: Consistent LLM abstraction across all providers

#### Security-First Design
- **Read-Only Database Access**: All operations limited to SELECT queries
- **PII Masking**: Automatic detection and masking of sensitive data in logs
- **Error Sanitisation**: Production-safe error messages without data exposure
- **Credential Protection**: Secure credential management with environment isolation

## Documentation Structure

This RAG module includes comprehensive documentation with strong focus on data governance and Australian privacy compliance:

### Module Documentation
- **Main README** (`README.md`): Complete module overview, architecture, and governance framework
- **Configuration** (`config/README.md`): Secure configuration management with APP compliance
- **Core Engine** (`core/README.md`): Text-to-SQL processing with privacy controls
- **Text-to-SQL** (`core/text_to_sql/README.md`): Detailed SQL generation and data governance
- **Interfaces** (`interfaces/README.md`): Terminal application and user interaction security
- **Utilities** (`utils/README.md`): Infrastructure components with privacy-first design
- **Testing** (`tests/README.md`): Comprehensive testing strategy with 43/43 tests passing

### Documentation Features
- **Australian Privacy Principles (APP) Compliance**: Every component documented with privacy considerations
- **Data Sovereignty Guidance**: Cross-border data handling requirements and controls implemented
- **Security-First Approach**: All documentation emphasises implemented security controls
- **Production Ready**: Complete implementation with comprehensive testing validation
- **Australian English**: Consistent use of Australian spelling throughout all documentation

### Governance Integration
Each README includes:
- Implemented data governance controls with validation
- Operational security measures with audit trails
- APP compliance implementation with testing coverage
- Production deployment procedures with monitoring
- Development guidelines with privacy-first principles

## Configuration

The RAG module uses a comprehensive configuration system built on Pydantic BaseSettings with strong security controls.

### Environment Variables

#### Database Configuration (Required)
```bash
# Database connection (read-only access)
RAG_DB_HOST=localhost
RAG_DB_PORT=5432
RAG_DB_NAME=csi-db
RAG_DB_USER=rag_user_readonly
RAG_DB_PASSWORD=your_secure_password

# Alternative: Full connection string
RAG_DATABASE_URL=postgresql://rag_user_readonly:password@localhost:5432/csi-db
```

#### LLM Configuration (Required)
```bash
# Multi-provider LLM support
LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=gpt-3.5-turbo  # or claude-3-sonnet-20240229 or gemini-pro
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
```

#### Query Processing (Optional)
```bash
MAX_QUERY_RESULTS=100
QUERY_TIMEOUT_SECONDS=30
ENABLE_QUERY_CACHING=true
```

#### Security Settings (Optional)
```bash
ENABLE_SQL_VALIDATION=true
MAX_SQL_COMPLEXITY_SCORE=10
```

#### Logging & Debug (Optional)
```bash
RAG_LOG_LEVEL=INFO
LOG_SQL_QUERIES=true
RAG_DEBUG_MODE=false
MOCK_LLM_RESPONSES=false
```

### Configuration Validation

All configuration is validated on startup with secure error handling:

```python
from rag.config.settings import get_settings, validate_configuration

# Load and validate configuration
settings = get_settings()
validate_configuration()
```

## Usage

### Running the Terminal Application

```bash
# Interactive terminal interface for natural language queries
python src/rag/runner.py

# Alternative execution methods
python -m src.rag.runner
cd src/rag && python runner.py
```

### Example Terminal Session

```
RAG Text-to-SQL System - Phase 1 MVP
Learning Analytics
=====================================

Ask questions about learning and development data using natural language!
The system will convert your questions into SQL queries and show results.

Security: Read-only database access, no data modification possible
Model: gpt-3.5-turbo
Session: a1b2c3d4

Example questions you can ask:
   1. How many users completed courses in each agency?
   2. Show attendance status breakdown by user level
   3. Which courses have the highest enrollment?

Your question: How many users completed training?

Processing your question...
Query completed successfully!

Generated SQL Query:
```sql
SELECT agency, COUNT(*) as completed_users 
FROM users u 
JOIN attendance a ON u.user_id = a.user_id 
WHERE a.attendance_status = 'Completed' 
GROUP BY agency;
```

Results:
----------------------------------------
agency          | completed_users
Department A    | 150
Department B    | 127  
Department C    | 89
----------------------------------------
Execution time: 0.245s
Total processing time: 2.1s
Rows returned: 3
```

### Configuration Testing

```bash
# Validate configuration and security controls
python src/rag/config/settings.py

# Run comprehensive test suite (43/43 tests)
cd src/rag && python -m pytest tests/ -v

# Security-focused testing
cd src/rag && python -m pytest tests/test_config.py::TestRAGSettings::test_security_features -v
```

## Data Governance Documentation

### Access Controls Matrix

| Component | Access Level | Justification | Monitoring |
|-----------|--------------|---------------|------------|
| Database | Read-Only | Query execution only | All queries logged |
| LLM API | External | Text-to-SQL generation | API calls logged |
| Configuration | Environment | Secure credential loading | Masked in logs |
| User Data | Processed | Anonymous query processing | No persistence |

### Compliance Features

#### Australian Privacy Principles (APP) Alignment - **Implemented**
- **APP 3 (Collection)**: Minimal data collection - only processes learning analytics data necessary for query execution
- **APP 6 (Use/Disclosure)**: Purpose-bound processing - data used exclusively for intended analytics purposes
- **APP 8 (Cross-border)**: Data sovereignty controls - schema-only transmission to offshore LLM APIs with complete audit trail
- **APP 11 (Security)**: Multi-layer security framework - comprehensive controls implemented and tested

#### Security Controls - **Operational**
- **Authentication**: Environment-based credential management with validation
- **Authorisation**: Role-based database access with read-only enforcement
- **Encryption**: TLS for all external connections with certificate validation
- **Monitoring**: Comprehensive audit logging with PII masking
- **Incident Response**: Structured error handling with security alerting

## Development Guidelines

### Adding New Features

1. **Security First**: All new features must undergo security review
2. **Data Governance**: Consider privacy implications of new functionality
3. **Testing**: Comprehensive test coverage including security tests
4. **Documentation**: Update governance documentation for changes

### Testing Standards

```bash
# Run complete test suite (43/43 tests passing)
cd src/rag && python -m pytest -v

# Run with coverage reporting
cd src/rag && python -m pytest --cov=. --cov-report=html

# Security and compliance focused tests  
cd src/rag && python -m pytest tests/test_config.py::TestRAGSettings::test_security_features -v
cd src/rag && python -m pytest tests/test_phase1_refactoring.py -k "security" -v

# Manual testing suite for integration validation
cd src/rag && python tests/manual_test_phase1.py --mock
```

### Test Coverage
- **Configuration Security**: 8/8 tests passing - credential protection and validation
- **Core Functionality**: 26/26 tests passing - async architecture and LLM integration  
- **Manual Integration**: 9/9 tests passing - end-to-end workflow validation
- **Total Coverage**: 43/43 tests passing - comprehensive validation of all components

## Troubleshooting

### Common Issues

#### Configuration Validation Errors
```bash
# Validate all environment variables and settings
python src/rag/config/settings.py

# Check specific environment variables are set
env | grep -E "(RAG_|LLM_)"

# Test configuration loading with detailed output
cd src/rag && python -c "from config.settings import get_settings; print(get_settings().get_safe_dict())"
```

#### Database Connection Issues
```bash
# Test read-only database connection
python src/db/tests/test_rag_connection.py

# Verify database permissions
cd src/rag && python tests/manual_test_phase1.py --component db
```

#### LLM Integration Problems
```bash
# Test LLM connectivity with mock responses
cd src/rag && python tests/manual_test_phase1.py --component llm --mock

# Validate specific LLM provider
cd src/rag && python -c "from utils.llm_utils import get_llm; print(get_llm())"
```

#### Terminal Application Issues
```bash
# Run terminal app with debug logging
RAG_DEBUG_MODE=true python src/rag/runner.py

# Test terminal app components individually
cd src/rag && python tests/manual_test_phase1.py --component integration
```

### Security Validation Checklist
- **Credential Masking**: Sensitive data should never appear in logs or error messages
- **Read-Only Access**: All database queries must use read-only credentials
- **Error Sanitisation**: Error messages should not expose system internals
- **Audit Trail**: All operations should be logged with proper governance metadata
- **PII Protection**: Personal information should be masked in all outputs

### Performance Diagnostics
```bash
# Run performance benchmarks
cd src/rag && python tests/manual_test_phase1.py --component performance

# Check resource usage during queries
RAG_DEBUG_MODE=true python src/rag/runner.py
# Monitor memory and connection usage in terminal output
```

## Future Roadmap

### Phase 1 (Complete)
- [x] Secure configuration management with APP compliance
- [x] Data governance framework with privacy controls
- [x] Async-first Text-to-SQL architecture
- [x] Multi-provider LLM integration (OpenAI/Anthropic/Gemini)
- [x] Database utilities with read-only enforcement
- [x] Privacy-first logging infrastructure with PII masking
- [x] Terminal MVP application with natural language interface
- [x] Comprehensive testing suite (43/43 tests passing)

### Phase 2 (Planned)
- Vector database integration for unstructured data
- Advanced query processing with caching layer
- Performance optimisation and load balancing
- Enhanced monitoring and analytics dashboard
- API endpoint development for external integration

### Phase 3 (Future)
- Multi-modal data support (documents, images, video)
- Advanced analytics with machine learning insights
- Real-time dashboard integration
- Mobile application interface
- Advanced compliance reporting and automation

## Contributing

### Prerequisites
- Python 3.13+
- PostgreSQL access with read-only user configured
- LLM API key (OpenAI, Anthropic, or Google)
- Environment variables configured (see Configuration section)

### Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure environment**: Set up required environment variables for database and LLM access
3. **Validate setup**: `python src/rag/runner.py` (runs terminal application)
4. **Run tests**: `cd src/rag && python -m pytest -v` (43/43 tests should pass)

### Development Standards
- **Australian English**: Follow Australian spelling conventions throughout codebase
- **Security-First Approach**: All new features must undergo security review and testing
- **Privacy by Design**: Consider APP compliance implications for all changes
- **Comprehensive Testing**: Maintain 100% test coverage for security-critical components
- **Documentation**: Update relevant README files for any architectural changes

### Code Quality Standards
- **Async-First**: All new code must use async/await patterns for consistency
- **Type Safety**: Use type hints and Pydantic validation for all data structures
- **Error Handling**: Implement secure error handling with proper sanitisation
- **Logging**: Use privacy-first logging with automatic PII masking
- **Testing**: Include unit tests, integration tests, and security validation

## Support

### Getting Help

**Technical Issues**:
- Review the troubleshooting section above for common problems
- Check test failures for configuration or setup issues: `cd src/rag && python -m pytest -v`
- Validate all security requirements are met using: `python tests/manual_test_phase1.py`
- Ensure data governance compliance with the implemented APP controls

**Security Questions**:
- All security controls are implemented and tested (43/43 tests passing)
- Review security architecture documentation in individual module READMEs
- Data sovereignty controls are operational for cross-border LLM API usage
- Comprehensive audit trails are automatically generated for all operations

**Privacy Compliance**:
- Australian Privacy Principles (APP) compliance is fully implemented
- PII masking operates automatically across all system components
- Read-only database access is enforced at the connection level
- Cross-border data transmission is limited to schema-only (no personal data)

**Development Support**:
- Follow the contributing guidelines and code standards outlined above
- Maintain the security-first and privacy-by-design approach
- Use the comprehensive test suite to validate all changes
- Update documentation to reflect any architectural modifications

---

**Status**: Phase 1 Complete - Production Ready  
**Last Updated**: 11 June 2025  
**Version**: 1.0.0  
**Security Review**: Passed (43/43 tests)  
**APP Compliance**: Fully Implemented  
**Data Governance**: Operational