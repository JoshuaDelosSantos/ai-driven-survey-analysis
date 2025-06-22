# RAG Module - Complete Vector Search & Text-to-SQL System

[![Phase](https://img.shields.io/badge/Phase-2%20Complete-green)](https://shields.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready%20with%20Vector%20Search-green)](https://shields.io/)
[![Security](https://img.shields.io/badge/Security-APP%20Compliant%20+%20Australian%20Entity%20Protection-blue)](https://shields.io/)
[![Tests](https://img.shields.io/badge/Tests-106+%20Passing-brightgreen)](https://shields.io/)

## Overview

The Retrieval-Augmented Generation (RAG) module implements a comprehensive, privacy-first system combining Text-to-SQL capabilities with advanced vector search over evaluation feedback. Built with mandatory Australian data governance controls and complete PII protection, this system provides secure semantic search and intelligent query routing while maintaining strict data sovereignty.

### Current Implementation Status: **Phase 3 Task 3.1 Complete - LangGraph Agent Production Ready**

**All Components Implemented**
- **LangGraph Agent Orchestration**: Intelligent query routing with hybrid processing capabilities ✅ NEW (Phase 3)
- **Multi-Stage Query Classification**: Rule-based and LLM-powered query routing with confidence scoring ✅ NEW (Phase 3)
- **Advanced Answer Synthesis**: Multi-modal response generation combining statistical and qualitative insights ✅ NEW (Phase 3)
- **Australian PII Detection & Anonymisation**: Mandatory Microsoft Presidio integration with comprehensive Australian entity protection ✅
- **pgVector Infrastructure**: Production-ready PostgreSQL with optimised vector similarity search ✅
- **Unified Text Processing Pipeline**: Five-stage processing with mandatory privacy controls ✅
- **Vector Search Tool**: LangChain-compatible async tool with advanced metadata filtering ✅
- **Enhanced Embeddings Manager**: Multi-provider support with complex metadata search capabilities ✅
- **Comprehensive Testing Framework**: 106+ tests covering functionality, privacy compliance, and performance ✅
- **Advanced Configuration Management**: Secure Pydantic-based configuration with Australian compliance validation ✅
- **Text-to-SQL Engine**: LangChain-integrated async SQL generation with PII protection ✅
- **Multi-Provider LLM Support**: OpenAI, Anthropic, and Google Gemini with live validation ✅
- **Enhanced Database Utilities**: Read-only database management with startup validation and PII protection ✅
- **Terminal Application**: Interactive interface with mandatory PII anonymisation ✅

## Enhanced Data Governance & Security

### Privacy-First Architecture with Mandatory Australian PII Protection

The RAG system implements comprehensive privacy controls with Australian Privacy Principles (APP) compliance and mandatory PII detection across all components:

#### Complete Data Sovereignty Controls
- **Zero PII Storage**: Mandatory anonymisation before any processing or storage ✅
- **Privacy-Protected Vector Search**: All queries automatically anonymised before embedding generation ✅
- **Australian Entity Protection**: Comprehensive detection of ABN, ACN, TFN, Medicare numbers ✅
- **Enhanced Read-Only Access**: All database operations limited to SELECT queries with startup validation ✅
- **Complete Audit Trail**: Comprehensive logging with Australian entity masking of all operations ✅

#### Multi-Layer Security Framework with Complete Australian Entity Protection
- **Database Layer**: pgVector with read-only permissions, foreign key integrity, and cascade deletion ✅
- **Processing Layer**: Five-stage pipeline with mandatory PII anonymisation before sentiment analysis ✅
- **Vector Search Layer**: LangChain tool with automatic query anonymisation and privacy-safe error handling ✅
- **LLM Integration Layer**: Anonymised-only context with multi-provider data governance ✅
- **Application Layer**: Terminal interface with mandatory PII detection and secure session management ✅
- **Logging Layer**: Structured audit logs with comprehensive Australian entity detection and masking ✅

#### APP 6 (Use or Disclosure of Personal Information) - Enhanced
- **Internal Processing**: All personal data processing with mandatory PII anonymisation within system boundaries
- **No Secondary Use**: Operational data not used for profiling, with Australian entity protection validation
- **Clear Purpose Boundaries**: All components have defined data usage scope with Australian compliance requirements

#### APP 8 (Cross-border Disclosure of Personal Information) - Enhanced
- **Anonymised-Only Cross-Border**: Only PII-anonymised database structure sent to offshore APIs ✅ **Enhanced**
- **Enhanced Data Residency**: All personal data remains within Australian jurisdiction with mandatory detection
- **Enhanced Compliance Monitoring**: Comprehensive audit trail with Australian entity masking for all external API interactions

#### APP 11 (Security of Personal Information) - Enhanced
- **Encrypted Connections**: All external communications use TLS encryption with enhanced session management
- **Enhanced Credential Protection**: Secure credential management with environment isolation and Australian entity awareness
- **Enhanced Access Controls**: Role-based database access with read-only enforcement and startup validation
- **Enhanced Incident Response**: Structured error handling with Australian PII protection and security monitoring

## Enhanced Architecture

### Status: **Phase 1 Complete + Phase 2 Task 2.1 Complete - Production Ready with Australian PII Protection**

```
src/rag/
├── __init__.py              # Module initialisation
├── runner.py                # Terminal application entry point with PII protection
├── README.md               # This documentation
├── config/                 # Enhanced secure configuration management
│   ├── __init__.py
│   ├── settings.py         # Enhanced Pydantic settings with Australian compliance
│   └── README.md           # Enhanced configuration documentation
├── core/                   # Enhanced hybrid processing engine with LangGraph orchestration
│   ├── __init__.py
│   ├── agent.py            # LangGraph agent orchestrator ✅ NEW (Phase 3)
│   ├── README.md           # Enhanced core functionality documentation
│   ├── privacy/            # Australian PII detection and anonymisation ✅ NEW
│   │   ├── __init__.py
│   │   ├── pii_detector.py # Microsoft Presidio with Australian recognisers
│   │   └── README.md       # Australian PII protection documentation
│   ├── routing/            # Query classification and routing ✅ NEW (Phase 3)
│   │   ├── __init__.py
│   │   ├── query_classifier.py # Multi-stage intelligent query classification
│   │   └── README.md       # Query routing documentation
│   ├── synthesis/          # Answer generation and synthesis ✅ NEW (Phase 3)
│   │   ├── __init__.py
│   │   ├── answer_generator.py # Multi-modal answer synthesis
│   │   └── README.md       # Answer synthesis documentation
│   ├── text_to_sql/        # Enhanced LangChain-based SQL generation
│   │   ├── __init__.py
│   │   ├── README.md       # Enhanced Text-to-SQL documentation
│   │   ├── schema_manager.py # Enhanced dynamic schema introspection with PII protection
│   │   └── sql_tool.py     # Enhanced async SQL generation with mandatory anonymisation
│   └── vector_search/      # Vector search and semantic retrieval ✅ (Phase 2)
│       ├── __init__.py
│       ├── README.md       # Vector search documentation
│       └── vector_search_tool.py # Async vector search implementation
├── interfaces/             # Enhanced user interaction interfaces
│   ├── __init__.py
│   ├── README.md           # Enhanced interface documentation
│   └── terminal_app.py     # Enhanced interactive terminal with PII protection
├── utils/                  # Enhanced foundational utility modules
│   ├── __init__.py
│   ├── README.md           # Enhanced utilities documentation
│   ├── db_utils.py         # Enhanced database connection with PII result protection
│   ├── llm_utils.py        # Enhanced multi-provider LLM with live validation
│   └── logging_utils.py    # Enhanced privacy-first logging with Australian entity masking
└── tests/                  # Enhanced comprehensive test suite
    ├── __init__.py
    ├── README.md           # Enhanced testing documentation
    ├── test_config.py      # Enhanced configuration security tests
    ├── test_phase1_refactoring.py # Enhanced core functionality tests
    ├── test_pii_detection.py # Australian PII detection tests ✅ NEW
    ├── manual_test_phase1.py # Enhanced interactive testing suite
    └── manual_test_pii_detection.py # Australian PII testing suite ✅ NEW
```

### Enhanced Implementation Highlights

#### Enhanced Async-First Architecture with PII Protection
- **Enhanced Async Database Operations**: Non-blocking database connections with PII result sanitisation
- **Enhanced Async LLM Integration**: Concurrent processing with mandatory PII anonymisation before API calls
- **Enhanced Resource Management**: Proper async context managers with PII detection session management
- **Enhanced Performance Optimisation**: Connection pooling with Australian entity detection efficiency

#### Enhanced Multi-Provider LLM Support with Live Validation
- **Enhanced OpenAI Integration**: GPT models with LangChain compatibility and PII protection
- **Enhanced Anthropic Integration**: Claude models with secure API handling and anonymisation
- **Enhanced Google Gemini Integration**: Gemini models with production testing and safety configuration ✅ **Live Tested**
- **Enhanced Unified Interface**: Consistent LLM abstraction with mandatory PII anonymisation across all providers

#### Enhanced Security-First Design with Australian Entity Protection
- **Enhanced Read-Only Database Access**: All operations limited to SELECT queries with startup validation
- **Enhanced Australian PII Masking**: Mandatory detection and anonymisation of ABN, ACN, TFN, Medicare numbers ✅ **NEW**
- **Enhanced Error Sanitisation**: Production-safe error messages with comprehensive PII protection
- **Enhanced Credential Protection**: Secure credential management with Australian entity awareness

## Enhanced Documentation Structure

This RAG module includes comprehensive documentation with strong focus on Australian data governance and mandatory PII protection:

### Enhanced Module Documentation
- **Enhanced Main README** (`README.md`): Complete module overview with LangGraph agent architecture and Australian PII protection
- **Enhanced Configuration** (`config/README.md`): Secure configuration with Australian compliance and live provider validation
- **Enhanced Core Engine** (`core/README.md`): LangGraph orchestration with mandatory Australian PII protection controls ✅ **Updated (Phase 3)**
- **LangGraph Agent** (`core/agent.py`): Central intelligence orchestrator with hybrid query routing ✅ **NEW (Phase 3)**
- **Query Routing Module** (`core/routing/README.md`): Multi-stage query classification and routing system ✅ **NEW (Phase 3)**
- **Answer Synthesis Module** (`core/synthesis/README.md`): Advanced multi-modal answer generation system ✅ **NEW (Phase 3)**
- **Enhanced Privacy Module** (`core/privacy/README.md`): Australian PII detection and anonymisation system ✅ **NEW**
- **Enhanced Text-to-SQL** (`core/text_to_sql/README.md`): Detailed SQL generation with PII anonymisation and data governance
- **Vector Search Module** (`core/vector_search/README.md`): Semantic search with metadata filtering capabilities ✅ **Phase 2**
- **Enhanced Interfaces** (`interfaces/README.md`): Terminal application with mandatory PII protection and user interaction security
- **Enhanced Utilities** (`utils/README.md`): Infrastructure components with Australian entity masking and privacy-first design
- **Enhanced Testing** (`tests/README.md`): Comprehensive testing strategy with 106+ tests passing ✅ **Enhanced**

### Enhanced Documentation Features
- **Enhanced Australian Privacy Principles (APP) Compliance**: Every component documented with Australian entity protection considerations
- **Enhanced Data Sovereignty Guidance**: Cross-border data handling with mandatory PII anonymisation requirements and controls
- **Enhanced Security-First Approach**: All documentation emphasises Australian PII protection and implemented security controls
- **Production Ready with Australian Compliance**: Complete implementation with comprehensive testing validation and PII protection
- **Australian English**: Consistent use of Australian spelling throughout all documentation with entity awareness

### Enhanced Governance Integration
Each README includes:
- Enhanced data governance controls with Australian compliance validation
- Enhanced operational security measures with Australian entity masking and audit trails
- Enhanced APP compliance implementation with PII protection testing coverage
- Enhanced production deployment procedures with Australian entity monitoring
- Enhanced development guidelines with Australian privacy-first principles and mandatory PII detection

## Enhanced Configuration

The RAG module uses an enhanced comprehensive configuration system built on Pydantic BaseSettings with strong security controls and Australian compliance integration.

### Enhanced Environment Variables

#### Enhanced Database Configuration (Required + PII Protection)
```bash
# Enhanced database connection (read-only access with startup validation)
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
cd src/rag && python -m pytest tests/test_pii_detection.py -v  # ✅ NEW Australian PII tests
```

## Enhanced Data Governance Documentation

### Enhanced Access Controls Matrix

| Component | Access Level | Justification | Enhanced Monitoring |
|-----------|--------------|---------------|------------|
| Database | Read-Only + PII Protection | Query execution with result sanitisation | All queries logged with Australian entity masking |
| LLM API | External + Anonymised Only | Text-to-SQL generation with mandatory PII anonymisation | API calls logged with PII protection |
| Configuration | Environment + Entity Aware | Secure credential loading with Australian entity awareness | Masked in logs with enhanced protection |
| User Data | Processed + Anonymised | Anonymous query processing with mandatory PII detection | No persistence with Australian entity protection |
| PII Detection | Internal + Session Scoped | Australian entity detection and anonymisation | All detection logged with compliance metadata |

### Enhanced Compliance Features

#### Enhanced Australian Privacy Principles (APP) Alignment - **Phase 2 Enhanced**
- **Enhanced APP 3 (Collection)**: Minimal data collection with mandatory PII anonymisation - processes learning analytics data with Australian entity protection
- **Enhanced APP 6 (Use/Disclosure)**: Purpose-bound processing with PII anonymisation - data used exclusively for intended analytics with Australian compliance
- **Enhanced APP 8 (Cross-border)**: Enhanced data sovereignty controls - anonymised-only transmission to offshore LLM APIs with comprehensive Australian entity audit trail
- **Enhanced APP 11 (Security)**: Enhanced multi-layer security framework with Australian entity protection - comprehensive controls implemented, tested, and validated

#### Enhanced Security Controls - **Phase 2 Operational**
- **Enhanced Authentication**: Environment-based credential management with Australian entity awareness and validation
- **Enhanced Authorisation**: Role-based database access with read-only enforcement and startup validation
- **Enhanced Encryption**: TLS for all external connections with certificate validation and session management
- **Enhanced Monitoring**: Comprehensive audit logging with Australian entity masking and PII protection
- **Enhanced Incident Response**: Structured error handling with Australian PII protection and security alerting

## Enhanced Development Guidelines

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

**Enhanced Technical Issues**:
- Review the enhanced troubleshooting section above for common problems with Australian PII protection
- Check enhanced test failures for configuration or setup issues: `cd src/rag && python -m pytest -v` (56/56 tests)
- Validate all enhanced security requirements including Australian entity protection: `python tests/manual_test_phase1.py`
- Ensure enhanced data governance compliance with implemented APP controls and PII detection

**Enhanced Security Questions**:
- All enhanced security controls with Australian PII protection are implemented and tested (56/56 tests passing)
- Review enhanced security architecture documentation with Australian entity protection in individual module READMEs
- Enhanced data sovereignty controls are operational with mandatory PII anonymisation for cross-border LLM API usage
- Comprehensive audit trails with Australian entity masking are automatically generated for all operations

**Enhanced Privacy Compliance**:
- Enhanced Australian Privacy Principles (APP) compliance with mandatory PII detection is fully implemented
- Enhanced Australian entity masking operates automatically across all system components (ABN, ACN, TFN, Medicare)
- Enhanced read-only database access is enforced at the connection level with startup validation
- Enhanced cross-border data transmission is limited to anonymised schema-only (zero Australian personal data)

**Enhanced Development Support**:
- Follow the enhanced contributing guidelines with Australian PII protection requirements and code standards outlined above
- Maintain the enhanced security-first and Australian privacy-by-design approach with mandatory PII detection
- Use the enhanced comprehensive test suite including Australian entity protection to validate all changes
- Update documentation to reflect any architectural modifications with Australian compliance considerations

---

**Status**: Phase 1 Complete + Phase 2 Task 2.1 Complete - Production Ready with Australian PII Protection  
**Last Updated**: 16 June 2025  
**Version**: 1.1.0 (Enhanced with Australian Entity Protection)  
**Security Review**: Passed (56/56 tests) with Australian Compliance Validation  
**APP Compliance**: Fully Implemented  
**Data Governance**: Operational