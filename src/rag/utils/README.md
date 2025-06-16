# RAG Utilities Module

This directory contains essential utility modules for the RAG (Retrieval-Augmented Generation) system, implementing secure, privacy-first infrastructure components with comprehensive data governance controls and Australian-compliant PII protection.

## Overview

The utilities module provides foundational services for the RAG system:
- **Database Management**: Secure read-only database connections with privacy controls
- **LLM Integration**: Multi-provider Language Model management with data sovereignty protections
- **Logging Infrastructure**: Structured logging with mandatory PII masking and audit compliance
- **Security Foundation**: Privacy-by-design components with Australian regulatory alignment

## Current Architecture

### Status: **Phase 1 Complete + Phase 2 Security Enhancements**

```
utils/
├── __init__.py          # Utilities module initialisation
├── README.md           # This documentation
├── db_utils.py         # Database connection management
├── llm_utils.py        # LLM provider integration  
└── logging_utils.py    # Secure logging with PII masking
```

## Data Governance Framework

### Enhanced Privacy Controls (Phase 2 Integration)

#### Database Security (`db_utils.py`)
- **Read-Only Enforcement**: All database connections limited to SELECT operations
- **Connection Verification**: Startup validation of read-only access permissions
- **Secure Credential Management**: Environment-based credential loading with masking
- **Audit Trail Integration**: All database operations logged with privacy protection
- **PII Detection Integration**: Seamless integration with privacy module for query result sanitisation

#### LLM Provider Governance (`llm_utils.py`)
- **Multi-Provider Support**: OpenAI, Anthropic, and Google Gemini integration with live API validation
- **Data Sovereignty Controls**: Schema-only transmission to external LLMs
- **Secure Configuration**: API key management with credential masking and fallback systems
- **Cross-Border Compliance**: APP-aligned data handling for offshore APIs
- **PII-Free Processing**: Mandatory anonymisation before any LLM interaction

#### Privacy-First Logging (`logging_utils.py`)
- **Mandatory PII Masking**: Automatic detection and anonymisation of sensitive data in logs
- **Australian Entity Protection**: Enhanced masking for ABN, ACN, TFN, Medicare numbers
- **Structured Logging**: JSON-formatted logs with compliance metadata and session tracking
- **Error Sanitisation**: Production-safe error messages without data exposure
- **Audit Trail Support**: Comprehensive logging for governance requirements with privacy compliance

### Australian Privacy Principles (APP) Compliance

#### APP 3 (Collection of Personal Information)
- **Minimal Collection**: Utilities collect only operational metadata required for functionality
- **No Data Sampling**: Database utilities never sample or cache personal data
- **Purpose Limitation**: All data collection aligned with learning analytics purpose

#### APP 6 (Use or Disclosure)
- **Internal Processing**: Utilities process data exclusively within system boundaries
- **No Secondary Use**: Operational data not used for profiling or analytics
- **Clear Purpose Boundaries**: All utility functions have defined data usage scope

#### APP 8 (Cross-border Disclosure)
- **Schema-Only Transmission**: LLM utilities transmit database structure only
- **No Personal Data**: Zero transmission of Australian personal data to offshore APIs
- **Audit Trail**: Complete logging of cross-border schema transmissions

#### APP 11 (Security of Personal Information)
- **Encrypted Connections**: All external connections use TLS encryption
- **Credential Protection**: Secure credential management with environment isolation
- **Error Handling**: Secure error propagation without sensitive data exposure
- **Resource Management**: Proper cleanup and resource protection

## Implementation Status

### Database Management (`db_utils.py`) - Complete

#### Core Components
- **`DatabaseManager`**: Async database connection manager with read-only enforcement
- **`get_connection()`**: Secure connection factory with credential validation
- **`verify_readonly_access()`**: Startup verification of database permissions
- **`execute_readonly_query()`**: Safe query execution with audit logging

#### Security Features
```python
# Example secure database operation
async with DatabaseManager() as db:
    # Automatic read-only verification
    # Comprehensive audit logging
    # Secure error handling
    result = await db.execute_readonly_query(sql_query)
```

### LLM Provider Management (`llm_utils.py`) - Complete + Phase 2 Enhanced

#### Multi-Provider Architecture
- **`LLMManager`**: Unified interface for multiple LLM providers with live API validation
- **`get_llm()`**: Provider-agnostic LLM factory with secure configuration and fallback systems
- **`LLMResponse`**: Structured response handling with metadata and privacy compliance
- **Provider Support**: OpenAI GPT, Anthropic Claude, Google Gemini (production-tested)

#### Enhanced Data Sovereignty Controls
```python
# Example privacy-safe LLM interaction with PII protection
llm_manager = LLMManager()
response = await llm_manager.generate_sql(
    schema_only=True,        # No personal data transmitted
    pii_anonymised=True,     # Mandatory PII anonymisation applied
    audit_trail=True         # Complete governance logging
)
```

### Logging Infrastructure (`logging_utils.py`) - Complete + Phase 2 Australian Compliance

#### Enhanced Privacy-First Logging
- **`PIIMaskingFormatter`**: Automatic PII detection with Australian entity recognition
- **`RAGLogger`**: Structured logger with APP compliance controls and session tracking
- **`get_logger()`**: Standard logger factory with privacy defaults and fallback configuration
- **Australian Entity Masking**: ABN, ACN, TFN, Medicare number protection in logs
- **Audit Integration**: Governance-ready logging with privacy compliance metadata

#### Enhanced Security Features
```python
# Example privacy-safe logging with Australian compliance
logger = get_logger(__name__)
logger.log_user_query(
    query_id="abc123",
    processing_time=1.2,
    success=True,
    # Automatic Australian PII masking applied
    # APP compliance metadata included
    # Session tracking for audit requirements
)
```

## Security Architecture

### Privacy Protection Layers
```
┌─────────────────────────────────────────────────────────────┐
│              Enhanced Utility Security Framework           │
├─────────────────────────────────────────────────────────────┤
│ 1. Database Layer                                          │
│    ├─ Read-only access enforcement with startup validation │
│    ├─ Connection security with TLS and credential masking  │
│    └─ PII detection integration for query result sanitisation│
├─────────────────────────────────────────────────────────────┤
│ 2. LLM Integration Layer                                   │
│    ├─ Schema-only transmission with mandatory anonymisation│
│    ├─ Multi-provider data sovereignty with live validation │
│    └─ Cross-border compliance with Australian regulatory align│
├─────────────────────────────────────────────────────────────┤
│ 3. Enhanced Logging Security Layer                         │
│    ├─ Australian PII masking with ABN/ACN/TFN/Medicare protection│
│    ├─ Structured logging with APP compliance metadata      │
│    └─ Error sanitisation with secure message propagation   │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Enhanced Secure Database Operations
```python
from rag.utils.db_utils import DatabaseManager

async def secure_query_example():
    """Example of secure database operation with enhanced governance."""
    async with DatabaseManager() as db:
        # Read-only verification automatic with startup validation
        # PII detection integration for results
        # Comprehensive audit logging enabled by default
        result = await db.execute_readonly_query(
            "SELECT agency, COUNT(*) FROM users GROUP BY agency"
        )
        return result
```

### Privacy-Safe Multi-Provider LLM Integration
```python
from rag.utils.llm_utils import get_llm

async def enhanced_llm_integration_example():
    """Example of privacy-safe LLM integration with Phase 2 enhancements."""
    llm = get_llm()  # Live provider validation (Gemini/OpenAI/Anthropic)
    
    # Schema-only transmission with mandatory PII anonymisation
    response = await llm.generate_response(
        prompt="Generate SQL for user count by agency",
        context=schema_only_context,     # Privacy-safe context
        pii_anonymised=True,             # Mandatory anonymisation applied
        audit_enabled=True               # Enhanced governance logging
    )
    return response
```

### Australian-Compliant Privacy Logging
```python
from rag.utils.logging_utils import get_logger

def enhanced_logging_example():
    """Example of Australian-compliant privacy-first logging."""
    logger = get_logger(__name__)
    
    # Enhanced Australian PII masking automatically applied
    logger.info("Processing evaluation data", extra={
        "user_level": "Level 3",
        "agency": "Treasury",
        "abn_detected": True,        # Automatic ABN masking
        "medicare_detected": False,  # Medicare number protection
        "session_id": "abc123",      # Session tracking for compliance
        "app_compliance": True       # Australian Privacy Principles alignment
    })
```

## Configuration Integration

Enhanced utilities integrate with the secure configuration system and Phase 2 privacy modules:

```python
from rag.config.settings import get_settings
from rag.core.privacy.pii_detector import get_pii_detector

settings = get_settings()

# Database configuration with enhanced security defaults
db_config = {
    'connection_string': settings.rag_database_url,  # Masked in logs
    'readonly_validation': True,                     # Enforced with startup verification
    'audit_enabled': settings.log_sql_queries,      # Enhanced governance logging
    'pii_integration': True                          # Mandatory PII detection for results
}

# LLM configuration with Australian compliance
llm_config = {
    'provider': settings.llm_model_name,             # Multi-provider support
    'api_key': settings.llm_api_key,                 # Secure credential masking
    'pii_anonymisation': True,                       # Mandatory for all processing
    'cross_border_compliance': True                  # APP-aligned offshore handling
}

# Enhanced logging configuration with Australian entity protection
logging_config = {
    'pii_masking': True,                             # Enhanced Australian PII detection
    'structured_logs': True,                         # APP compliance metadata
    'audit_trail': True,                             # Complete governance logging
    'australian_entities': ['ABN', 'ACN', 'TFN', 'MEDICARE']  # Protected entity types
}
```

## Compliance & Governance

### Australian Privacy Principles (APP) Implementation
- **APP 3**: Minimal collection with clear purpose boundaries in all utility functions
- **APP 6**: Internal processing with no secondary use of operational data
- **APP 8**: Schema-only transmission for cross-border compliance with offshore LLMs
- **APP 11**: Enhanced security with mandatory PII masking and encrypted connections

### Security Status
- ✅ **Phase 1**: Database security, LLM integration, logging infrastructure complete
- ✅ **Phase 2**: Australian PII detection integration and enhanced privacy compliance
- ✅ **Production Ready**: All utilities tested with comprehensive security validation
- ✅ **Audit Compliant**: Complete governance logging with privacy protection

### Dependencies
- **Phase 2 Privacy Module**: Integration with Australian PII detection system
- **Configuration System**: Secure Pydantic settings with credential masking
- **Database Security**: Read-only role enforcement with startup validation
- **Monitoring**: Structured logging with privacy compliance and audit trails

---

**Last Updated**: 16 June 2025  
**Phase Status**: Phase 1 Complete + Phase 2 Security Enhancements  
**Compliance**: Australian Privacy Principles (APP) aligned  
**Security Clearance**: Production deployment ready
- **PII Masking**: Automatic detection and masking verification
- **Error Sanitisation**: Secure error message validation

### Privacy Compliance Validation
- **APP Compliance**: Australian Privacy Principles alignment tested
- **Data Sovereignty**: Cross-border data transmission controls validated
- **Audit Trail**: Complete governance logging verification
- **Resource Security**: Connection and credential management testing

---

**Status**: Phase 1 Complete  
**Security Priority**: Critical  
**Privacy Compliance**: APP Aligned  
**Data Governance**: Comprehensive  
**Last Updated**: 11 June 2025