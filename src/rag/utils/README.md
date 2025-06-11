# RAG Utilities Module

This directory contains essential utility modules for the RAG (Retrieval-Augmented Generation) system, implementing secure, privacy-first infrastructure components with comprehensive data governance controls.

## Overview

The utilities module provides foundational services for the RAG system:
- **Database Management**: Secure read-only database connections with privacy controls
- **LLM Integration**: Multi-provider Language Model management with data sovereignty protections
- **Logging Infrastructure**: Structured logging with PII masking and audit compliance
- **Security Foundation**: Privacy-by-design components across all system operations

## Current Architecture

### Status: **Phase 1 Complete**

```
utils/
├── __init__.py          # Utilities module initialisation
├── README.md           # This documentation
├── db_utils.py         # Database connection management
├── llm_utils.py        # LLM provider integration
└── logging_utils.py    # Secure logging infrastructure
```

## Data Governance Framework

### Implemented Privacy Controls

#### Database Security (`db_utils.py`)
- **Read-Only Enforcement**: All database connections limited to SELECT operations
- **Connection Verification**: Startup validation of read-only access permissions
- **Secure Credential Management**: Environment-based credential loading with masking
- **Audit Trail Integration**: All database operations logged with privacy protection

#### LLM Provider Governance (`llm_utils.py`)
- **Multi-Provider Support**: OpenAI, Anthropic, and Google Gemini integration
- **Data Sovereignty Controls**: Schema-only transmission to external LLMs
- **Secure Configuration**: API key management with credential masking
- **Cross-Border Compliance**: APP-aligned data handling for offshore APIs

#### Privacy-First Logging (`logging_utils.py`)
- **PII Masking**: Automatic detection and masking of sensitive data in logs
- **Structured Logging**: JSON-formatted logs with compliance metadata
- **Error Sanitisation**: Production-safe error messages without data exposure
- **Audit Trail Support**: Comprehensive logging for governance requirements

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

### LLM Provider Management (`llm_utils.py`) - Complete

#### Multi-Provider Architecture
- **`LLMManager`**: Unified interface for multiple LLM providers
- **`get_llm()`**: Provider-agnostic LLM factory with secure configuration
- **`LLMResponse`**: Structured response handling with metadata
- **Provider Support**: OpenAI GPT, Anthropic Claude, Google Gemini

#### Data Sovereignty Controls
```python
# Example privacy-safe LLM interaction
llm_manager = LLMManager()
response = await llm_manager.generate_sql(
    schema_only=True,  # No personal data transmitted
    audit_trail=True   # Complete governance logging
)
```

### Logging Infrastructure (`logging_utils.py`) - Complete

#### Privacy-First Logging
- **`PIIMaskingFormatter`**: Automatic PII detection and masking
- **`RAGLogger`**: Structured logger with compliance controls
- **`get_logger()`**: Standard logger factory with privacy defaults
- **Audit Integration**: Governance-ready logging with metadata

#### Security Features
```python
# Example privacy-safe logging
logger = get_logger(__name__)
logger.log_user_query(
    query_id="abc123",
    processing_time=1.2,
    success=True,
    # Automatic PII masking applied
    # Audit metadata included
)
```

## Security Architecture

### Privacy Protection Layers
```
┌─────────────────────────────────────────────────────────────┐
│                 Utility Security Framework                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Database Layer                                          │
│    ├─ Read-only access enforcement                         │
│    ├─ Connection security with TLS                         │
│    └─ Credential masking and protection                    │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM Integration Layer                                   │
│    ├─ Schema-only transmission controls                    │
│    ├─ Multi-provider data sovereignty                      │
│    └─ Cross-border compliance monitoring                   │
├─────────────────────────────────────────────────────────────┤
│ 3. Logging Security Layer                                  │
│    ├─ Automatic PII detection and masking                  │
│    ├─ Structured audit trail generation                    │
│    └─ Error sanitisation before exposure                   │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Secure Database Operations
```python
from rag.utils.db_utils import DatabaseManager

async def secure_query_example():
    """Example of secure database operation with governance."""
    async with DatabaseManager() as db:
        # Read-only verification automatic
        # Audit logging enabled by default
        result = await db.execute_readonly_query(
            "SELECT agency, COUNT(*) FROM users GROUP BY agency"
        )
        return result
```

### Multi-Provider LLM Integration
```python
from rag.utils.llm_utils import get_llm

async def llm_integration_example():
    """Example of privacy-safe LLM integration."""
    llm = get_llm()  # Provider determined by configuration
    
    # Schema-only transmission (no personal data)
    response = await llm.generate_response(
        prompt="Generate SQL for user count by agency",
        context=schema_only_context,  # Privacy-safe context
        audit_enabled=True  # Governance logging
    )
    return response
```

### Privacy-Compliant Logging
```python
from rag.utils.logging_utils import get_logger

def logging_example():
    """Example of privacy-first logging."""
    logger = get_logger(__name__)
    
    # Automatic PII masking in all log messages
    logger.info("Processing query for user: {user_id}")  # Masked automatically
    logger.log_user_query(
        query_id="xyz789",
        success=True,
        # All sensitive data automatically masked
    )
```

## Configuration Integration

All utilities integrate with the secure configuration system:

```python
from rag.config.settings import get_settings

settings = get_settings()

# Database configuration with secure defaults
db_config = {
    'connection_string': settings.rag_database_url,  # Masked in logs
    'readonly_validation': True,                     # Enforced by default
    'audit_enabled': settings.log_sql_queries        # Governance logging
}

# LLM configuration with privacy controls
llm_config = {
    'provider': settings.llm_model_name,      # Multi-provider support
    'api_key': settings.llm_api_key,          # Secure credential handling
    'schema_only': True,                      # Privacy-safe transmission
    'audit_trail': True                       # Governance compliance
}
```

## Testing and Validation

### Security Testing Coverage
- **Credential Protection**: All sensitive data masking validated
- **Read-Only Enforcement**: Database permission testing comprehensive
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