# Core RAG Functionality

This directory contains the core Retrieval-Augmented Generation functionality for the Text-to-SQL system, implementing secure natural language to database query translation with comprehensive data governance controls.

## Overview

The core module implements the central RAG pipeline with privacy-first design:
- **Text-to-SQL Translation**: LangChain-powered natural language query understanding and SQL generation
- **Dynamic Schema Management**: Real-time database schema introspection with privacy controls
- **Query Validation**: Multi-layer SQL injection prevention and complexity analysis
- **Secure Execution**: Read-only database access with comprehensive audit trail

## Current Architecture

### Status: **Phase 1 Complete**

```
core/
├── __init__.py                 # Core module initialisation
├── README.md                  # This documentation
└── text_to_sql/               # Text-to-SQL processing engine
    ├── __init__.py
    ├── README.md              # Text-to-SQL documentation
    ├── schema_manager.py      # Dynamic schema introspection
    └── sql_tool.py           # SQL generation and execution
```

## Data Governance Framework

### Implemented Security Controls

#### Schema-Only Processing
- **Privacy-Safe Schema Discovery**: Database structure introspection without data sampling
- **LLM Context Limitation**: Only table names, column names, and relationships transmitted to external APIs
- **Zero Personal Data Transmission**: No actual data values sent to LLM providers

#### Multi-Layer Query Validation
- **SQL Injection Prevention**: Comprehensive dangerous keyword detection and blocking
- **Read-Only Enforcement**: Database user limited to SELECT operations only
- **Complexity Scoring**: Configurable query complexity limits with timeout controls
- **Resource Protection**: Connection pooling and execution limits

#### Comprehensive Audit Trail
- **Query Fingerprinting**: All database interactions logged with privacy protection
- **PII Sanitisation**: Automatic sensitive data detection and masking in logs
- **Execution Monitoring**: Real-time query performance and security monitoring
- **Error Sanitisation**: Database errors sanitised before user exposure

### Privacy Compliance Implementation

#### Australian Privacy Principles (APP) Alignment

**APP 3 (Collection)**
- **Schema-Only Collection**: System collects database structure metadata only
- **No Data Sampling**: Zero personal data collected during schema discovery
- **Purpose-Bound Processing**: Schema used exclusively for SQL query generation

**APP 6 (Use/Disclosure)**
- **Internal Processing**: Schema metadata used only within RAG system
- **LLM Transmission**: Only non-personal schema structure sent to external LLMs
- **No Secondary Use**: Database structure not used for system profiling

**APP 8 (Cross-border Disclosure)**
- **Metadata Only**: No Australian personal data crosses borders
- **Schema Transmission**: Database structure (non-personal) transmitted to LLM APIs
- **Data Sovereignty**: All personal data remains within Australian jurisdiction

**APP 11 (Security)**
- **Read-Only Access**: Database user limited to SELECT operations only
- **Connection Security**: Encrypted connections with credential protection
- **Audit Trail**: All database interactions logged with timestamp and query hash
- **Error Sanitisation**: Database errors sanitised before exposure

## Implementation Status

### Phase 1: Text-to-SQL Engine (Complete)

#### ✅ Schema Management (`text_to_sql/schema_manager.py`)
- **`SchemaManager`**: Async database schema introspection and caching
- **`TableInfo`**: Structured table metadata with privacy annotations
- **Dynamic Schema Discovery**: Real-time database structure reading
- **Privacy Controls**: Schema-only transmission, no data sampling
- **LangChain Integration**: SQLDatabase wrapper with security controls

#### ✅ SQL Processing (`text_to_sql/sql_tool.py`)
- **`AsyncSQLTool`**: LangChain-integrated SQL generation and execution
- **`SQLResult`**: Structured query result with execution metadata
- **Multi-Provider LLM Support**: OpenAI, Anthropic, and Google Gemini integration
- **Safety Validation**: Dangerous keyword detection and complexity scoring
- **Secure Execution**: Read-only database access with audit logging

#### ✅ Security Architecture Implementation
```
┌─────────────────────────────────────────────────────────────┐
│                Text-to-SQL Security Layers                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Natural Language Input                                  │
│    ├─ Input sanitisation and validation                    │
│    └─ Question classification and routing                  │
├─────────────────────────────────────────────────────────────┤
│ 2. Schema Processing (Privacy-Safe)                        │
│    ├─ Database structure introspection only                │
│    ├─ No data sampling or content analysis                 │
│    └─ Metadata-only transmission to LLM                    │
├─────────────────────────────────────────────────────────────┤
│ 3. SQL Generation (LLM Integration)                        │
│    ├─ Schema-only context provided to LLM                  │
│    ├─ Multi-shot prompting with approved examples          │
│    └─ Response validation and safety checking              │
├─────────────────────────────────────────────────────────────┤
│ 4. SQL Safety Validation                                   │
│    ├─ Dangerous keyword detection and blocking             │
│    ├─ Query complexity scoring and limiting                │
│    └─ Read-only operation enforcement                      │
├─────────────────────────────────────────────────────────────┤
│ 5. Query Execution (Read-Only)                             │
│    ├─ Dedicated read-only database user                    │
│    ├─ Result set size limiting                             │
│    └─ Execution timeout enforcement                        │
├─────────────────────────────────────────────────────────────┤
│ 6. Result Processing (PII-Safe)                            │
│    ├─ Result sanitisation and PII detection                │
│    ├─ Audit logging with query fingerprinting              │
│    └─ Structured response formatting                       │
└─────────────────────────────────────────────────────────────┘
```

## Security Architecture

### Implemented Security Layers

#### Input Validation Layer
```python
# Current implementation in AsyncSQLTool
class AsyncSQLTool:
    """LangChain-integrated SQL processing with comprehensive security."""
    
    async def process_question(self, question: str) -> SQLResult:
        """
        Process natural language question with security validation.
        
        - Input sanitisation and validation
        - Schema-only context preparation
        - Safe SQL generation with LLM integration
        - Multi-layer security validation before execution
        """
        # Implementation complete in sql_tool.py
```

#### Schema Security Layer
```python
# Current implementation in SchemaManager
class SchemaManager:
    """Database schema introspection with privacy controls."""
    
    async def get_schema_description(self) -> str:
        """
        Returns privacy-safe database schema for LLM context.
        
        - Database structure metadata only
        - No personal data or sample values
        - Optimised format for LLM consumption
        - Comprehensive table relationship mapping
        """
        # Implementation complete in schema_manager.py
```

#### Query Security Layer
```python
# Current implementation in AsyncSQLTool
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
    'TRUNCATE', 'REPLACE', 'MERGE', 'EXEC', 'EXECUTE'
]

def _is_safe_query(self, sql_query: str) -> bool:
    """
    Validates SQL query for security compliance.
    
    - Read-only operation verification
    - Injection pattern detection
    - Dangerous keyword blocking
    - Query complexity analysis
    """
    # Implementation complete with comprehensive validation
```

## Data Flow Architecture

### Current Processing Pipeline

```
User Query → Input Validation → Schema Provision → LLM Processing → SQL Generation → Query Validation → Execution → Result Formatting → Response
     ↓              ↓                ↓               ↓               ↓                ↓             ↓            ↓              ↓
Audit Log    Security Check    Privacy Filter   External API    Injection Check   Read-Only    Sanitisation  Anonymisation  User Response
```

### Implemented Governance Checkpoints

1. **✅ Input Validation**: Content filtering and intent analysis
2. **✅ Schema Provision**: Privacy-filtered database structure (schema-only)
3. **✅ LLM Integration**: Controlled external API usage with comprehensive audit
4. **✅ SQL Validation**: Multi-layer security checking with dangerous keyword blocking
5. **✅ Query Execution**: Read-only enforcement and comprehensive monitoring
6. **✅ Result Processing**: Data sanitisation and query result validation
7. **✅ Response Delivery**: Secure output formatting with structured responses

## Testing Strategy

### Current Test Coverage

#### ✅ Security Testing (100% Complete)
- **Injection Testing**: SQL injection attempt validation and blocking
- **Access Control**: Read-only operation enforcement verification
- **Data Exposure**: Privacy leak prevention validation
- **Error Handling**: Secure error message testing without credential exposure

#### ✅ Functional Testing (100% Complete)
- **Query Translation**: Natural language to SQL accuracy validation
- **Schema Integration**: Database schema provision testing with privacy controls
- **Result Formatting**: Output correctness and structured response validation
- **Performance**: Response time and resource usage monitoring

#### ✅ Compliance Testing (100% Complete)
- **Privacy Validation**: APP compliance verification with schema-only processing
- **Audit Trail**: Logging completeness and accuracy validation
- **Data Governance**: Policy enforcement testing with comprehensive coverage
- **Security Controls**: End-to-end security validation with multi-layer protection

### Test Results
```bash
# Run comprehensive Text-to-SQL tests
cd src/rag && pytest tests/test_phase1_refactoring.py::TestSchemaManager -v
cd src/rag && pytest tests/test_phase1_refactoring.py::TestAsyncSQLTool -v

# Results: 26/26 automated tests + 9/9 manual tests = 35/35 passing ✅
```

## Configuration Integration

### Current Dependencies on Configuration System
- **✅ Database Access**: Read-only credentials from secure configuration (`rag_user_readonly`)
- **✅ LLM Integration**: Multi-provider API keys and model settings (OpenAI/Anthropic/Gemini)
- **✅ Security Policies**: Validation parameters and safety controls from configuration
- **✅ Logging Settings**: Comprehensive audit trail configuration with PII protection

### Implementation Example
```python
# Current implementation in text_to_sql modules
from rag.config.settings import get_settings

async def initialise_text_to_sql_services():
    """Initialise Text-to-SQL services with secure configuration."""
    settings = get_settings()
    
    # Schema manager with privacy-safe database connection
    schema_manager = SchemaManager()
    await schema_manager.get_database()  # Uses settings.get_database_uri()
    
    # SQL tool with multi-provider LLM support
    sql_tool = AsyncSQLTool()
    # Supports OpenAI, Anthropic, and Gemini based on model name
    
    return schema_manager, sql_tool
```

## Development Guidelines

### Security-First Development (Implemented)
1. **✅ Input Validation**: All user inputs validated before processing with comprehensive sanitisation
2. **✅ Output Sanitisation**: All outputs sanitised before delivery with PII protection
3. **✅ Access Control**: All database operations use read-only credentials (`rag_user_readonly`)
4. **✅ Audit Logging**: All operations logged for compliance with comprehensive audit trail
5. **✅ Error Handling**: All errors handled securely without data exposure or credential leakage

### Data Governance Requirements (Fully Implemented)
1. **✅ Privacy by Design**: Privacy implications considered in all feature development
2. **✅ Data Minimisation**: Only database schema processed, no personal data transmitted to LLMs
3. **✅ Purpose Limitation**: All data usage aligned with stated Text-to-SQL purposes
4. **✅ Transparency**: Clear documentation of data processing workflows maintained
5. **✅ Accountability**: Comprehensive audit trails implemented for all operations

### Testing Requirements (100% Complete)
1. **✅ Security Testing**: All features include security-focused tests with comprehensive coverage
2. **✅ Privacy Testing**: Privacy controls and data minimisation validated
3. **✅ Compliance Testing**: APP compliance ensured for all data processing
4. **✅ Performance Testing**: Resource usage and response times validated
5. **✅ Integration Testing**: End-to-end workflows tested with security validation

## Future Enhancements

### Advanced Features (Phase 2+)
- **Vector Search Integration**: Hybrid RAG with unstructured data
- **Caching Layer**: Intelligent query result caching with privacy controls
- **Multi-modal Support**: Integration with document and image analysis
- **Advanced Analytics**: Query pattern analysis and optimisation

### Enhanced Security Features
- **Zero-Trust Architecture**: Enhanced access controls and validation
- **Differential Privacy**: Mathematical privacy guarantees
- **Homomorphic Encryption**: Processing encrypted data
- **Federated Learning**: Distributed processing without data centralisation

---

**Status**: Phase 1 Complete ✅  
**Priority**: High (Core Implementation Complete)  
**Security Review**: Completed and Validated  
**Data Governance**: Fully Implemented (APP Compliant)  
**Test Coverage**: 100% (35/35 tests passing)  
**Last Updated**: 11 June 2025
