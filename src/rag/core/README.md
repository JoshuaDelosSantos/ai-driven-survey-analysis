# Core RAG Functionality

This directory will contain the core Retrieval-Augmented Generation functionality for the Text-to-SQL system, implementing secure natural language to database query translation.

## Overview

The core module implements the central RAG pipeline with strong data governance controls:
- **Text-to-SQL Translation**: Natural language query understanding and SQL generation
- **Schema Integration**: Dynamic database schema provision to LLM context
- **Query Validation**: SQL injection prevention and complexity analysis
- **Result Processing**: Secure data retrieval and response formatting

## Planned Architecture

### Current Status: **In Development**

```
core/
├── __init__.py                 # Core module initialisation
├── README.md                  # This documentation
├── schema/                    # Database schema management
│   ├── __init__.py
│   ├── provider.py           # Schema provision to LLM
│   └── validator.py          # SQL validation and security
├── llm/                       # LLM integration
│   ├── __init__.py
│   ├── client.py             # LLM API client with governance
│   └── prompts.py            # Secure prompt templates
├── query/                     # Query processing pipeline
│   ├── __init__.py
│   ├── processor.py          # Natural language processing
│   ├── translator.py         # Text-to-SQL translation
│   └── executor.py           # Secure query execution
└── response/                  # Response formatting
    ├── __init__.py
    ├── formatter.py          # Result formatting
    └── sanitiser.py          # Output sanitisation
```

## Data Governance Framework

### Planned Security Controls

#### Query Validation Pipeline
- **SQL Injection Prevention**: Comprehensive input sanitisation
- **Complexity Scoring**: Configurable query complexity limits
- **Resource Protection**: Connection pooling and timeout controls
- **Access Validation**: Read-only operation enforcement

#### Data Processing Principles
- **Minimal Exposure**: Only necessary data included in LLM context
- **Anonymous Processing**: No user identification in query processing
- **Temporary Context**: No persistent storage of user queries or results
- **Audit Trail**: Complete logging of query processing pipeline

### Privacy Compliance Features

#### Australian Privacy Principles (APP) Alignment

**APP 3 (Collection)**
- Purpose-bound data collection for query processing only
- No collection of personal information beyond query intent
- Clear data minimisation in schema provision

**APP 5 (Notification)**
- Transparent processing notification through system documentation
- Clear explanation of LLM integration for Text-to-SQL translation

**APP 6 (Use/Disclosure)**
- Data used solely for intended query processing
- No secondary use of query patterns or results
- Clear boundaries on LLM API data transmission

**APP 8 (Cross-border Disclosure)**
- Documented LLM API usage (OpenAI/external services)
- Schema-only transmission (no actual data to external services)
- Data sovereignty considerations documented

**APP 11 (Security)**
- Encrypted connections for all external API calls
- Secure credential management through configuration system
- Comprehensive audit logging for security monitoring

## Implementation Roadmap

### Phase 1.4: Schema Provision (Planned)
- **Database Schema Reader**: Secure connection to read table structures
- **Schema Formatter**: LLM-optimised schema representation
- **Metadata Provider**: Table descriptions and relationships
- **Privacy Filter**: Sensitive field identification and handling

### Phase 1.5: Text-to-SQL Implementation (Planned)
- **LangGraph Integration**: Workflow-based query processing
- **Prompt Engineering**: Secure and effective prompt templates
- **SQL Generation**: Validated SQL query creation
- **Result Validation**: Output verification and sanitisation

### Phase 1.6: Terminal Interface (Planned)
- **User Interface**: Secure terminal-based query interface
- **Session Management**: Stateless query processing
- **Error Handling**: User-friendly error messages without data exposure
- **Logging Integration**: Comprehensive audit trail

## Security Architecture

### Planned Security Layers

#### Input Validation Layer
```python
# Planned implementation concept
class QueryValidator:
    """Validates natural language queries for security and complexity."""
    
    def validate_input(self, query: str) -> ValidationResult:
        """
        Validates user input against security policies.
        
        - Content filtering for injection attempts
        - Complexity analysis for resource protection
        - Intent validation for appropriate access
        """
        pass
```

#### Schema Security Layer
```python
# Planned implementation concept  
class SchemaProvider:
    """Provides database schema with privacy controls."""
    
    def get_filtered_schema(self, context: QueryContext) -> FilteredSchema:
        """
        Returns database schema with sensitive fields filtered.
        
        - Removes PII field descriptions
        - Filters sensitive table metadata
        - Applies access control based on query context
        """
        pass
```

#### Query Security Layer
```python
# Planned implementation concept
class SQLValidator:
    """Validates generated SQL for security compliance."""
    
    def validate_sql(self, sql: str) -> ValidationResult:
        """
        Validates SQL query for security and policy compliance.
        
        - Read-only operation verification
        - Injection pattern detection
        - Resource usage analysis
        - Complexity scoring
        """
        pass
```

## Data Flow Architecture

### Planned Processing Pipeline

```
User Query → Input Validation → Schema Provision → LLM Processing → SQL Generation → Query Validation → Execution → Result Formatting → Response
     ↓              ↓                ↓               ↓               ↓                ↓             ↓            ↓              ↓
Audit Log    Security Check    Privacy Filter   External API    Injection Check   Read-Only    Sanitisation  Anonymisation  User Response
```

### Governance Checkpoints

1. **Input Validation**: Content filtering and intent analysis
2. **Schema Provision**: Privacy-filtered database structure
3. **LLM Integration**: Controlled external API usage with audit
4. **SQL Validation**: Comprehensive security checking
5. **Query Execution**: Read-only enforcement and monitoring
6. **Result Processing**: Data sanitisation and anonymisation
7. **Response Delivery**: Secure output formatting

## Testing Strategy

### Planned Test Coverage

#### Security Testing
- **Injection Testing**: SQL injection attempt validation
- **Access Control**: Read-only operation enforcement
- **Data Exposure**: Privacy leak prevention validation
- **Error Handling**: Secure error message testing

#### Functional Testing
- **Query Translation**: Natural language to SQL accuracy
- **Schema Integration**: Database schema provision testing
- **Result Formatting**: Output correctness and formatting
- **Performance**: Response time and resource usage

#### Compliance Testing
- **Privacy Validation**: APP compliance verification
- **Audit Trail**: Logging completeness and accuracy
- **Data Governance**: Policy enforcement testing
- **Security Controls**: End-to-end security validation

## Configuration Integration

### Dependencies on Configuration System
- **Database Access**: Read-only credentials from secure configuration
- **LLM Integration**: API keys and model settings from environment
- **Security Policies**: Validation parameters from configuration
- **Logging Settings**: Audit trail configuration from settings

### Example Configuration Usage
```python
# Planned implementation concept
from rag.config.settings import get_settings

def initialise_core_services():
    """Initialise core RAG services with secure configuration."""
    settings = get_settings()
    
    # Database connection with read-only access
    db_client = DatabaseClient(
        connection_string=settings.rag_database_url,
        read_only=True,
        timeout=settings.query_timeout_seconds
    )
    
    # LLM client with governance controls
    llm_client = LLMClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens
    )
    
    return CoreRAGService(db_client, llm_client, settings)
```

## Development Guidelines

### Security-First Development
1. **Input Validation**: All user inputs must be validated before processing
2. **Output Sanitisation**: All outputs must be sanitised before delivery
3. **Access Control**: All database operations must use read-only credentials
4. **Audit Logging**: All operations must be logged for compliance
5. **Error Handling**: All errors must be handled securely without data exposure

### Data Governance Requirements
1. **Privacy by Design**: Consider privacy implications in all feature development
2. **Data Minimisation**: Only process data necessary for query resolution
3. **Purpose Limitation**: Ensure all data usage aligns with stated purposes
4. **Transparency**: Maintain clear documentation of data processing workflows
5. **Accountability**: Implement comprehensive audit trails for all operations

### Testing Requirements
1. **Security Testing**: All features must include security-focused tests
2. **Privacy Testing**: Validate privacy controls and data minimisation
3. **Compliance Testing**: Ensure APP compliance for all data processing
4. **Performance Testing**: Validate resource usage and response times
5. **Integration Testing**: Test end-to-end workflows with security validation

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

**Status**: Planning Phase  
**Priority**: High (Phase 1 Core)  
**Security Review**: Required  
**Data Governance**: Critical Path  
**Last Updated**: 9 June 2025
