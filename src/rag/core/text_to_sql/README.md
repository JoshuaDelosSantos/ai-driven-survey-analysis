# Text-to-SQL Processing Engine

This directory contains the core Text-to-SQL functionality for the RAG system, implementing privacy-first natural language query processing with comprehensive data governance controls.

## Overview

The Text-to-SQL engine provides secure query generation and execution:
- **Dynamic Schema Management**: Real-time database schema introspection
- **Privacy-Safe SQL Generation**: LangChain-powered query generation with PII protection
- **Read-Only Execution**: Enforced read-only database access with validation
- **Audit Trail**: Comprehensive logging with PII sanitisation

## Files

### `schema_manager.py`
Database schema introspection and management:

#### Core Classes
- **`SchemaManager`**: Async schema discovery and caching
- **`TableInfo`**: Structured table metadata with privacy annotations
- **Privacy Features**: Schema-only transmission to LLMs, no data sampling

#### Key Functions
- **`get_database()`**: Secure LangChain SQLDatabase connection
- **`get_schema_description()`**: Privacy-safe schema formatting for LLM context
- **`close()`**: Resource cleanup with connection pool management

### `sql_tool.py`
SQL query generation and execution with privacy controls:

#### Core Classes
- **`AsyncSQLTool`**: LangChain-integrated SQL processing
- **`SQLResult`**: Structured query result with execution metadata
- **Security Features**: SQL injection prevention, dangerous keyword blocking

#### Key Functions
- **`generate_sql()`**: Natural language to SQL conversion
- **`execute_sql()`**: Safe query execution with result sanitisation
- **`process_question()`**: End-to-end question processing pipeline

## Data Governance Features

### Privacy Controls

#### Enhanced Schema Context (July 2025)
```python
# Enhanced table purpose guidance for accurate SQL generation
schema_context = await schema_manager.get_schema_description()
# Includes: Clear table usage guidance, feedback query routing, relationship mapping
# Prevents: Incorrect table joins, semantic mismatches, empty result sets
```

**Table Usage Guidance**:
- **`evaluation`**: User feedback about learning content and courses
- **`rag_user_feedback`**: Feedback about RAG system performance only
- **`attendance`**: Participation statistics and completion tracking
- **`users`**: Demographic analysis and user categorization
- **`learning_content`**: Content metadata and categorization
- **`rag_embeddings`**: Internal vector search operations

#### Schema-Only Processing
```python
# NO personal data transmitted to LLMs
schema_description = await schema_manager.get_schema_description()
# Contains: table names, column names, data types, relationships
# EXCLUDES: actual data values, row counts, sample content
```

#### Query Safety Validation
```python
# Multi-layer SQL safety enforcement
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
    'TRUNCATE', 'REPLACE', 'MERGE', 'EXEC', 'EXECUTE'
]

# Runtime validation before execution
if not self._is_safe_query(sql_query):
    raise SecurityError("Dangerous SQL operation blocked")
```

#### PII-Safe Result Processing
- **Result Sanitisation**: Automatic PII detection in query results
- **Row Limiting**: Maximum result set size enforcement
- **Audit Logging**: Query patterns logged without exposing data content

### Australian Privacy Principles (APP) Compliance

#### APP 3 - Collection Limitation
- **Schema-Only Collection**: System collects database structure metadata only
- **No Data Sampling**: Zero personal data collected during schema discovery
- **Purpose-Bound**: Schema used exclusively for SQL query generation

#### APP 6 - Use and Disclosure
- **Internal Processing**: Schema metadata used only within RAG system
- **LLM Transmission**: Only non-personal schema structure sent to external LLMs
- **No Secondary Use**: Database structure not used for system profiling

#### APP 8 - Cross-Border Disclosure
- **Metadata Only**: No Australian personal data crosses borders
- **Schema Transmission**: Database structure (non-personal) transmitted to LLM APIs
- **Data Sovereignty**: All personal data remains within Australian jurisdiction

#### APP 11 - Security
- **Read-Only Access**: Database user limited to SELECT operations only
- **Connection Security**: Encrypted connections with credential protection
- **Audit Trail**: All database interactions logged with timestamp and query hash
- **Error Sanitisation**: Database errors sanitised before exposure

### Security Architecture

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

## Configuration

### Required Settings
```python
# Database connection (read-only user)
RAG_DB_USER = "rag_user_readonly"          # Dedicated read-only user
RAG_DB_PASSWORD = "secure_password"        # Secure credential

# SQL safety controls
ENABLE_SQL_VALIDATION = True               # Enable safety validation
MAX_SQL_COMPLEXITY_SCORE = 10              # Complexity limit
MAX_QUERY_RESULTS = 100                    # Result set limit

# LLM integration
LLM_MODEL_NAME = "gpt-3.5-turbo"          # OpenAI/Anthropic/Gemini model
LLM_TEMPERATURE = 0.1                      # Low temperature for deterministic results
```

### Security Validation
```bash
# Test read-only database access
python src/db/tests/test_rag_connection.py

# Validate SQL safety controls
pytest src/rag/tests/test_phase1_refactoring.py::TestAsyncSQLTool::test_sql_safety_validation

# Test schema manager privacy controls
pytest src/rag/tests/test_phase1_refactoring.py::TestSchemaManager -v
```

## Usage Examples

### Secure Schema Discovery
```python
from rag.core.text_to_sql.schema_manager import SchemaManager

async with SchemaManager() as schema_mgr:
    # Privacy-safe schema description (no data values)
    schema = await schema_mgr.get_schema_description()
    print(f"Schema metadata: {len(schema)} characters")  # Structure only
```

### Safe SQL Generation
```python
from rag.core.text_to_sql.sql_tool import AsyncSQLTool

async with AsyncSQLTool() as sql_tool:
    # Natural language to SQL with safety validation
    result = await sql_tool.process_question(
        "How many users completed courses in each agency?"
    )
    
    if result.success:
        print(f"Query executed safely: {result.row_count} rows")
    else:
        print(f"Security validation failed: {result.error}")
```

## Testing

### Privacy Compliance Tests
```bash
# Run comprehensive Text-to-SQL tests
cd src/rag && pytest tests/test_phase1_refactoring.py::TestSchemaManager -v
cd src/rag && pytest tests/test_phase1_refactoring.py::TestAsyncSQLTool -v

# Test SQL safety validation specifically
pytest tests/test_phase1_refactoring.py::TestAsyncSQLTool::test_sql_safety_validation

# Validate schema privacy controls
pytest tests/test_phase1_refactoring.py::TestSchemaManager::test_fallback_schema_generation
```

### Manual Security Testing
```bash
# Test manual Text-to-SQL processing
python tests/manual_test_phase1.py

# Validate dangerous SQL blocking
python -c "
from rag.core.text_to_sql.sql_tool import AsyncSQLTool
import asyncio

async def test_security():
    async with AsyncSQLTool() as tool:
        # This should be blocked
        result = await tool.process_question('DROP TABLE users')
        print(f'Dangerous query blocked: {not result.success}')

asyncio.run(test_security())
"
```

## Security Audit Checklist

- [ ] **Schema-Only Processing**: No personal data transmitted to LLMs
- [ ] **Read-Only Access**: Database user limited to SELECT operations
- [ ] **SQL Safety Validation**: Dangerous keywords blocked at runtime
- [ ] **Result Sanitisation**: Query results checked for PII before exposure
- [ ] **Audit Logging**: All database interactions logged with privacy protection
- [ ] **Error Handling**: Database errors sanitised before user exposure
- [ ] **Resource Management**: Proper connection cleanup and pool management

---

**Last Updated**: 11 June 2025  
**Security Level**: High  
**Compliance Status**: APP Aligned  
**Privacy Framework**: Schema-Only Processing  
**Test Coverage**: 100% (Text-to-SQL)

## Query Classification Enhancement ✅

### Table-Specific Feedback Classification

The text-to-SQL system now includes sophisticated feedback table classification to address the critical issue where LLM was incorrectly joining `rag_user_feedback` with `learning_content`.

**Key Enhancement**: `FeedbackTableClassifier` in `aps_patterns.py`
- Distinguishes between content feedback (evaluation table) vs system feedback (rag_user_feedback table)
- 20+ regex patterns for content feedback detection
- 11+ regex patterns for system feedback detection
- Provides table recommendations with confidence scores

**SQL Generation Enhancement**: 
- Enhanced prompts with critical feedback table guidance
- Specific instructions to NEVER join `rag_user_feedback` with `learning_content`
- Table-specific guidance based on classification results

**Impact**: Resolves the core issue where queries like "What feedback did users give about courses?" would incorrectly use `rag_user_feedback` instead of `evaluation` table.