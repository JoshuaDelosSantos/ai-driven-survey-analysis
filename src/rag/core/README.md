# Core RAG Functionality

This directory contains the core Retrieval-Augmented Generation functionality for the hybrid Text-to-SQL and vector search system, implementing secure natural language processing with comprehensive Australian data governance controls and mandatory PII protection.

## Overview

The core module implements the central RAG pipeline with privacy-first design:
- **Conversational Intelligence**: Advanced pattern recognition with Australian-friendly responses ✅ **NEW**
- **LangGraph Agent Orchestration**: Intelligent query routing and processing coordination ✅ **NEW**
- **Text-to-SQL Translation**: LangChain-powered natural language query understanding and SQL generation
- **Vector Search Integration**: Semantic search over user feedback with advanced metadata filtering ✅ **NEW**
- **Query Classification**: Multi-stage intelligent routing between SQL, vector, and hybrid processing ✅ **NEW**
- **Answer Synthesis**: Advanced multi-modal response generation combining statistical and qualitative insights ✅ **NEW**
- **Dynamic Schema Management**: Real-time database schema introspection with privacy controls
- **Australian PII Protection**: Mandatory detection and anonymisation of Australian-specific entities
- **Query Validation**: Multi-layer SQL injection prevention and complexity analysis
- **Secure Execution**: Read-only database access with comprehensive audit trail

## Current Architecture

### Status: **Phase 1 Complete + Phase 2 Complete + Phase 3 Task 3.1 Complete**

```
core/
├── __init__.py                 # Core module initialisation
├── agent.py                   # LangGraph agent orchestrator ✅ NEW  
├── README.md                  # This documentation
├── conversational/            # Conversational intelligence system ✅ NEW
│   ├── __init__.py
│   ├── README.md             # Conversational intelligence documentation
│   └── handler.py            # Pattern recognition and response generation
├── privacy/                   # Australian PII detection and anonymisation ✅ NEW
│   ├── __init__.py
│   ├── README.md             # Privacy module documentation
│   └── pii_detector.py       # Australian-specific PII detection
├── routing/                   # Query classification and routing ✅ NEW
│   ├── __init__.py
│   ├── README.md             # Query routing documentation
│   └── query_classifier.py   # Multi-stage query classification
├── synthesis/                 # Answer generation and synthesis ✅ NEW
│   ├── __init__.py
│   ├── README.md             # Answer synthesis documentation
│   └── answer_generator.py   # Multi-modal answer generation
├── text_to_sql/              # Text-to-SQL processing engine
│   ├── __init__.py
│   ├── README.md             # Text-to-SQL documentation
│   ├── schema_manager.py     # Dynamic schema introspection
│   └── sql_tool.py          # SQL generation and execution
└── vector_search/            # Vector search and semantic retrieval
    ├── __init__.py
    ├── README.md             # Vector search documentation
    └── vector_search_tool.py # Async vector search implementation
```

## Enhanced Data Governance Framework

### Phase 2 Security Enhancements

#### Mandatory Australian PII Protection ✅ **NEW**
- **Presidio Integration**: Microsoft Presidio with custom Australian entity recognisers
- **Australian Business Entities**: ABN, ACN, TFN, Medicare number detection and anonymisation
- **Zero PII Transmission**: All text anonymised before LLM processing or vector embedding
- **Compliance Enforcement**: Non-negotiable PII detection for all user inputs and database results

#### Enhanced Schema-Only Processing
- **Privacy-Safe Schema Discovery**: Database structure introspection without data sampling
- **PII-Protected Metadata**: Schema transmission with mandatory anonymisation validation
- **LLM Context Limitation**: Only sanitised table names, column names, and relationships transmitted
- **Zero Personal Data Transmission**: Enhanced validation ensuring no Australian PII in LLM calls

#### Enhanced Multi-Layer Query Validation
- **SQL Injection Prevention**: Comprehensive dangerous keyword detection and blocking
- **Read-Only Enforcement**: Database user limited to SELECT operations with startup validation
- **Complexity Scoring**: Configurable query complexity limits with timeout controls
- **Resource Protection**: Enhanced connection pooling and execution limits with session management

#### Comprehensive Audit Trail with PII Protection
- **Query Fingerprinting**: All database interactions logged with enhanced privacy protection
- **Australian PII Sanitisation**: Automatic ABN, ACN, TFN, Medicare number detection and masking
- **Execution Monitoring**: Real-time query performance and security monitoring with privacy compliance
- **Error Sanitisation**: Database errors sanitised with comprehensive PII protection before user exposure

### Enhanced Privacy Compliance Implementation

#### Australian Privacy Principles (APP) Alignment - Phase 2 Enhanced

**APP 3 (Collection)**
- **Schema-Only Collection**: System collects database structure metadata only
- **No Data Sampling**: Zero personal data collected during schema discovery
- **Purpose-Bound Processing**: Schema used exclusively for SQL query generation with mandatory PII anonymisation

**APP 6 (Use/Disclosure)**
- **Internal Processing**: Schema metadata used only within RAG system with PII protection
- **LLM Transmission**: Only non-personal, anonymised schema structure sent to external LLMs
- **No Secondary Use**: Database structure not used for system profiling, all PII anonymised

**APP 8 (Cross-border Disclosure)**
- **Metadata Only**: No Australian personal data crosses borders (enhanced validation)
- **Anonymised Transmission**: Only PII-anonymised schema structure transmitted to LLM APIs
- **Data Sovereignty**: All personal data remains within Australian jurisdiction with mandatory detection

**APP 11 (Security)**
- **Read-Only Access**: Database user limited to SELECT operations with startup validation
- **Connection Security**: Encrypted connections with credential protection and session management
- **Enhanced Audit Trail**: All interactions logged with Australian PII protection and compliance metadata
- **Comprehensive Error Sanitisation**: Database errors sanitised with full PII protection before exposure

## Enhanced Implementation Status

### Phase 1: Text-to-SQL Engine (Complete)

#### ✅ Schema Management (`text_to_sql/schema_manager.py`)
- **`SchemaManager`**: Async database schema introspection and caching
- **`TableInfo`**: Structured table metadata with privacy annotations
- **Dynamic Schema Discovery**: Real-time database structure reading
- **Privacy Controls**: Schema-only transmission, no data sampling
- **LangChain Integration**: SQLDatabase wrapper with enhanced security controls

#### ✅ SQL Processing (`text_to_sql/sql_tool.py`)
- **`AsyncSQLTool`**: LangChain-integrated SQL generation and execution
- **`SQLResult`**: Structured query result with execution metadata
- **Multi-Provider LLM Support**: OpenAI, Anthropic, and Google Gemini with live validation
- **Safety Validation**: Dangerous keyword detection and complexity scoring
- **Secure Execution**: Read-only database access with comprehensive audit logging

### Phase 2: Australian PII Protection (Complete) ✅ **NEW**

#### ✅ PII Detection System (`privacy/pii_detector.py`)
- **`AustralianPIIDetector`**: Async PII detection with Microsoft Presidio integration
- **Australian Entity Recognition**: Custom ABN, ACN, TFN, Medicare number patterns
- **Batch Processing**: Efficient multi-text processing with session management
- **Mandatory Anonymisation**: Non-negotiable PII protection before any LLM processing
- **Global Singleton**: Optimised session-scoped instance with proper resource cleanup

#### ✅ Enhanced Security Architecture Implementation
```
┌─────────────────────────────────────────────────────────────┐
│           Enhanced Text-to-SQL Security Layers             │
├─────────────────────────────────────────────────────────────┤
│ 1. Natural Language Input + PII Protection                 │
│    ├─ Input sanitisation and validation                    │
│    ├─ Mandatory Australian PII detection (ABN/ACN/TFN/Medicare)│
│    └─ Question classification and routing                  │
├─────────────────────────────────────────────────────────────┤
│ 2. Schema Processing (Privacy-Safe + PII Anonymised)       │
│    ├─ Database structure introspection only                │
│    ├─ No data sampling or content analysis                 │
│    ├─ Metadata anonymisation before LLM transmission       │
│    └─ PII-protected metadata-only transmission to LLM      │
├─────────────────────────────────────────────────────────────┤
│ 3. SQL Generation (LLM Integration + PII Protection)       │
│    ├─ Anonymised schema-only context provided to LLM       │
│    ├─ Multi-shot prompting with approved examples          │
│    └─ Response validation and safety checking              │
├─────────────────────────────────────────────────────────────┤
│ 4. SQL Safety Validation                                   │
│    ├─ Dangerous keyword detection and blocking             │
│    ├─ Query complexity scoring and limiting                │
│    └─ Read-only operation enforcement with startup validation│
├─────────────────────────────────────────────────────────────┤
│ 5. Query Execution (Read-Only + Session Management)        │
│    ├─ Dedicated read-only database user with validation    │
│    ├─ Enhanced result set limiting                         │
│    └─ Execution timeout enforcement with session tracking  │
├─────────────────────────────────────────────────────────────┤
│ 6. Result Processing (Enhanced PII Protection)             │
│    ├─ Australian entity sanitisation (ABN/ACN/TFN/Medicare)│
│    ├─ Comprehensive audit logging with PII masking         │
│    └─ Structured response formatting with privacy compliance│
└─────────────────────────────────────────────────────────────┘
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

### Phase 3: Conversational Intelligence (Complete) ✅ **NEW**

#### ✅ Conversational Handler (`conversational/handler.py`)
- **`ConversationalHandler`**: Advanced pattern recognition with Australian-friendly responses
- **Pattern Categories**: 25+ conversation patterns including greetings, system inquiries, social interactions
- **Response Generation**: Multi-variant Australian templates with context-aware selection
- **Learning System**: Feedback-driven pattern recognition with continuous improvement
- **Privacy Integration**: All conversational data processed with mandatory PII protection

#### ✅ Australian-Friendly Response System
```python
# Example conversational interactions
GREETING_PATTERNS = [
    "hello", "hi", "g'day", "good morning", "how are you"
]

AUSTRALIAN_RESPONSES = [
    "G'day! I'm doing well, thanks for asking. How can I help you today?",
    "Hello! I'm here and ready to assist you with your learning analytics queries.",
    "Good day! I'm operating perfectly and ready to help you explore the data."
]

CAPABILITY_RESPONSES = [
    "I can help you analyse learning and development data in several ways:\n\n"
    "📊 **Data Analysis**: Ask questions about course completions, attendance rates\n"
    "🔍 **Data Exploration**: Browse datasets and understand available information\n"
    "📈 **Trend Analysis**: Identify patterns in training participation\n"
    "🎯 **Targeted Insights**: Filter data by agency, user level, or time period"
]
```

#### ✅ Intelligent Query Routing Integration
```python
# Conversational detection integrated with query classification
async def classify_query(self, query: str) -> ClassificationResult:
    """Classify query with conversational pattern detection."""
    
    # Check for conversational patterns first
    conversational_result = await self.conversational_handler.detect_pattern(query)
    
    if conversational_result.confidence > 0.8:
        return ClassificationResult(
            classification_type=ClassificationType.CONVERSATIONAL,
            confidence=conversational_result.confidence,
            method_used="CONVERSATIONAL_PATTERN_MATCH"
        )
    
    # Continue with data analysis classification
    return await self.classify_data_query(query)
```

#### ✅ Pattern Learning and Feedback Integration
- **Feedback Collection**: User ratings (1-5 stars) improve pattern recognition
- **Pattern Weights**: Automatic adjustment based on user feedback
- **Usage Analytics**: Tracks pattern effectiveness and user satisfaction
- **Privacy-First Learning**: All learning data anonymised and privacy-protected

### Phase 3: LangGraph Agent Orchestration (Complete) ✅ **NEW**
