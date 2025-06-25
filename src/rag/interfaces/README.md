# User Interfaces

This directory contains secure user interfaces for the RAG system, implementing comprehensive data governance controls for user interactions and external service integration.

## Overview

The interfaces module provides secure, compliant user interaction capabilities:
- **Terminal Application**: Phase 1 MVP terminal interface for Text-to-SQL queries  
- **Database Interface**: Read-only database connectivity with audit controls
- **LLM API Interface**: Secure multi-provider LLM integration with data sovereignty considerations
- **Monitoring Interface**: Audit logging and compliance monitoring integration

## Current Architecture

### Status: **Phase 1 Complete**

```
interfaces/
├── __init__.py                 # Interface module initialisation
├── README.md                  # This documentation
└── terminal_app.py            # Terminal MVP application (Phase 1)
```

## Implementation Status

### Phase 1: Terminal MVP Application (Complete)

#### Terminal Application (`terminal_app.py`)
- **`TerminalApp`**: Async terminal interface for natural language queries
- **`run_terminal_app()`**: Main entry point with asyncio event loop integration
- **Session Management**: UUID-based session tracking for audit purposes
- **Error Handling**: Secure error messages without credential exposure
- **Query Processing**: Integration with Text-to-SQL pipeline

#### Key Features Implemented
```python
# Current implementation highlights
class TerminalApp:
    """Terminal application for RAG Text-to-SQL system with async-first design."""
    
    async def run(self):
        """Main terminal loop with natural language query processing."""
        # Async event loop with proper resource management
        # Integration with AsyncSQLTool for Text-to-SQL processing
        # Secure error handling and result formatting
        
    async def process_query(self, question: str) -> None:
        """Process natural language question with comprehensive audit trail."""
        # Query validation and sanitisation
        # Integration with LLM-powered SQL generation
        # Result formatting with privacy protection
```

#### Security Architecture
- **Read-Only Access**: Terminal operations limited to database SELECT queries
- **Session Isolation**: Each terminal session isolated with unique identifiers
- **Input Sanitisation**: All user input validated before processing
- **Output Protection**: No sensitive data exposed in terminal output
- **Audit Trail**: Complete logging of terminal sessions and queries

## Data Governance Framework

### Current Security Controls

#### Terminal Interface Security
- **Read-Only Enforcement**: All terminal operations use read-only database credentials
- **Session Management**: UUID-based session tracking with comprehensive audit logging
- **Input Validation**: Natural language query sanitisation and validation
- **Output Protection**: Query results sanitised to prevent PII exposure
- **Resource Limits**: Query timeout and result size limits enforced

#### Integrated External Services
- **Database Interface**: Read-only database access through text_to_sql module
- **LLM API Interface**: Multi-provider LLM integration (OpenAI/Anthropic/Gemini)
- **Schema Interface**: Privacy-safe database schema provision to LLMs
- **Audit Interface**: Comprehensive logging with PII sanitisation

### Privacy Compliance Implementation

#### Australian Privacy Principles (APP) Compliance

**APP 3 (Collection of Personal Information)**
- **Terminal Input**: Only natural language queries collected, no personal identification
- **Session Data**: Minimal session metadata collected for audit purposes only
- **No Data Sampling**: Terminal interface never samples or stores actual database content

**APP 6 (Use or Disclosure of Personal Information)**  
- **Purpose-Bound Processing**: Terminal queries used solely for Text-to-SQL translation
- **No Secondary Use**: Terminal session data not used for analytics or profiling
- **Internal Processing**: All processing occurs within secure Australian jurisdiction

**APP 8 (Cross-border Disclosure of Personal Information)**
- **Schema-Only Transmission**: Only database schema structure sent to external LLMs
- **No Personal Data**: Zero transmission of actual user data to external APIs
- **Audit Trail**: Complete logging of cross-border schema transmissions

**APP 11 (Security of Personal Information)**
- **Encrypted Connections**: All external API communications use TLS encryption
- **Credential Protection**: Secure credential management through configuration system
- **Session Security**: Terminal sessions isolated with secure cleanup procedures
- **Error Sanitisation**: All terminal error messages sanitised before display

## Usage Examples

### Terminal Application Usage

#### Starting the Terminal Application
```bash
# Run the terminal MVP application
cd /Users/josh/Desktop/CP3101/ai-driven-analysis-project/src/rag
python runner.py

# Alternative direct execution
python -c "
import asyncio
from rag.interfaces.terminal_app import run_terminal_app
asyncio.run(run_terminal_app())
"
```

#### Example Terminal Session
```
🚀 RAG Text-to-SQL Terminal (Session: a1b2c3d4)
================================================
📋 Example queries:
  • How many users completed courses in each agency?
  • Show attendance status breakdown by user level
  • Which courses have the highest enrollment?

💬 Enter your question (or 'quit' to exit): How many users completed training?

🔍 Processing your question...
📊 Query Result:
┌─────────────────┬─────────┐
│ agency          │ count   │
├─────────────────┼─────────┤
│ Department A    │ 150     │
│ Department B    │ 127     │
│ Department C    │ 89      │
└─────────────────┴─────────┘

⏱️  Query executed in 2.3 seconds
📝 Query logged for audit (Session: a1b2c3d4)
```

#### Secure Query Processing
```python
# Example of terminal query processing with governance controls
from rag.interfaces.terminal_app import TerminalApp

async def secure_query_example():
    """Example of secure terminal query processing."""
    app = TerminalApp()
    
    # Question processing with comprehensive security
    question = "How many users completed courses in each agency?"
    
    # Security validation occurs automatically:
    # 1. Input sanitisation and validation
    # 2. Schema-only context provided to LLM
    # 3. SQL generation with safety validation
    # 4. Read-only database execution
    # 5. Result sanitisation and formatting
    
    await app.process_query(question)
    # Output: Secure, formatted results with audit trail
```

## Configuration Integration

### Interface Configuration
All interfaces integrate with the secure configuration system:

```python
# Example configuration integration
from rag.config.settings import get_settings

def initialise_interfaces():
    """Initialise all external service interfaces with secure configuration."""
    settings = get_settings()
    
    # Database interface with read-only access
    db_interface = SecureDatabaseInterface(
        connection_string=settings.rag_database_url,
        timeout=settings.query_timeout_seconds,
        audit_enabled=settings.log_sql_queries
    )
    
    # LLM interface with governance controls
    llm_interface = SecureLLMInterface(
        api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens
    )
    
    # Monitoring interface with compliance tracking
    monitoring_interface = ComplianceMonitoringInterface(
        log_level=settings.log_level,
        audit_sql=settings.log_sql_queries,
        debug_mode=settings.debug_mode
    )
    
    return InterfaceManager(db_interface, llm_interface, monitoring_interface)
```

## Security Testing Strategy

### Planned Test Coverage

#### Database Interface Testing
- **Connection Security**: TLS validation and credential security
- **Read-Only Enforcement**: Attempt write operations (should fail)
- **SQL Injection**: Comprehensive injection attack testing
- **Audit Completeness**: Verify all operations are logged

#### LLM Interface Testing
- **API Security**: Credential handling and encrypted communications
- **Data Transmission**: Verify only schema metadata is transmitted
- **Response Validation**: Test LLM response sanitisation
- **Cross-Border Compliance**: Validate data sovereignty controls

#### Monitoring Interface Testing
- **Audit Trail**: Comprehensive logging validation
- **Privacy Masking**: Verify sensitive data masking in logs
- **Compliance Metrics**: Validate privacy policy adherence tracking
- **Incident Detection**: Test security event detection and alerting

### Security Test Examples
```bash
# Planned test commands
cd src/rag && pytest tests/test_interfaces/ -v

# Database security tests
pytest tests/test_interfaces/test_database_security.py -v

# LLM API governance tests  
pytest tests/test_interfaces/test_llm_governance.py -v

# Monitoring compliance tests
pytest tests/test_interfaces/test_monitoring_compliance.py -v
```

## Development Guidelines

### Security Requirements
1. **Encrypted Communications**: All external connections must use TLS/SSL
2. **Credential Security**: All credentials must be loaded from secure configuration
3. **Audit Logging**: All interface operations must be comprehensively logged
4. **Error Handling**: All errors must be handled securely without data exposure
5. **Input Validation**: All external inputs must be validated before processing

### Privacy Requirements
1. **Data Minimisation**: Only transmit data necessary for operation
2. **Purpose Limitation**: All data usage must align with stated purposes
3. **Retention Policies**: Implement appropriate data retention and deletion
4. **Anonymisation**: Apply anonymisation where technically feasible
5. **Consent Management**: Respect user consent boundaries in all operations

### Compliance Requirements
1. **APP Compliance**: All interfaces must align with Australian Privacy Principles
2. **Audit Trail**: Maintain comprehensive audit logs for compliance
3. **Data Sovereignty**: Document and control cross-border data transfers
4. **Incident Response**: Implement security incident detection and response
5. **Regular Review**: Conduct regular privacy and security reviews

## Future Enhancements

### Advanced Interface Features
- **Multi-Provider LLM**: Support for multiple LLM providers with unified governance
- **Advanced Caching**: Intelligent caching with privacy-preserving techniques
- **Real-time Monitoring**: Enhanced real-time compliance monitoring
- **Automated Compliance**: Automated privacy policy compliance validation

### Enhanced Security Features
- **Zero-Trust Interfaces**: Enhanced access controls and validation
- **Homomorphic Encryption**: Processing encrypted data without decryption
- **Federated Interfaces**: Distributed processing without data centralisation
- **Blockchain Audit**: Immutable audit trail using blockchain technology

---

**Status**: Phase 1 Complete  
**Security Priority**: Critical  
**Privacy Compliance**: APP Aligned  
**Data Governance**: Comprehensive  
**Last Updated**: 11 June 2025

## Feedback System Integration

### User Feedback Collection & Analytics

The terminal application includes a comprehensive feedback system for collecting user ratings and improving system quality over time.

#### Features
- **1-5 Scale Rating Collection**: Professional rating system with clear user prompts
- **Optional Comments**: Users can provide detailed feedback with automatic PII anonymisation
- **Database Storage**: Secure storage in `rag_user_feedback` table with Australian Privacy Principles compliance
- **On-Demand Analytics**: `/feedback-stats` command provides comprehensive feedback analysis
- **Privacy Protection**: Automatic detection and anonymisation of emails, phone numbers, and names
- **Configurable**: Easy enable/disable via environment variables

#### User Experience Flow
```
📝 Help us improve! Please rate your experience:
   1⭐ - Very poor    2⭐ - Poor       3⭐ - Average
   4⭐ - Good         5⭐ - Excellent

Rate this response (1-5, or 'skip'): 4
Optional comment (press Enter to skip): Very helpful analysis

✅ Thank you for the positive feedback!
💾 Feedback stored for analysis and improvements.
```

#### Analytics Display
```
📊 Feedback Analysis (30 days)
==================================================
Total responses: 25
Average rating: 4.2/5.0

📈 Rating Distribution:
  1⭐:   1 ( 4.0%) 
  2⭐:   2 ( 8.0%) █
  3⭐:   5 (20.0%) ████
  4⭐:  12 (48.0%) █████████
  5⭐:   5 (20.0%) ████

💬 Recent Comments (5):
  1. "Great system, very accurate results"
  2. "Could be faster but very helpful"  
  3. "Excellent analysis of the data"
```

#### Commands Available
- **Standard queries**: Natural language questions processed by RAG system
- **`examples`**: Display example queries for different analysis types
- **`help`**: Show available commands and usage information
- **`stats`**: Display session statistics (agent mode only)
- **`/feedback-stats`**: View comprehensive feedback analytics
- **`quit`/`exit`**: Exit the application

#### Configuration
```python
# Environment Variables
ENABLE_FEEDBACK_COLLECTION=true      # Master enable/disable switch
FEEDBACK_DATABASE_ENABLED=true       # Control database storage
```

#### Privacy & Security
- **Australian Privacy Principles Compliant**: All feedback handling follows APP guidelines
- **PII Anonymisation**: Automatic detection and masking of sensitive information
- **Error Resilience**: Feedback failures don't impact query processing
- **Data Minimisation**: Only necessary data is collected and stored
