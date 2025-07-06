# User Interfaces

This directory contains secure user i        # Enhanced async event loop with auto-embedding and feedback integration
        # System commands: /feedback-stats, /help
        # Graceful error handling with user guidance
        
    async def process_query(self, question: str) -> None:
        """Process query with hybrid routing and optional feedback collection."""
        # LangGraph agent integration with intelligent routing
        # Automatic PII anonymisation across all processing stages
        # Optional post-query feedback collection with privacy protection
        
    async def _ensure_embeddings_ready(self) -> None:
        """Automatic embedding processing to ensure vector search readiness."""
        # Incremental processing - only processes new/missing records
        # Progress indicators for user feedback during initial setup
        # Batch processing with configurable performance settings
        # Error resilience with graceful degradation
        
class FeedbackComponent:
    """User feedback collection with privacy protection and analytics."""
    
    async def collect_feedback(self, session_id: str, response_id: str) -> bool:
        """Collect 1-5 scale rating with optional anonymous comments."""
        # Automatic PII detection and anonymisation in comments
        # Validation and error handling for rating input
        # Privacy-first storage with audit compliance
```

## Auto-Embedding System

### Intelligent Startup Processing
The terminal application includes an intelligent auto-embedding system that ensures vector search is always ready:

#### Key Features
- **Startup Processing**: Automatically checks and processes missing embeddings when the application starts
- **Incremental Updates**: Only processes new evaluation records that don't have embeddings yet
- **Progress Feedback**: Real-time progress indicators during processing
- **Performance Optimized**: Configurable batch sizes and concurrent processing for optimal performance
- **Error Resilient**: Graceful handling of processing errors with system continuity

#### User Experience
```
🚀 RAG System - Production Ready (Phase 3 Complete + Modular Architecture)
===============================================================================

🔍 Checking embedding status...
📊 Embedding status: 91/100 records processed
🔄 Processing 9 new records...
   Processing batch 1/1 (9 records)...
   ✅ Batch 1 completed: 9/9 records processed
🎉 Embedding processing completed in 12.3s
✅ RAG system ready for feedback analysis queries!

🧠 Ask questions about learning data using natural language!
   The system intelligently routes your questions to:
   📊 SQL Analysis - for statistics and numerical data
   💬 Vector Search - for user feedback and experiences  
   🔄 Hybrid Processing - for comprehensive insights
```

#### Technical Implementation
```python
async def _ensure_embeddings_ready(self) -> None:
    """Ensure all evaluation records have embeddings for vector search."""
    
    # Check current status
    total_with_content = await self._count_evaluation_records_with_content()
    embedded_count = await self._count_embedded_records()
    missing_count = total_with_content - embedded_count
    
    if missing_count == 0:
        print(f"✅ Embeddings up-to-date: {embedded_count}/{total_with_content} records ready")
        return
    
    # Process missing embeddings with progress updates
    async with ContentProcessor(config) as processor:
        results = await processor.process_evaluation_records(missing_ids)
        print(f"🎉 Embedding processing completed!")
```

#### Configuration Options
- **Batch Size**: Configurable batch sizes for performance tuning (default: 25)
- **Concurrent Processing**: Adjustable concurrency levels (default: 3 for startup)
- **Progress Logging**: Enable/disable detailed progress feedback
- **Error Handling**: Configurable retry policies and error recovery

#### Benefits
1. **Zero Maintenance**: No manual script execution required
2. **Always Ready**: Vector search capabilities always available
3. **Performance Optimized**: Only processes what's needed
4. **User Friendly**: Clear progress feedback during processing
5. **Production Safe**: Robust error handling ensures system stabilityAG system, implementing comprehensive data governance controls, user feedback collection, automatic embedding management, and privacy-first interaction design.

## Overview

The interfaces module provides secure, compliant user interaction capabilities:
- **Enhanced Terminal Application**: Phase 3 complete terminal interface with feedback integration, auto-embedding, and hybrid query processing
- **Feedback Collection Interface**: 1-5 scale rating system with anonymous comment collection and PII protection
- **Auto-Embedding System**: Intelligent startup embedding processing to ensure vector search readiness
- **Database Interface**: Read-only database connectivity with feedback table support and audit controls
- **LLM API Interface**: Secure multi-provider LLM integration with data sovereignty considerations
- **Analytics Interface**: Real-time feedback analytics and system performance monitoring

## Current Architecture

### Status: **Phase 3 Complete - Production Ready with Auto-Embedding & User Feedback Analytics**

```
interfaces/
├── __init__.py                 # Interface module initialisation
├── README.md                  # This documentation
└── terminal_app.py            # Enhanced terminal application with auto-embedding and feedback system (Phase 3)
```

## Implementation Status

### Phase 3: Enhanced Terminal Application with Auto-Embedding & Feedback System (Complete)

#### Enhanced Terminal Application (`terminal_app.py`)
- **`TerminalApp`**: Async terminal interface with hybrid query processing, auto-embedding, and feedback collection
- **`_ensure_embeddings_ready()`**: Automatic embedding processing on startup to ensure vector search readiness ✅ NEW
- **`FeedbackComponent`**: Integrated 1-5 scale rating system with optional anonymous comments ✅ Enhanced
- **`run_terminal_app()`**: Main entry point with auto-embedding, feedback analytics, and system commands ✅ Enhanced
- **Session Management**: Enhanced UUID-based session tracking with feedback correlation for audit purposes
- **Error Handling**: Robust error recovery with graceful degradation and user guidance ✅ Enhanced
- **Query Processing**: Integration with LangGraph agent for hybrid Text-to-SQL and vector search ✅ Enhanced

#### Key Features Implemented (Phase 3)
```python
# Enhanced implementation highlights
class TerminalApp:
    """Enhanced terminal application for hybrid RAG system with auto-embedding and feedback analytics."""
    
    def __init__(self):
        """Initialize with auto-embedding and feedback collection integration."""
        self.feedback_component = FeedbackComponent()  # Enhanced: Feedback integration
        self.agent = RAGAgent()  # Enhanced: Hybrid agent integration
        
    async def initialize(self):
        """Initialize with automatic embedding processing."""
        # Auto-process embeddings to ensure vector search readiness
        await self._ensure_embeddings_ready()  # NEW: Auto-embedding on startup
        
    async def run(self):
        """Main terminal loop with auto-embedding and feedback collection."""
        # Enhanced async event loop with auto-embedding and feedback integration
        # System commands: /feedback-stats, /help
        # Graceful error handling with user guidance
        
    async def process_query(self, question: str) -> None:
        """Process query with hybrid routing and optional feedback collection."""
        # LangGraph agent integration with intelligent routing
        # Automatic PII anonymisation across all processing stages
        # Optional post-query feedback collection with privacy protection
        
class FeedbackComponent:
    """User feedback collection with privacy protection and analytics."""
    
    async def collect_feedback(self, session_id: str, response_id: str) -> bool:
        """Collect 1-5 scale rating with optional anonymous comments."""
        # Automatic PII detection and anonymisation in comments
        # Validation and error handling for rating input
        # Privacy-first storage with audit compliance
```

#### Enhanced Security Architecture (Phase 3)
- **Read-Only Access**: Terminal operations limited to database SELECT queries with feedback table support
- **Session Isolation**: Each terminal session isolated with unique identifiers and feedback correlation
- **Input Sanitisation**: All user input (queries and feedback) validated and anonymised before processing ✅ Enhanced
- **Output Protection**: No sensitive data exposed in terminal output with feedback privacy compliance ✅ Enhanced
- **Audit Trail**: Complete logging of terminal sessions, queries, and feedback with privacy protection ✅ Enhanced
- **Feedback Privacy**: All user comments automatically anonymised with Australian entity detection ✅ Enhanced
- **Auto-Embedding Security**: Startup embedding processing with PII anonymisation and secure storage ✅ NEW
- **Error Recovery**: Graceful degradation with user guidance and session continuity ✅ Enhanced

## Data Governance Framework

### Current Security Controls

#### Enhanced Terminal Interface Security (Phase 3)
- **Read-Only Enforcement**: All terminal operations use read-only database credentials with feedback table access
- **Session Management**: UUID-based session tracking with feedback correlation and comprehensive audit logging ✅ Enhanced
- **Input Validation**: Natural language query and feedback comment sanitisation and validation ✅ Enhanced
- **Output Protection**: Query results and feedback analytics sanitised to prevent PII exposure ✅ Enhanced
- **Resource Limits**: Query timeout and result size limits enforced with feedback collection efficiency ✅ Enhanced
- **Feedback Privacy**: Automatic PII detection and anonymisation in all user feedback ✅ Enhanced
- **Auto-Embedding Security**: Secure background processing with privacy protection during startup ✅ NEW
- **System Commands**: Secure access to feedback analytics through `/feedback-stats` command ✅ Enhanced

#### Enhanced Integrated External Services (Phase 3)
- **LangGraph Agent Interface**: Hybrid query processing with intelligent routing and confidence scoring ✅ NEW
- **Feedback Analytics Interface**: Real-time feedback trend analysis and sentiment classification ✅ NEW
- **Database Interface**: Read-only database access with feedback table support through core modules ✅ Enhanced
- **LLM API Interface**: Multi-provider LLM integration (OpenAI/Anthropic/Gemini) with feedback context ✅ Enhanced
- **Schema Interface**: Privacy-safe database schema provision to LLMs with feedback table metadata ✅ Enhanced
- **Audit Interface**: Comprehensive logging with PII sanitisation and feedback system compliance ✅ Enhanced

### Enhanced Privacy Compliance Implementation (Phase 3)

#### Australian Privacy Principles (APP) Compliance with Feedback System

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

The terminal application includes a comprehensive feedback system for collecting user ratings and improving system quality over time. All feedback is stored in the dedicated `rag_user_feedback` table with full privacy protection.

#### Enhanced Database Schema
The `rag_user_feedback` table provides comprehensive feedback storage:

```sql
-- Core feedback tracking
feedback_id           SERIAL PRIMARY KEY
session_id           VARCHAR(50)        -- Session correlation
query_id             VARCHAR(50)        -- Query correlation
query_text           TEXT               -- PII-anonymised user question
response_text        TEXT               -- System response (truncated)
rating              INTEGER            -- 1-5 scale rating
comment             TEXT               -- Optional PII-anonymised comment
response_sources    TEXT[]             -- Data sources used
created_at          TIMESTAMP          -- Feedback timestamp
```

#### Features
- **1-5 Scale Rating Collection**: Professional rating system with clear user prompts
- **Optional Comments**: Users can provide detailed feedback with automatic PII anonymisation
- **Database Storage**: Secure storage in `rag_user_feedback` table with Australian Privacy Principles compliance
- **On-Demand Analytics**: `/feedback-stats` command provides comprehensive feedback analysis
- **Privacy Protection**: Automatic detection and anonymisation of emails, phone numbers, and names
- **Session Correlation**: Full traceability between queries, responses, and feedback
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

#### Enhanced Analytics Display
```
📊 Feedback Analysis (30 days)
==================================================
Total responses: 127
Average rating: 4.3/5.0

📈 Rating Distribution:
  1⭐:   3 ( 2.4%) █
  2⭐:   8 ( 6.3%) ██
  3⭐:  15 (11.8%) ████
  4⭐:  67 (52.8%) █████████████
  5⭐:  34 (26.8%) ███████

💬 Recent Comments (5):
  1. "Great system, very accurate results"
  2. "Could be faster but very helpful"  
  3. "Excellent analysis of the data"
  4. "Auto-embedding on startup is brilliant"
  5. "Love the comprehensive feedback options"

🔄 Query Types Rated:
  SQL Analysis: 89 responses (avg: 4.2/5.0)
  Vector Search: 23 responses (avg: 4.5/5.0)
  Hybrid Analysis: 15 responses (avg: 4.6/5.0)
```

#### Commands Available
- **Standard queries**: Natural language questions processed by RAG system
- **`examples`**: Display example queries for different analysis types
- **`help`**: Show available commands and usage information
- **`stats`**: Display session statistics (agent mode only)
- **`/feedback-stats`**: View comprehensive feedback analytics with database insights
- **`quit`/`exit`**: Exit the application

#### Configuration
```python
# Environment Variables
ENABLE_FEEDBACK_COLLECTION=true      # Master enable/disable switch
FEEDBACK_DATABASE_ENABLED=true       # Control database storage
```

#### Privacy & Security
- **Australian Privacy Principles Compliant**: All feedback handling follows APP guidelines
- **PII Anonymisation**: Automatic detection and masking of sensitive information in all text fields
- **Error Resilience**: Feedback failures don't impact query processing
- **Data Minimisation**: Only necessary data is collected and stored
- **Audit Trail**: Complete feedback collection logging for compliance
- **Session Isolation**: Feedback data isolated by session for privacy protection

## Conversational Intelligence Features

### Australian-Friendly Interaction Design

The terminal application includes comprehensive conversational intelligence capabilities that provide natural, Australian-friendly interactions:

#### Key Features
- **Pattern Recognition**: Automatic detection of conversational queries (greetings, system questions, social interactions)
- **Australian Context**: Responses tailored for Australian professional environments with appropriate tone and terminology
- **Intelligent Routing**: Seamless switching between conversational and analytical modes based on query confidence
- **Learning System**: Pattern recognition improves based on user feedback and interaction history
- **Privacy Protection**: All conversational data processed with mandatory PII detection and anonymisation

#### Conversational Interaction Examples

```
🤖 Terminal Application - Enhanced with Conversational Intelligence
===============================================================

Your question: Hello, how are you?

🤖 G'day! I'm doing well, thanks for asking. I'm here to help you explore 
   the learning analytics data. How can I assist you today?

💡 You might want to try:
   • "What data do you have access to?"
   • "Show me course completion rates"
   • "Which agencies have the highest training participation?"

---

Your question: What can you help me with?

🤖 I can help you analyse learning and development data in several ways:

📊 **Data Analysis**: Ask questions about course completions, attendance rates, user performance
🔍 **Data Exploration**: Browse available datasets and understand what information is available
📈 **Trend Analysis**: Identify patterns in training participation and completion rates
🎯 **Targeted Insights**: Filter data by agency, user level, course type, or time period

💡 Some example questions to get you started:
   • "How many users completed courses in each agency?"
   • "Show attendance patterns by user level"
   • "What are the most popular courses?"

Would you like to explore any specific area?

---

Your question: Thanks for your help!

🤖 You're very welcome! Happy to help you explore the data anytime. 
   Feel free to ask if you need anything else - I'm here whenever you need 
   assistance with your learning analytics.

Have a great day! 🌟
```

#### Technical Implementation

```python
class TerminalApp:
    """Terminal application with conversational intelligence integration."""
    
    def __init__(self):
        """Initialize with conversational handler integration."""
        self.conversational_handler = ConversationalHandler()
        self.agent = RAGAgent()
        
    async def process_query(self, question: str) -> None:
        """Process query with intelligent routing between conversational and analytical modes."""
        
        # Agent automatically routes based on confidence
        result = await self.agent.process_query(question)
        
        # Display conversational responses with special formatting
        if result.method_used == "CONVERSATIONAL":
            self.display_conversational_response(result)
        else:
            self.display_analytical_response(result)
    
    def display_conversational_response(self, result: AgentResult):
        """Display conversational response with Australian-friendly formatting."""
        
        # Special emoji and formatting for conversational responses
        print(f"\n🤖 {result.response}")
        
        # Show suggested queries if available
        if result.suggested_queries:
            print(f"\n💡 You might want to try:")
            for i, query in enumerate(result.suggested_queries, 1):
                print(f"   • {query}")
        
        # Collect feedback for pattern learning
        feedback = self.collect_feedback("conversational")
        if feedback:
            # Feedback improves pattern recognition over time
            self.conversational_handler.record_feedback(
                query=result.original_query,
                pattern_type=result.pattern_type,
                feedback=feedback
            )
```

#### Conversational Pattern Categories

1. **Greeting Patterns**: "Hello", "Hi", "G'day", "Good morning"
2. **System Information**: "What can you do?", "How do you work?", "What data do you have?"
3. **Social Interactions**: "Thank you", "Please", "Goodbye", "How are you?"
4. **Meta Questions**: "Are you working?", "How fast are you?", "Can I trust you?"
5. **Support Requests**: "I need help", "I don't understand", "Can you assist me?"

#### Learning and Adaptation

The conversational system learns from user interactions:

- **Feedback Integration**: User ratings (1-5 stars) improve pattern recognition
- **Usage Analytics**: Tracks which patterns are most effective
- **Confidence Adjustment**: Pattern weights adapt based on success rates
- **Privacy-First Learning**: All learning data anonymised and privacy-protected

#### Benefits for Users

1. **Natural Interaction**: Conversational queries feel like talking to a colleague
2. **Australian Context**: Responses appropriate for Australian professional environments
3. **Intelligent Guidance**: System suggests relevant queries based on conversation context
4. **Seamless Experience**: Smooth transition between conversational and analytical modes
5. **Continuous Improvement**: System gets better at understanding user needs over time
