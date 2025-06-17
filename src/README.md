# AI-Driven Analysis Project - Source Code

This directory contains the core source code modules for the AI-driven analysis project, designed with Australian Privacy Principles (APP) compliance and data sovereignty as foundational requirements.

## Project Architecture

The system implements a privacy-first, modular architecture supporting Australian Public Service evaluation data analysis while maintaining strict data governance controls and comprehensive PII protection.

### Core Modules

#### `csv/` - Data Foundation
**Purpose**: Source data repository with comprehensive schema documentation
- **Contents**: User, evaluation, learning content, and attendance datasets
- **Data Dictionary**: Complete schema definitions with privacy classifications
- **Free-Text Analysis**: Three key evaluation fields for semantic processing
- **Governance**: Australian entity-aware data structures and privacy controls

#### `db/` - Database Infrastructure
**Purpose**: Secure database operations with read-only access patterns
- **Table Management**: Automated table creation and data loading scripts
- **Security Architecture**: RAG read-only role with minimal privileges
- **Vector Storage**: pgvector integration with configurable dimensions
- **Testing Infrastructure**: Comprehensive validation of security constraints
- **Compliance**: Designed for Australian Privacy Principles alignment

#### `rag/` - Retrieval-Augmented Generation
**Purpose**: AI-powered query processing with privacy protection
- **Multi-Provider LLM**: OpenAI, Anthropic, and Google Gemini integration
- **Local Embeddings**: Privacy-focused sentence transformers (all-MiniLM-L6-v2)
- **PII Protection**: Australian entity detection and anonymisation
- **Vector Search**: Semantic similarity search with metadata filtering
- **Async Architecture**: Production-ready async/await patterns

#### `sentiment-analysis/` - Text Analytics
**Purpose**: Sentiment analysis of evaluation free-text with privacy controls
- **Multi-Field Processing**: Analysis of evaluation feedback and comments
- **Privacy Protection**: PII detection and anonymisation before processing
- **Database Integration**: Secure storage of sentiment scores and metrics
- **Australian Context**: Optimised for Australian Public Service language patterns

## Data Privacy & Governance

### Australian Privacy Principles (APP) Compliance

#### Core Privacy Controls
- **APP 1 (Transparency)**: Comprehensive audit logging with privacy protection
- **APP 3 (Collection)**: Minimal data collection with mandatory anonymisation
- **APP 5 (Notification)**: Clear data usage transparency in processing workflows
- **APP 6 (Use/Disclosure)**: PII anonymisation before any processing or analysis
- **APP 8 (Cross-border)**: Data sovereignty controls with anonymised-only transmission
- **APP 11 (Security)**: Multi-layered security with Australian entity protection
- **APP 12 (Access)**: Privacy-protected audit trails and access logging

#### Australian Entity Protection
- **Business Identifiers**: ABN, ACN detection and anonymisation
- **Personal Identifiers**: TFN, Medicare number protection
- **Standard PII**: Email, phone, names, locations anonymisation
- **Context Preservation**: Semantic meaning retained while protecting identity

### Data Sovereignty Architecture

#### Local Processing Priority
- **Embedding Generation**: Local sentence transformer models (no API costs)
- **PII Detection**: On-premises Australian entity recognition
- **Database Operations**: Local PostgreSQL with privacy controls
- **Sentiment Analysis**: Local processing with privacy protection

#### Cloud Integration Controls
- **Schema-Only Transmission**: Database structure without sensitive data
- **Anonymised Content**: PII-cleaned text for LLM processing
- **Audit Trails**: Complete logging of cross-border data flows
- **Provider Independence**: Multi-provider support reducing vendor lock-in

## Security Architecture

### Access Control Framework
- **Read-Only Database Access**: RAG module limited to SELECT operations
- **Credential Protection**: Secure environment variable management
- **Connection Pooling**: Efficient resource management with security controls
- **Permission Validation**: Comprehensive testing of access constraints

### Data Protection Measures
- **Sensitive Data Masking**: Automatic credential and PII masking in logs
- **Error Sanitisation**: Production-safe error messages without information leakage
- **Secure Configuration**: Pydantic-based validation with security controls
- **Audit Integration**: Structured logging for compliance and monitoring

## Testing & Validation

### Comprehensive Test Coverage
- **75/75 Tests Passing**: Complete automated and manual test validation
- **Security Testing**: Access controls, credential protection, and error handling
- **Privacy Testing**: PII detection, anonymisation, and APP compliance
- **Functionality Testing**: All modules tested with real data patterns
- **Integration Testing**: End-to-end workflows with privacy protection

### Australian Context Testing
- **Real Data Compatibility**: Tested with Australian Public Service evaluation structure
- **Entity Recognition**: Validation of Australian business and personal identifier detection
- **Language Patterns**: Optimised for Australian English and government terminology
- **Compliance Validation**: Automated APP compliance checking in test suites

## Deployment Considerations

### Environment Configuration
- **Local Development**: Complete privacy protection with local processing
- **Government Cloud**: Australian data residency with enhanced security
- **Hybrid Deployment**: Flexible architecture supporting various deployment models
- **Scaling Considerations**: Async architecture ready for concurrent processing

### Operational Security
- **Monitoring Integration**: Structured logging compatible with government monitoring systems
- **Backup Procedures**: Privacy-aware backup and recovery processes
- **Access Auditing**: Comprehensive audit trails for regulatory compliance
- **Performance Monitoring**: Resource usage tracking with privacy protection

## Future Enhancements

### Planned Developments
- **Advanced Analytics**: Enhanced sentiment analysis with Australian context
- **Model Improvements**: Continued optimisation of local embedding models
- **Integration Expansion**: Additional government data source compatibility
- **Performance Optimisation**: Scaling improvements for large dataset processing

### Compliance Evolution
- **Regulatory Updates**: Continuous alignment with evolving privacy requirements
- **International Standards**: Integration with global privacy frameworks
- **Enhanced Auditing**: Advanced compliance monitoring and reporting capabilities
- **Security Hardening**: Ongoing security improvements and threat mitigation

---
**Last Updated**: 17 June 2025
