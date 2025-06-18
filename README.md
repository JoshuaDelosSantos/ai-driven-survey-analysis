# AI-Enhanced Survey Analysis

## 1. Overview
This project aims to augment our survey-analysis workflow by applying modern AI techniques to free-text feedback. Instead of manually reviewing thousands of open-ended comments, stakeholders will be able to query an AI system and receive context-aware summaries, sentiment trends, and actionable insights.

**Status**: Phase 2 Complete - Production-ready RAG infrastructure with Australian data governance compliance established.

## 2. Problem Statement
- We collect a variety of data (course information, user profiles, attendance, completions, and course evaluation from surveys), but most of our qualitative insights are buried in multiple Excel files.
- Manually reading and coding hundreds or thousands of free-text responses is time-consuming, error-prone, and prone to human bias.
- Without semantic context, business users may overlook patterns or misinterpret feedback.

## 3. Implemented Solution

### Phase 2: RAG Infrastructure & Data Governance (✅ Complete)

#### 3.1 Australian PII Detection & Anonymisation
- **Comprehensive Protection**: Mandatory detection and anonymisation of Australian entities (ABN, ACN, TFN, Medicare numbers)
- **Privacy-First Processing**: All text processing occurs after PII anonymisation using Microsoft Presidio
- **Audit Compliance**: Complete logging with privacy-protected analytics for governance requirements

#### 3.2 Vector Embeddings Infrastructure  
- **pgVector Database**: Production-ready PostgreSQL with vector similarity search capabilities
- **Multi-Provider Embeddings**: Support for OpenAI and Sentence Transformers with configurable switching
- **Async Architecture**: Non-blocking operations designed for high-concurrency environments

#### 3.3 Unified Text Processing Pipeline
- **Five-Stage Processing**: Extract → Anonymise → Analyse → Chunk → Store with mandatory privacy controls
- **Real Sentiment Analysis**: Integrated transformer models (cardiffnlp/twitter-roberta-base-sentiment)
- **Batch Processing**: Efficient handling of large evaluation datasets with configurable batch sizes

#### 3.4 Vector Search Tool & LangChain Integration
- **LangChain Compatibility**: Native `BaseTool` implementation for agent orchestration
- **Advanced Metadata Filtering**: Search by user level, agency, sentiment scores, delivery type
- **Privacy-Protected Queries**: Automatic query anonymisation before embedding generation
- **Performance Monitoring**: Built-in metrics with processing time analysis

#### 3.5 Comprehensive Testing Framework
- **106+ Tests**: Comprehensive coverage including privacy compliance, integration, and performance testing
- **Real Model Validation**: Testing with actual transformer models and database connections
- **Australian Compliance**: Automated validation of PII detection and privacy-safe operations

## 4. Key Benefits (Implemented)
- **Privacy-First Architecture**: Mandatory Australian PII detection and anonymisation before any processing
- **Semantic Search Capabilities**: Vector similarity search with advanced metadata filtering for precise feedback analysis
- **Real-Time Sentiment Analysis**: Automated sentiment classification using production transformer models
- **LangChain Integration**: Ready for agent-based workflows and intelligent query routing
- **Scalable Infrastructure**: Async-first design with efficient database connection pooling and batch processing
- **Audit Compliance**: Comprehensive logging and monitoring aligned with Australian data governance requirements

## 5. Current Technical Architecture

### 5.1 Implemented Infrastructure
1. **Database (PostgreSQL + pgvector) ✅**  
   - Production pgvector schema with optimised vector similarity indexing
   - Foreign key integrity with evaluation data and cascade deletion support
   - Efficient metadata filtering using GIN indexes on JSONB fields
   - Read-only database roles for secure search operations

2. **Privacy & Data Governance ✅**  
   - **Australian PII Detection**: Comprehensive detection of ABN, ACN, TFN, Medicare numbers
   - **Mandatory Anonymisation**: All text processing occurs after PII anonymisation
   - **Privacy-Safe Logging**: Automatic masking of detected entities in audit logs
   - **Data Sovereignty**: Local processing with controlled external API usage

3. **Vector Embeddings System ✅**  
   - **Multi-Provider Support**: OpenAI and Sentence Transformers with configurable switching
   - **Async Processing**: Non-blocking embedding generation and storage
   - **Batch Optimisation**: Configurable batch sizes for API efficiency and memory management
   - **Model Versioning**: Explicit embedding model tracking for audit trail and migration support

4. **Sentiment Analysis Pipeline ✅**  
   - **Real Transformer Models**: cardiffnlp/twitter-roberta-base-sentiment integration
   - **Confidence Scoring**: Detailed sentiment scores with positive/negative/neutral classification
   - **Metadata Integration**: Sentiment scores stored with vector embeddings for advanced filtering

5. **Vector Search Tool ✅**  
   - **LangChain Integration**: Native `BaseTool` implementation for agent workflows
   - **Advanced Filtering**: Multi-dimensional search by user level, agency, sentiment, delivery type
   - **Privacy Protection**: Automatic query anonymisation before embedding generation
   - **Performance Monitoring**: Built-in response time tracking and similarity score analysis

### 5.2 Production-Ready Components
- **Data Processing**: Complete ETL pipeline with privacy-first text processing and sentiment analysis
- **Vector Storage**: Optimised pgvector database with efficient similarity search and metadata filtering  
- **Search Interface**: LangChain-compatible async tool with comprehensive privacy controls
- **Testing Framework**: 106+ tests covering functionality, privacy compliance, and performance validation
- **Configuration Management**: Environment-based settings with secure credential handling

### 5.3 Usage Examples

#### Basic Vector Search
```python
from src.rag.core.vector_search import VectorSearchTool

# Initialize and use vector search tool
tool = VectorSearchTool()
await tool.initialize()

# Basic semantic search with privacy protection
response = await tool.search(
    query="feedback about technical issues",  # Auto-anonymised
    max_results=10,
    similarity_threshold=0.75
)

# Advanced metadata filtering
response = await tool.search(
    query="senior staff feedback on virtual learning",
    filters={
        "user_level": ["Level 5", "Level 6", "Exec Level 1"],
        "agency": "Department of Finance",
        "sentiment": {"type": "negative", "min_score": 0.6}
    }
)
```

#### LangChain Tool Integration
```python
# Use as LangChain tool in agent workflows
result = await tool.ainvoke({
    "query": "What challenges did participants face with course materials?",
    "max_results": 15,
    "filters": {"course_delivery_type": "Virtual"}
})
```

---

## 6. Australian Data Privacy & Governance
We implement a comprehensive, "defence-in-depth" approach aligned with Australian data sovereignty requirements:

### 6.1 Mandatory Privacy Protection
1. **Australian PII Detection**  
   - Comprehensive detection of ABN, ACN, TFN, Medicare numbers and other Australian entities
   - **Zero PII Storage**: Mandatory anonymisation occurs before any processing or storage
   - Automatic masking in all logs and audit trails

2. **Privacy-First Processing Pipeline**  
   - All text processing occurs after PII anonymisation
   - Vector embeddings generated only from anonymised content
   - Sentiment analysis performed on privacy-protected text

3. **Data Sovereignty Controls**  
   - Local processing with controlled external API usage
   - Configurable embedding providers (local Sentence Transformers vs external OpenAI)
   - Australian data residency compliance for all stored content

### 6.2 Access Controls & Security
1. **Database Security**  
   - Read-only roles for search operations with minimal privileges
   - Foreign key constraints ensuring data integrity
   - Encrypted connections and credential management

2. **Audit & Compliance**  
   - Complete operation logging with privacy-protected analytics
   - Performance monitoring without exposing sensitive content
   - Structured audit trail for governance requirements

3. **Data Lifecycle Management**  
   - Foreign key cascade deletion for data consistency
   - Model versioning for embedding migration and audit compliance
   - Privacy-safe error handling and monitoring

### 6.3 Regulatory Alignment
- **Australian Privacy Principles (APP)**: Comprehensive PII protection and data minimisation
- **Data Sovereignty**: Local processing capabilities with controlled external dependencies
- **Audit Requirements**: Complete logging framework with privacy-protected analytics
- **Access Control**: Role-based permissions with least-privilege principles

---

## 7. Development Status & Next Steps

### Phase 2 Completion Status ✅
- [x] **Task 2.1**: Australian PII Detection & Anonymisation
- [x] **Task 2.2**: pgVector Infrastructure & Schema Design  
- [x] **Task 2.3**: Unified Text Processing & Ingestion Pipeline
- [x] **Task 2.4**: Content Processor Integration
- [x] **Task 2.5**: Vector Search Tool & LangChain Integration

### Phase 3 Roadmap
- **LangGraph Agent Development**: Intelligent query routing and conversation management
- **Advanced Analytics**: Search pattern analysis and recommendation engine
- **Performance Optimisation**: Large-scale vector search tuning and caching strategies
- **User Interface**: Stakeholder-friendly dashboard for semantic search and sentiment analysis

### Ready for Production
The RAG infrastructure is now production-ready with:
- ✅ **Comprehensive Testing**: 106+ tests covering all functionality and privacy compliance
- ✅ **Australian Data Governance**: Complete PII protection and audit compliance
- ✅ **Scalable Architecture**: Async-first design with efficient resource management
- ✅ **LangChain Integration**: Ready for agent-based workflows and intelligent automation
