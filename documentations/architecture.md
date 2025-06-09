# MVP 1

## 1. Data
- Mock data is created with the help of Google Gemini 2.5 Pro.
- Mock data are stored in google sheets.

## 1.1 DB Schema
### User Table:
- user_id
- user_level (Level 1-6, Exec Level 1-2)
- agency (Australian Public Service agencies)

### Learning Content Table:
- name
- content_id (course id are not unique to each course, a 'Video' and 'Course' learning content can share the same content id)
- content_type (Live Learning, Video, Course)
- target_level (learning content's intended target audience)
- governing_bodies (learning content's governing bodies)
- surrogate_key (from natural key of content_id and content_type)

### Attendance Table:
- user_id
- learning_content.surrgorate_key
- date_start
- date_end
- status (Enrolled, In-progress, Completed, Withdrew)

### Learning Content Evaluation Table:
- response_id
- course_end_date
- learning_content.surrogate_key
- user_id
- course_delivery_type (single choice)
    -  In-person
    -  Virtual
    -  Blended
- agency (an Australian Public Service agency)
- attendance_motivation (multi choice)
    1. To improve my performance in my current role
    2. To develop for a future role or career progression
    3. Recommendation from others who have attended
    4. My supervisor/manager encouraged me to attend
    5. My attendance was mandatory
    6. This course is a part of a broader learning pathway or program I am participating in
- positive_learning_experience (likert scale)
- effective_use_of_time (likert scale)
- relevant_to_work (likert scale)
- did_experience_issue (multi choice)
    1. Technical difficulties negatively impacted my participation
    2. Learning resources, such as workbooks or slides, were poorly designed or inconsistent with each other
    3. Some of the course content was factually incorrect or out of date
    4. One or more elements of the course did not meet my accessibility needs and no suitable alternative was provided
    5. None of the above
- did_experience_issue_detail (free text - if not choice #5 is selected)
- facilitator_skills (multi choice)
    1. Displayed strong knowledge of the subject matter
    2. Communicated clearly
    3. Encouraged participation and discussion
    4. Provided clear responses to learner questions
    5. Managed time and pacing
    6. Used examples that are relevant to my context
    7. None of the above
- had_guest_speakers
    - Yes
    - No
- guest_contribution (multi choice - if yes to guest speakers)
    1. Strengthened my understanding of they key concepts
    2. Enhanced my understanding of how learning is relevant to my work context
    3. Provided insights into the challenges and barriers I may face
    4. Gave me confidence I will be able to successfully apply the learning
    5. Brought specialist knowledge or expertise
    6. The contributions of the guest speakers or presenters did not enhance my learning
- knowledge_level_prior
    1. Novice
    2. Beginner
    3. Early Practitioner
    4. Experienced Practitioner
    5. Expert
- course_aaplication (multi choice)
    1. During the course I had the opportunity to practice new skills
    2. The course has prompted me to reflect in how I will do things differently
    3. I have taken away useful tools or resources
    4. I have increased my confidence in my ability to use what I have learned
    5. I can put what I have learned into my own words
    6. I have a plan on how I will apply what I have learned
    7. I am still unsure how to apply what I have learned
    8. Other
- course_application_other (free text - if 'other' is selected)
- course_application_timeframe (single choice)
    1. Immediately
    2. Within the next month
    3. Within the next 1-3 months
    4. Longer term
    5. I am not sure
    6. I do not intent to apply anything covered in this course
    7. I will not have the opportunity to apply anything covered in this course
- general_feedback (free text)


## 2. Local setup
- Python virtual environment is used to contain dependencies
- Dependencies are stored in requirements.txt
- Docker compose is used for services, currently it has these services:
    - Database: pgvector
- PostgreSQL for database client (VS Code extension)

## 3. Sentiment Analysis

### Overview
The `src/sentiment-analysis` module provides an end-to-end pipeline for analysing free-text survey responses using a locally hosted RoBERTa model. It comprises configurable components for loading data, performing sentiment analysis, and persisting results.

### Module Structure
- **config.py**  
  Centralises configuration: model name (`MODEL_NAME`), database URI (`DATABASE_URI`), target table name, and list of free-text columns.
- **analyser.py**  
  Defines `SentimentAnalyser`, which initialises the Hugging Face tokenizer and model and exposes `analyse(text: str) -> dict` to return probability scores for negative, neutral and positive sentiment.
- **db_operations.py**  
  Defines `DBOperations` for interacting with the database. Establishes a connection using credentials from `config.py` and provides `write_sentiment(response_id: int, column: str, scores: dict)` to upsert sentiment results into the sentiment table.
- **data_processor.py**  
  Defines `DataProcessor` which orchestrates fetching evaluation rows, iterating over configured free-text columns, invoking `SentimentAnalyser.analyse()`, and writing results via `DBOperations`.
- **runner.py**  
  Script entry point. Ensures the sentiment table exists (via `src/db/create_sentiment_table.py`), parses any CLI arguments, initialises the above components, and calls `DataProcessor.process_all()`.

### Workflow
1. **Initialise Environment**  
   - Create the sentiment table:  
     ```bash
     python src/db/create_sentiment_table.py
     ```  
   - Install Python dependencies:  
     ```bash
     pip install -r requirements.txt
     ```  

2. **Run Sentiment Pipeline**  
   ```bash
   python src/sentiment-analysis/runner.py
   ```  

3. **Execution Steps**  
   - **Sentiment Table Setup**: `runner.py` calls the table creation script to guarantee the target table exists.  
   - **Component Initialisation**: Instances of `SentimentAnalyser`, `DBOperations`, and `DataProcessor` are created.  
   - **Data Loading**: `DataProcessor` fetches rows from the evaluation table for all free-text columns defined in `config.py`.  
   - **Sentiment Analysis**: For each row and column, `SentimentAnalyser.analyse()` produces a dict of probability scores.  
   - **Result Persistence**: `DBOperations.write_sentiment()` upserts results (linked by `response_id` and `column`) into the sentiment table.  
   - **Logging & Error Handling**: Each step is logged. Errors in individual records are caught and do not interrupt the pipeline.

### Free-Text Columns
The target columns are defined in `config.py` under `FREE_TEXT_COLUMNS`, typically including:  
- `did_experience_issue_detail`  
- `course_application_other`  
- `general_feedback`

### Data Flow
- **Source**: Free-text fields stored in the evaluation table.  
- **Processing**: Local inference with Hugging Face RoBERTa.  
- **Storage**: Sentiment scores stored in a dedicated sentiment table; raw text is not persisted.

### Data Governance & Security
- Analysis occurs entirely within the local environment.  
- Only numerical sentiment scores are stored.  
- Database connections use secure credentials from environment variables.  
- Transactions ensure atomic writes and consistency.

---

## 4. Hybrid RAG Module

### Overview
The `src/rag` module implements a sophisticated hybrid Retrieval-Augmented Generation system that combines Text-to-SQL capabilities with vector search over unstructured data. This system enables natural language queries across both structured relational data and free-text content, with intelligent routing to determine the optimal retrieval strategy.

### Architectural Foundation
Based on **Architecture 2: Hybrid Approach – SQL Execution with Semantic Search over Serialised Data**, this module provides:

- **Text-to-SQL Engine**: Converts natural language queries into secure, validated SQL for structured data retrieval
- **Vector Search Engine**: Semantic search over embedded textual content using pgvector
- **Intelligent Query Router**: LangGraph-based orchestration to route queries to appropriate engines or hybrid approaches
- **Data Privacy Layer**: PII detection and anonymization using Presidio before LLM processing
- **Security Framework**: Read-only database access, SQL validation, and secure credential management

### Module Structure (Target Architecture)
```
src/rag/
├── config.py                    # Configuration management
├── core/
│   ├── __init__.py
│   ├── router.py                # LangGraph query routing orchestration
│   ├── text_to_sql/
│   │   ├── __init__.py
│   │   ├── schema_manager.py    # Database schema understanding
│   │   ├── sql_generator.py     # LLM-based SQL generation
│   │   ├── sql_validator.py     # SQL security validation
│   │   └── sql_executor.py      # Safe SQL execution
│   ├── vector_search/
│   │   ├── __init__.py
│   │   ├── embedder.py          # Text embedding generation
│   │   ├── indexer.py           # Vector indexing management
│   │   ├── retriever.py         # Semantic retrieval
│   │   └── chunk_processor.py   # Text chunking strategies
│   └── privacy/
│       ├── __init__.py
│       ├── pii_detector.py      # Presidio-based PII detection
│       ├── anonymizer.py        # Data anonymization
│       └── access_control.py    # Database access management
├── data/
│   ├── __init__.py
│   ├── embeddings_manager.py    # Vector storage operations
│   ├── schema_embedder.py       # Schema metadata embedding
│   └── content_processor.py     # Data ingestion and serialization
├── api/
│   ├── __init__.py
│   ├── query_interface.py       # Main query processing interface
│   └── response_formatter.py    # Output formatting
├── utils/
│   ├── __init__.py
│   ├── db_utils.py             # Database utilities
│   ├── llm_utils.py            # LLM interaction utilities
│   └── logging_utils.py        # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_router.py
│   ├── test_text_to_sql/
│   ├── test_vector_search/
│   ├── test_privacy/
│   └── fixtures/
└── runner.py                    # Terminal application entry point
```

---

## 5. RAG Integration: Phased Implementation Plan

### Phase 1: Minimal Text-to-SQL MVP (Weeks 1-2)

#### Objectives
- Create a **narrow, functional Text-to-SQL slice** answering basic questions about course evaluations and attendance
- Establish mandatory database security with read-only PostgreSQL role
- Build minimal terminal application using LangChain SQLDatabaseToolkit and LangGraph
- Focus on **one specific query type**: "Show me attendance statistics by [filter]"

#### Key Tasks

**1.1 Database Security Setup (NON-NEGOTIABLE)**
- **Create dedicated read-only PostgreSQL role**: `rag_user_readonly`
  ```sql
  CREATE ROLE rag_user_readonly WITH LOGIN PASSWORD 'secure_password';
  GRANT CONNECT ON DATABASE your_db TO rag_user_readonly;
  GRANT USAGE ON SCHEMA public TO rag_user_readonly;
  GRANT SELECT ON attendance, users, learning_content TO rag_user_readonly;
  -- Explicitly NO INSERT, UPDATE, DELETE, CREATE permissions
  ```
- Document role permissions and security constraints
- Test role restrictions to ensure no write access

**1.2 Pydantic Configuration Management**
- Create `src/rag/config/settings.py` using Pydantic:
  ```python
  from pydantic import BaseSettings
  
  class RAGSettings(BaseSettings):
      database_url: str
      llm_api_key: str  # OpenAI or chosen provider
      llm_model_name: str = "gpt-3.5-turbo"
      max_query_results: int = 100
      
      class Config:
          env_file = ".env"
  ```
- Environment variable management for DATABASE_URL, LLM API keys
- Typed configuration with validation

**1.3 Initial src/rag Module Structure**
```
src/rag/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic-based configuration
├── core/
│   ├── __init__.py
│   └── sql_tool.py              # LangChain SQL tool wrapper
├── interfaces/
│   ├── __init__.py
│   └── terminal_app.py          # Simple CLI interface
├── tests/
│   ├── __init__.py
│   └── test_sql_tool.py         # Basic unit tests
└── runner.py                    # Main entry point
```

**1.4 Schema Provision for LLM (Manual MVP Approach)**
- Create hardcoded schema context string for attendance/users/learning_content tables:
  ```python
  SCHEMA_CONTEXT = """
  Tables:
  - attendance: user_id, learning_content_surrogate_key, date_effective, status
  - users: user_id, user_level, agency  
  - learning_content: surrogate_key, name, content_type, target_level
  
  Relationships:
  - attendance.user_id -> users.user_id
  - attendance.learning_content_surrogate_key -> learning_content.surrogate_key
  """
  ```
- Use LangChain's SQLDatabase utility for programmatic schema retrieval (prepare for Phase 2)
- Document the need for automated schema extraction in future phases

**1.5 Minimal LangGraph Text-to-SQL Implementation**
- Implement `src/rag/core/sql_tool.py` using LangChain SQLDatabaseToolkit:
  - Use `QuerySQLDatabaseTool` for SQL execution
  - Use `QuerySQLCheckerTool` for basic validation
  - Wrap in a simple LangGraph node (single-node graph for MVP)
- Create basic prompt template for attendance-related queries
- Focus on simple aggregations: COUNT, GROUP BY, basic JOINs

**1.6 Terminal MVP Application**
- Build `src/rag/interfaces/terminal_app.py`:
  - Simple input loop for natural language queries
  - Integration with LangGraph SQL workflow
  - Basic error handling and result formatting
- Create `src/rag/runner.py` as entry point
- Target query types:
  - "How many users completed courses in each agency?"
  - "Show attendance status breakdown by user level"
  - "Which courses have the highest enrollment?"

#### Components to be Developed/Modified

**New Components:**
- Complete `src/rag` module foundation
- Schema embedding infrastructure in pgvector
- Read-only database role and access controls
- LangChain integration for SQL generation

**Modified Components:**
- Extend `docker-compose.yml` to include LLM service containers (if using local models)
- Update `requirements.txt` with LangChain, Presidio, and related dependencies
- Create new environment variables in `.env` for LLM configuration

#### Architecture Documentation Updates
- Document new module architecture and data flow
- Add Text-to-SQL processing pipeline diagrams
- Include database security model and read-only access patterns
- Document query types and capabilities for MVP

#### Data Privacy/Governance Steps
- Implement database role-based access control (RBAC)
- Create audit logging for all SQL queries generated and executed
- Establish query validation and sanitization procedures
- Document compliance with Australian Privacy Principles (APPs) for structured data access

#### Testing/Validation
- Unit tests for schema management and SQL generation
- Integration tests for end-to-end Text-to-SQL pipeline
- Security tests for SQL injection prevention
- Performance benchmarks for query processing speed
- Manual testing via terminal application with representative questions

**Example Terminal Interactions:**
```bash
python src/rag/runner.py
> "Which courses had the highest completion rates in 2024?"
> "What are the most common issues reported in course evaluations?"
> "Show me attendance patterns by user level"
```

#### Success Criteria
- Terminal application successfully processes natural language queries about structured data
- All generated SQL queries pass security validation
- Response time under 5 seconds for typical queries
- 90%+ accuracy on predefined test question set
- Zero SQL injection vulnerabilities in security audit

---

### Phase 2: Vector Search & Content Embedding

#### Objectives
- Implement semantic search capabilities over free-text content
- Create comprehensive content ingestion and embedding pipeline
- Integrate vector search with existing sentiment analysis data
- Build hybrid retrieval combining structured and unstructured data

#### Key Tasks

**2.1 Vector Infrastructure Setup**
- Create dedicated pgvector tables for content embeddings:
  - `rag_embeddings`: chunked content with metadata
  - `rag_schema_embeddings`: schema component embeddings
  - Indexes and metadata tables for embedding management
- Implement `src/rag/data/embeddings_manager.py` for vector operations

**2.2 Content Processing Pipeline**
- Develop `src/rag/core/vector_search/chunk_processor.py`:
  - Semantic chunking strategies for evaluation free-text responses
  - Metadata preservation (response_id, column, sentiment scores, user_level, agency)
  - Integration with existing sentiment analysis results
  - Content deduplication and filtering

**2.3 Embedding Generation System**
- Implement `src/rag/core/vector_search/embedder.py`:
  - Support for multiple embedding models (OpenAI, Sentence-BERT, local models)
  - Batch processing for efficiency
  - Embedding dimension consistency management
  - Integration with content processing pipeline

**2.4 Semantic Retrieval Engine**
- Create `src/rag/core/vector_search/retriever.py`:
  - Similarity search with configurable distance metrics
  - Metadata filtering (by sentiment, user_level, agency, course_type)
  - Hybrid ranking combining similarity scores and metadata relevance
  - Result reranking and diversification

**2.5 Content Indexing Management**
- Develop `src/rag/core/vector_search/indexer.py`:
  - Incremental indexing for new evaluation responses
  - Index optimization and maintenance
  - Version control for embedding models
  - Performance monitoring and tuning

**2.6 Data Serialization Strategy**
- Implement `src/rag/data/content_processor.py`:
  - Row serialization for evaluation responses with context
  - Relationship descriptions (user-course-evaluation connections)
  - Structured data augmentation for vector search
  - Integration with existing database schema

#### Components to be Developed/Modified

**New Components:**
- Complete vector search infrastructure
- Content embedding and indexing pipeline
- Semantic retrieval with metadata filtering
- Integration layer between structured and unstructured data

**Modified Components:**
- Extend database schema with embedding tables
- Update sentiment analysis pipeline to support RAG integration
- Enhance `runner.py` to support vector search queries

#### Architecture Documentation Updates
- Document vector search architecture and embedding strategies
- Add content processing pipeline diagrams
- Include metadata filtering and hybrid ranking explanations
- Document integration patterns with existing sentiment analysis

#### Data Privacy/Governance Steps
- Implement content anonymization before embedding generation
- Create embedding versioning and lineage tracking
- Establish data retention policies for vector embeddings
- Document PII handling in unstructured content processing

#### Testing/Validation
- Unit tests for embedding generation and vector operations
- Integration tests for content processing pipeline
- Performance tests for similarity search and retrieval
- Quality tests for embedding accuracy and relevance
- Comparative evaluation against keyword-based search

**Example Queries Enabled:**
```bash
> "Find courses with negative feedback about facilitator communication"
> "What do users say about technical difficulties in virtual courses?"
> "Show similar feedback across different agencies"
```

#### Success Criteria
- Vector search returns relevant results with >80% precision@10
- Content processing pipeline handles full evaluation dataset
- Embedding generation completes in <1 hour for complete dataset
- Metadata filtering provides accurate scope control
- Integration with sentiment data enhances result relevance

---

### Phase 3: Intelligent Query Routing & Hybrid Integration

#### Objectives
- Implement LangGraph-based query orchestration and routing
- Create intelligent decision-making for query type classification
- Build hybrid approaches combining Text-to-SQL and vector search
- Establish comprehensive query processing pipeline

#### Key Tasks

**3.1 Query Classification System**
- Develop query intent classification:
  - Structured data queries (aggregations, filters, joins)
  - Unstructured content queries (sentiment analysis, thematic search)
  - Hybrid queries requiring both approaches
  - Complex analytical queries needing multi-step processing

**3.2 LangGraph Router Implementation**
- Create `src/rag/core/router.py` using LangGraph:
  - Multi-agent workflow for query processing
  - Conditional routing based on query classification
  - State management for complex multi-step queries
  - Error handling and fallback strategies

**3.3 Hybrid Query Processing**
- Implement hybrid processing workflows:
  - Structured data retrieval → context for semantic search
  - Semantic search → filters for SQL queries
  - Parallel processing with result fusion
  - Cross-reference validation between structured and unstructured results

**3.4 Result Integration and Ranking**
- Develop result fusion algorithms:
  - Reciprocal Rank Fusion (RRF) for combining ranked lists
  - Confidence scoring for different result types
  - Relevance-based result ordering
  - Duplicate detection across result types

**3.5 Query Interface Enhancement**
- Upgrade `src/rag/api/query_interface.py`:
  - Support for complex multi-part queries
  - Context preservation across query sessions
  - Query refinement and clarification prompts
  - Result explanation and source attribution

**3.6 Response Generation System**
- Implement `src/rag/api/response_formatter.py`:
  - LLM-based response synthesis using retrieved context
  - Citation and source linking
  - Different output formats (summary, detailed, tabular)
  - Confidence indicators and uncertainty handling

#### Components to be Developed/Modified

**New Components:**
- LangGraph-based query router and orchestration
- Hybrid query processing workflows
- Result fusion and ranking algorithms
- Enhanced query interface with session management

**Modified Components:**
- Integrate router with existing Text-to-SQL and vector search engines
- Enhance terminal application with advanced query capabilities
- Update configuration management for routing parameters

#### Architecture Documentation Updates
- Document LangGraph workflow architecture and decision trees
- Add hybrid query processing flowcharts
- Include result fusion algorithms and ranking strategies
- Document query session management and context preservation

#### Data Privacy/Governance Steps
- Implement query audit logging with classification tracking
- Create result provenance and source attribution
- Establish query complexity limits and resource controls
- Document cross-component data flow and privacy boundaries

#### Testing/Validation
- Unit tests for query classification and routing logic
- Integration tests for hybrid query workflows
- End-to-end tests for complex analytical queries
- Performance tests for multi-step query processing
- User acceptance testing with domain experts

**Example Advanced Queries:**
```bash
> "Which courses with low satisfaction scores also have negative sentiment about workload?"
> "Compare completion rates between agencies for courses with positive facilitator feedback"
> "Find patterns in technical issues across different delivery types"
```

#### Success Criteria
- Query router correctly classifies 95%+ of test queries
- Hybrid queries return comprehensive and accurate results
- Response generation includes proper source attribution
- Processing time remains under 10 seconds for complex queries
- User satisfaction rating >4.0/5.0 for result relevance and clarity

---

### Phase 4: Privacy Enhancement & Security Hardening

#### Objectives
- Implement comprehensive PII detection and anonymization
- Enhance security framework with advanced threat protection
- Establish compliance with Australian Privacy Principles (APPs)
- Create privacy-preserving analytics capabilities

#### Key Tasks

**4.1 PII Detection and Anonymization**
- Implement `src/rag/core/privacy/pii_detector.py` using Presidio:
  - Australian-specific PII patterns (TFN, Medicare numbers, addresses)
  - Custom entity recognition for APS-specific identifiers
  - Confidence scoring for PII detection
  - Real-time detection in query inputs and results

**4.2 Data Anonymization Framework**
- Develop `src/rag/core/privacy/anonymizer.py`:
  - Multiple anonymization strategies (masking, pseudonymization, generalization)
  - Reversible anonymization for authorized access
  - Consistency preservation across related records
  - Integration with embedding generation pipeline

**4.3 Access Control Enhancement**
- Upgrade `src/rag/core/privacy/access_control.py`:
  - Role-based access control (RBAC) integration
  - Attribute-based access control (ABAC) for fine-grained permissions
  - Query result filtering based on user permissions
  - Audit trail for all access attempts and data modifications

**4.4 Security Monitoring and Compliance**
- Implement security monitoring:
  - Anomaly detection for unusual query patterns
  - Rate limiting and abuse prevention
  - Encrypted data transmission and storage
  - Compliance reporting for APP requirements

**4.5 Privacy-Preserving Analytics**
- Create privacy-preserving analysis capabilities:
  - Differential privacy for aggregate statistics
  - k-anonymity preservation in result sets
  - Secure multi-party computation for cross-agency analysis
  - Privacy budget management for repeated queries

#### Components to be Developed/Modified

**New Components:**
- Complete privacy and security framework
- PII detection and anonymization pipeline
- Advanced access control and monitoring systems
- Compliance reporting and audit tools

**Modified Components:**
- Integrate privacy controls into all query processing components
- Update database access patterns with enhanced security
- Modify result formatting to include privacy indicators

#### Architecture Documentation Updates
- Document comprehensive privacy and security architecture
- Add APP compliance mapping and controls
- Include threat model and security risk assessment
- Document privacy-preserving analytics capabilities

#### Data Privacy/Governance Steps
- Complete Australian Privacy Principles (APP) compliance assessment
- Implement data minimization and purpose limitation controls
- Create privacy impact assessment documentation
- Establish incident response procedures for privacy breaches

#### Testing/Validation
- Security penetration testing for all components
- Privacy compliance testing against APP requirements
- Performance impact assessment for privacy controls
- User experience testing with privacy features enabled
- Compliance audit preparation and documentation

#### Success Criteria
- Zero PII exposure in anonymized query results
- 100% compliance with applicable APP requirements
- Security controls pass penetration testing
- Privacy features maintain system performance within 20% impact
- Compliance documentation ready for audit

---

### Phase 5: API Development & Production Readiness

#### Objectives
- Develop FastAPI interface for web application integration
- Implement production-grade monitoring and logging
- Create comprehensive documentation and deployment guides
- Establish performance optimization and scalability framework

#### Key Tasks

**5.1 FastAPI Interface Development**
- Create RESTful API endpoints:
  - `/query` - Main query processing endpoint
  - `/explain` - Query explanation and debugging
  - `/health` - System health and status monitoring
  - `/admin` - Administrative functions and configuration
- Implement async processing for concurrent queries
- Create API documentation with OpenAPI/Swagger

**5.2 Production Monitoring and Observability**
- Implement comprehensive monitoring:
  - Query performance metrics and latency tracking
  - System resource utilization monitoring
  - Error rate and failure analysis
  - User behavior and query pattern analytics
- Create alerting for critical system events
- Implement distributed tracing for complex queries

**5.3 Scalability and Performance Optimization**
- Implement caching strategies:
  - Query result caching with TTL management
  - Embedding cache for frequently accessed content
  - Schema metadata caching with invalidation
- Create connection pooling and resource management
- Implement load balancing for multiple instances

**5.4 Documentation and Developer Experience**
- Create comprehensive documentation:
  - API reference with examples
  - Developer integration guides
  - Configuration and deployment documentation
  - Troubleshooting and FAQ sections
- Implement SDK/client libraries for common platforms
- Create interactive examples and tutorials

**5.5 Deployment and Operations**
- Create Docker containerization for all components
- Implement CI/CD pipeline with automated testing
- Create deployment scripts and infrastructure as code
- Establish backup and disaster recovery procedures

#### Components to be Developed/Modified

**New Components:**
- Complete FastAPI application with production features
- Monitoring and observability infrastructure
- Caching and performance optimization systems
- Deployment and operations tooling

**Modified Components:**
- Adapt terminal application to use API backend
- Update configuration management for production settings
- Enhance logging and monitoring across all components

#### Architecture Documentation Updates
- Document production architecture and deployment patterns
- Add API reference and integration guides
- Include monitoring and observability setup
- Document scalability patterns and performance tuning

#### Data Privacy/Governance Steps
- Implement API authentication and authorization
- Create API access logging and audit trails
- Establish rate limiting and abuse prevention
- Document API security and privacy controls

#### Testing/Validation
- Load testing for API performance and scalability
- Integration testing for all API endpoints
- Deployment testing in staging environment
- User acceptance testing with web interface
- Production readiness assessment and checklist

**Example API Usage:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main issues in virtual courses?", "format": "summary"}'
```

#### Success Criteria
- API handles 100+ concurrent queries with <2s response time
- Monitoring provides comprehensive system visibility
- Documentation enables self-service developer adoption
- Deployment process is automated and repeatable
- System passes production readiness review

---

## 6. Architect's Notes and Considerations

### Strategic Recommendations

**Phased Implementation Benefits:**
- **Risk Mitigation**: Each phase builds on proven components, reducing integration risk
- **Early Value**: Text-to-SQL MVP provides immediate utility for structured data queries
- **Iterative Improvement**: User feedback from each phase informs subsequent development
- **Resource Management**: Phased approach allows for team skill development and resource allocation

**Architecture Decision Rationale:**
- **Architecture 2 Selection**: Hybrid approach provides maximum flexibility for diverse query types while maintaining reasonable complexity
- **LangGraph Integration**: Provides robust orchestration capabilities with visual workflow design and debugging
- **pgvector Choice**: Leverages existing PostgreSQL infrastructure while providing enterprise-grade vector capabilities
- **Privacy-First Design**: Australian compliance requirements necessitate privacy considerations throughout the architecture

### Technical Challenges and Mitigations

**Challenge 1: Query Classification Accuracy**
- **Risk**: Misrouting queries leads to poor results or system failures
- **Mitigation**: Comprehensive training dataset, fallback routing, and continuous learning from user feedback
- **Monitoring**: Query classification confidence scores and manual review queues

**Challenge 2: Embedding Model Selection and Evolution**
- **Risk**: Model changes break existing embeddings and require reindexing
- **Mitigation**: Embedding versioning, gradual migration strategies, and model performance monitoring
- **Strategy**: Start with proven models (OpenAI/Sentence-BERT) and plan for model upgrades

**Challenge 3: Performance at Scale**
- **Risk**: System performance degrades with large datasets and concurrent users
- **Mitigation**: Caching strategies, connection pooling, horizontal scaling design
- **Monitoring**: Performance baselines and automated scaling triggers

**Challenge 4: Data Privacy Compliance**
- **Risk**: Inadvertent PII exposure in query results or logs
- **Mitigation**: Multi-layer privacy controls, comprehensive testing, and audit procedures
- **Governance**: Regular compliance reviews and privacy impact assessments

### Integration with Existing Architecture

**Sentiment Analysis Synergy:**
- RAG module leverages existing sentiment scores for enhanced context
- Sentiment data provides valuable metadata for result ranking and filtering
- Shared database infrastructure reduces complexity and improves performance

**Database Schema Evolution:**
- New RAG tables integrate cleanly with existing schema
- Foreign key relationships maintain data integrity
- Minimal impact on existing sentiment analysis workflows

**Development Workflow Alignment:**
- Similar modular design patterns as sentiment analysis module
- Consistent testing and documentation standards
- Shared utility functions and database connection management

### Future Enhancement Opportunities

**Advanced Analytics Integration:**
- Time-series analysis for trend identification
- Predictive modeling for course success factors
- Network analysis for user behavior patterns

**Multi-Modal Capabilities:**
- Document ingestion and analysis beyond text
- Integration with course materials and resources
- Voice query processing for accessibility

**Federated Learning:**
- Cross-agency knowledge sharing while preserving privacy
- Distributed model training and inference
- Secure multi-party computation for collaborative analytics

### Success Metrics and KPIs

**Technical Performance:**
- Query response time: <5s for 95% of queries
- System availability: >99.5% uptime
- Accuracy: >90% relevance for returned results
- Scalability: Support 1000+ concurrent users

**User Experience:**
- User satisfaction: >4.0/5.0 rating
- Query success rate: >95% of queries return useful results
- Adoption rate: >80% of target users active monthly
- Support burden: <5% of queries require manual intervention

**Business Impact:**
- Decision-making speed: 50% reduction in time to insights
- Data utilization: 3x increase in survey data usage
- Cost efficiency: 30% reduction in manual analysis efforts
- Compliance: 100% adherence to privacy requirements
