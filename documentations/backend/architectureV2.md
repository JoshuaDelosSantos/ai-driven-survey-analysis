# MVP 1

## 1. Data
- Mock data is created with the help of Google Gemini.
- Mock data are stored in Google Sheets.

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
- learning_content.surrogate_key
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
- positive_learning_experience (likert scale)
- effective_use_of_time (likert scale)
- relevant_to_work (likert scale)
- did_experience_issue (multi choice)
- did_experience_issue_detail (free text - if not choice #5 is selected)
- facilitator_skills (multi choice)
- had_guest_speakers
    - Yes
    - No
- guest_contribution (multi choice - if yes to guest speakers)
- knowledge_level_prior
- course_application (multi choice)
- course_application_other (free text - if 'other' is selected)
- course_application_timeframe (single choice)
- general_feedback (free text)


## 2. Local setup
- Python virtual environment is used to contain dependencies
- Dependencies are stored in requirements.txt
- Docker compose is used for services, currently it has these services:
    - Database: pgvector
- PostgreSQL for database client (VS Code extension)

## 3. Sentiment Analysis as a Reusable Service (Refactored)

### Overview
The `src/sentiment-analysis` module is designed as a **reusable analysis component**, not a standalone pipeline. It provides a core `SentimentAnalyser` class that can be instantiated and used by other parts of the system, such as the RAG ingestion pipeline, to perform on-demand sentiment analysis on free-text data. This design choice eliminates data processing latency between sentiment analysis and RAG indexing.

### Module Structure
- **config.py** Centralises configuration: model name (`MODEL_NAME`), and other relevant settings. Database URI is removed as this module no longer interacts with the DB directly.
- **analyser.py** Defines `SentimentAnalyser`, which initialises the Hugging Face tokenizer and model. It exposes a primary method:
    - `analyse(text: str) -> dict`: Takes a string and returns a dictionary of probability scores for negative, neutral, and positive sentiment. The method is designed to be self-contained, efficient, and easily portable.
- **db_operations.py** This file is **deprecated** and will be removed. The component no longer writes to the database; its results are consumed and persisted by the calling service (e.g., the RAG module).
- **data_processor.py** This file is **deprecated** and will be removed. Orchestration is now handled by the RAG ingestion pipeline.
- **runner.py** This file is **deprecated** and will be removed.

### Data Flow
- **Source**: In-memory text data passed from a calling function (e.g., `content_processor.py` in the RAG module).
- **Processing**: Local inference with a Hugging Face RoBERTa model.
- **Output**: A dictionary of sentiment scores returned directly to the caller. Raw text and results are not persisted by this module.

---

## 4. Hybrid RAG Module

### Overview
The `src/rag` module implements a sophisticated hybrid Retrieval-Augmented Generation system that combines Text-to-SQL capabilities with vector search over unstructured data. This system enables natural language queries across both structured relational data and free-text content, with intelligent routing to determine the optimal retrieval strategy.

### Architectural Foundation
Based on **Hybrid Approach – SQL Execution with Semantic Search over Serialised Data**, this module provides:
- **Text-to-SQL Engine**: Converts natural language queries into secure, validated SQL for structured data retrieval.
- **Vector Search Engine**: Semantic search over embedded textual content using pgvector.
- **Intelligent Query Router**: LangGraph-based orchestration to route queries to appropriate engines or hybrid approaches.
- **Data Privacy Layer**: PII detection and anonymisation using Presidio before LLM processing.
- **Security Framework**: Read-only database access, SQL validation, and secure credential management.

### Module Structure (Target Architecture)
```
src/rag/
├── config.py                    # Configuration management
├── core/
│   ├── __init__.py
│   ├── agent.py                 # (Phase 3) Main ASYNC LangGraph agent
│   ├── router.py                # (Phase 3) LangGraph query routing orchestration
│   ├── text_to_sql/
│   │   ├── __init__.py
│   │   ├── schema_manager.py    # (Phase 1 Refactored) Database schema understanding
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
│   ├── content_processor.py     # Data ingestion and serialization
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
└── runner.py                    # Terminal application entry point
```

---

## 5. RAG Integration: Phased Implementation Plan

### Phase 1: Minimal Text-to-SQL MVP

#### Objectives
- Create a **narrow, functional, and async-ready Text-to-SQL slice** answering basic questions about course evaluations and attendance.
- Establish mandatory database security with a read-only PostgreSQL role.
- Build a minimal terminal application using LangChain and LangGraph, running in an `asyncio` event loop.
- Focus on **one specific query type**: "Show me attendance statistics by [filter]".

#### **Current Progress & Refactoring Needs**
> You have completed Tasks 1.1 - 1.3.
> **Immediate Action Required:** Review your existing `sql_tool.py` and other core components. Ensure that the primary functions are defined with `async def` and use `asyncio`-compatible libraries where I/O is performed. This is crucial for seamless integration in Phase 3 and 4.

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
- **(Refactored) Add Startup Verification**: The application must include a startup check that verifies the permissions of its database role. It must refuse to launch if the role has any write privileges, ensuring the read-only constraint is actively enforced.

**1.2 Pydantic Configuration Management**
- Create `src/rag/config/settings.py` using Pydantic for typed, validated configuration loaded from environment variables.

**1.3 Initial src/rag Module Structure**
- Establish the foundational directory structure for the `src/rag` module.
- **(Refactored) Async-First Design**: All data-accessing or tool-related classes and functions (e.g., in what will become `sql_tool.py`) must be designed with `async` methods from the outset. This is a core architectural principle to ensure future performance and scalability.

**1.4 (Refactored) Dynamic Schema Provision for LLM**
- **Deprecate hardcoded schema context.** A static string is brittle and difficult to maintain.
- **Implement `src/rag/core/text_to_sql/schema_manager.py`**:
  - This module will programmatically connect to the database using `langchain_community.utilities.SQLDatabase`.
  - Its primary function will be to generate a curated, simplified schema description in plain English. This description will include table relationships, definitions of key columns, and examples of valid values, providing an optimized context to the LLM that is far more effective and maintainable than a raw schema dump.

**1.5 Minimal LangGraph Text-to-SQL Implementation**
- Implement `src/rag/core/sql_tool.py` using LangChain's SQL toolkit with **async methods**.
- Use `QuerySQLDatabaseTool` for SQL execution and `QuerySQLCheckerTool` for basic validation, ensuring both are invoked asynchronously.
- Wrap this functionality in a simple, single-node LangGraph graph as the initial workflow.

**1.6 Terminal MVP Application**
- Build `src/rag/interfaces/terminal_app.py` that runs within an `asyncio` event loop (e.g., using `asyncio.run()` in the main `runner.py` script).
- The main input loop should `await` responses from the LangGraph SQL workflow, ensuring the application is non-blocking.
- Target query types:
  - "How many users completed courses in each agency?"
  - "Show attendance status breakdown by user level"
  - "Which courses have the highest enrollment?"

---

### Phase 2: Vector Search Implementation & Unstructured Data Ingestion

#### Objectives
- Implement semantic search capabilities over the three free-text evaluation fields.
- Create a **unified, secure content ingestion pipeline** with mandatory PII anonymisation and integrated sentiment analysis to eliminate data latency.
- Build a testable, async-ready vector search tool for future integration.

#### Key Tasks

**2.1 Data Source Analysis & PII Detection Strategy**
- Document the three target free-text fields and their data characteristics.
- Implement Australian-specific PII detection using Microsoft Presidio in `src/rag/core/privacy/pii_detector.py`.
- **PII anonymisation is a mandatory, non-negotiable step** before any data is sent to an LLM or used for embedding generation.

**2.2 (Refactored) pgVector Infrastructure & Schema Design**
- Create a dedicated pgvector table with an explicit schema that includes versioning for embeddings to support future model upgrades.
  ```sql
  CREATE TABLE rag_embeddings (
      embedding_id SERIAL PRIMARY KEY,
      response_id INTEGER NOT NULL REFERENCES "Learning Content Evaluation"(response_id),
      field_name VARCHAR(50) NOT NULL, -- 'did_experience_issue_detail', 'course_application_other', 'general_feedback'
      chunk_text TEXT NOT NULL,        -- Anonymised text chunk
      chunk_index INTEGER NOT NULL,    -- Chunk position within original text
      embedding VECTOR(1536) NOT NULL, -- e.g., OpenAI ada-002 dimension
      model_version VARCHAR(50) NOT NULL, -- e.g., 'text-embedding-ada-002-v1'
      metadata JSONB,                  -- {user_level, agency, sentiment_scores, course_type}
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(response_id, field_name, chunk_index)
  );
  
  CREATE INDEX ON rag_embeddings USING ivfflat (embedding vector_cosine_ops);
  CREATE INDEX ON rag_embeddings (response_id);
  CREATE INDEX ON rag_embeddings (field_name);
  ```
- Implement `src/rag/data/embeddings_manager.py` for async vector operations.

**2.3 (Refactored) Unified Text Processing & Ingestion Pipeline**
- Develop `src/rag/data/content_processor.py` to orchestrate a unified, sequential ingestion workflow. This design eliminates temporal dependencies between different analysis steps. For each evaluation record, the flow is:
  1.  **Extract**: Get the target free-text fields from the evaluation table.
  2.  **Anonymise**: Process the raw text through the `pii_detector.py` to remove/replace sensitive data.
  3.  **Analyse Sentiment**: For each anonymised text field, make a direct, in-memory call to the `SentimentAnalyser` component to get sentiment scores.
  4.  **Chunk**: Split the anonymised text into sentence-level chunks suitable for embedding.
  5.  **Embed & Store**: Generate embeddings for each chunk and store them in the `rag_embeddings` table along with rich metadata (user context, the calculated sentiment scores) and the embedding model version.

**2.4 Embedding Generation & Storage System**
- Implement `src/rag/core/vector_search/embedder.py` with async batch processing capabilities for efficiency.
- Sentence-BERT model with a support for OpenAI `text-embedding-ada-002`.

**2.5 Vector Search Tool Development**
- Create `src/rag/core/tools/vector_search_tool.py` as a standalone, **async-ready** LangChain tool.
- It must support metadata filtering (e.g., by sentiment, agency) during retrieval, allowing for more targeted searches.

---

### Phase 3: Hybrid Query Routing (LangGraph) & Answer Synthesis

#### Objectives
- Implement the **core async LangGraph agent** as the central intelligence for the architecture.
- Create a **resilient, multi-stage query classification system** to accurately route user intent.
- Build a comprehensive answer synthesis system with an early feedback mechanism.

#### Key Tasks

**3.1 (Refactored) LangGraph Agent Development (ASYNC CORE)**
- Create `src/rag/core/agent.py` as the main **async** LangGraph orchestrator.
- **All nodes and edges must be designed for and support async execution.**
- **Graph Nodes:**
    - `classify_query_node`: Determines the query's initial intent.
    - `clarification_node`: Engages the user if classification confidence is low.
    - `sql_tool_node`: Executes Text-to-SQL tasks.
    - `vector_search_tool_node`: Executes semantic search tasks.
    - `synthesis_node`: Generates a coherent final answer from retrieved context.
    - `error_handling_node`: Manages failures gracefully and provides informative responses.

**3.2 (Refactored) Resilient Query Classification Component**
- Implement a multi-stage process in `src/rag/core/routing/query_classifier.py` to improve accuracy and reduce errors:
  1.  **Rules-Based Pre-filter:** Use simple keyword and regex matching to immediately route obvious queries (e.g., "count", "how many", "average" -> SQL; "what did people say about", "feedback on" -> Vector). This avoids unnecessary LLM calls.
  2.  **LLM-based Classification:** For queries that are not caught by the pre-filter, use an LLM with a prompt optimised for classification (`SQL`, `Vector`, `Hybrid`) and confidence scoring.
  3.  **Confidence-Based Routing:**
      - **High confidence** -> Route directly to the appropriate tool(s).
      - **Low confidence** -> Route to the `clarification_node`.
- The `clarification_node` will present the user with options to resolve the ambiguity (e.g., "Are you looking for a statistical summary or to search through user comments?") before routing to the correct tool.

**3.2.1 Prompt Engineering Patterns for Query Classification**

**(A) Primary Classification Prompt Template:**
```
You are an expert query router for an Australian Public Service learning analytics system. Your task is to classify user queries into one of three categories:

SQL: Queries requiring statistical analysis, aggregations, or structured data retrieval.
VECTOR: Queries requiring semantic search through free-text feedback and comments.  
HYBRID: Queries requiring both statistical context and semantic content analysis.

DOMAIN CONTEXT:
- Users ask about course evaluations, attendance patterns, and learning outcomes.
- Structured data: attendance records, user levels (1-6, Exec 1-2), agencies, course types.
- Unstructured data: general feedback, issue details, course applications.

CLASSIFICATION RULES:
SQL indicators: "how many", "count", "average", "percentage", "breakdown by", "statistics", "numbers", "total".
VECTOR indicators: "what did people say", "feedback about", "experiences with", "opinions on", "comments", "issues mentioned".
HYBRID indicators: "analyze satisfaction", "compare feedback across", "trends in opinions", "sentiment by agency".

RESPONSE FORMAT:
Classification: [SQL|VECTOR|HYBRID]
Confidence: [HIGH|MEDIUM|LOW]
Reasoning: Brief explanation of classification decision

Query: "{user_query}"
```

**(B) Confidence Scoring Guidelines:**
- **HIGH (0.8-1.0)**: Clear keyword matches, unambiguous intent
- **MEDIUM (0.5-0.79)**: Some indicators present, minor ambiguity
- **LOW (0.0-0.49)**: Unclear intent, multiple possible interpretations

**(C) Clarification Prompt Templates:**
```
I need to understand your query better to provide the most accurate answer. 

Your query: "{user_query}"

Please clarify what you're looking for:

A) 📊 Statistical summary or numerical breakdown
B) 💬 Specific feedback, comments, or experiences  
C) 📈 Combined analysis with both numbers and feedback

Type A, B, or C to continue.
```

**(D) Error Recovery Prompts:**
```
I encountered an issue processing your query. Let me try to understand what you need:

Could you rephrase your question using terms like:
- For statistics: "How many...", "What percentage...", "Show me the breakdown..."
- For feedback: "What did people say about...", "Show me comments about..."
- For analysis: "Analyze the relationship between...", "Compare feedback across..."
```

**3.3 Answer Synthesis System & Early Feedback**
- Implement `src/rag/core/synthesis/answer_generator.py` for context aggregation and answer generation.
- **(New) Early Feedback Loop:** The terminal application will be enhanced to ask for simple user feedback (e.g., a thumbs up/down rating) on each generated answer. This data is critical for early-stage evaluation and iteration on prompt quality and router accuracy.

**3.4 Terminal Application Integration (CRITICAL UPDATE)**
- The `terminal_app.py` must be updated to invoke the full `async` LangGraph agent from `agent.py` as its primary entry point. All user queries from the CLI will be routed through this complete agent.

**3.5 LangGraph Error Recovery Scenarios & Node Resilience**

**(A) Node-Level Error Recovery Patterns:**

**classify_query_node Recovery:**
```python
async def classify_query_with_recovery(state: AgentState) -> AgentState:
    try:
        # Primary classification attempt
        classification = await llm_classifier.ainvoke(state["query"])
        return {**state, "classification": classification, "confidence": "HIGH"}
    except (LLMException, TimeoutError) as e:
        # Fallback to rule-based classification
        logger.warning(f"LLM classification failed, using rule-based fallback: {e}")
        fallback_result = rule_based_classifier(state["query"])
        return {**state, "classification": fallback_result, "confidence": "MEDIUM", "fallback_used": True}
    except Exception as e:
        # Last resort: route to clarification
        logger.error(f"Classification completely failed: {e}")
        return {**state, "classification": "CLARIFICATION_NEEDED", "confidence": "LOW", "error": str(e)}
```

**sql_tool_node Recovery:**
```python
async def sql_tool_with_recovery(state: AgentState) -> AgentState:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Attempt SQL generation and execution
            sql_result = await sql_toolkit.ainvoke(state["query"])
            return {**state, "sql_result": sql_result, "success": True}
        except SQLValidationError as e:
            # SQL was invalid, try regeneration with error context
            logger.warning(f"SQL validation failed (attempt {attempt + 1}): {e}")
            enhanced_query = f"{state['query']} [Previous attempt failed: {e.safe_message}]"
            state = {**state, "query": enhanced_query}
        except DatabaseConnectionError as e:
            # Database connectivity issues
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            return {**state, "error": "Database temporarily unavailable", "success": False}
        except Exception as e:
            logger.error(f"Unexpected SQL tool error: {e}")
            return {**state, "error": "Unable to process SQL query", "success": False}
    
    # All retries exhausted
    return {**state, "error": "Could not generate valid SQL after multiple attempts", "success": False}
```

**vector_search_tool_node Recovery:**
```python
async def vector_search_with_recovery(state: AgentState) -> AgentState:
    try:
        # Primary vector search
        search_results = await vector_search_tool.ainvoke(state["query"])
        if not search_results or len(search_results) == 0:
            # No results found, expand search
            expanded_query = await query_expander.ainvoke(state["query"])
            search_results = await vector_search_tool.ainvoke(expanded_query)
        return {**state, "vector_results": search_results, "success": True}
    except EmbeddingServiceError as e:
        # Fallback to keyword search
        logger.warning(f"Embedding service failed, using keyword fallback: {e}")
        keyword_results = await keyword_search_fallback(state["query"])
        return {**state, "vector_results": keyword_results, "success": True, "fallback_used": True}
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return {**state, "error": "Search service temporarily unavailable", "success": False}
```

**(B) Graph-Level Error Recovery:**

**Circuit Breaker Pattern:**
```python
class RAGCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Service temporarily disabled")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e
```

**(C) Graceful Degradation Strategies:**

**Service Degradation Hierarchy:**
1. **Full Service**: SQL + Vector + Synthesis working
2. **Partial Service**: One tool working, acknowledge limitations
3. **Minimal Service**: Rule-based responses only
4. **Maintenance Mode**: Acknowledge system issues, provide contact information

**User Communication Templates:**
```python
DEGRADATION_MESSAGES = {
    "sql_unavailable": "I'm having trouble accessing the database right now. I can search through feedback and comments instead. Would you like me to look for qualitative insights related to your question?",
    "vector_unavailable": "The search service is temporarily down. I can provide statistical summaries and numerical data. Would you like me to focus on quantitative aspects of your query?",
    "partial_results": "I was able to get some information, but not everything you requested. Here's what I found: {partial_results}. The system is experiencing technical difficulties with {failed_component}.",
    "maintenance_mode": "The system is currently undergoing maintenance. Please try again in a few minutes, or contact support at [contact-info] for urgent queries."
}
```

---

## 6. Performance Benchmarking & Quality Metrics Framework

### Phase-Specific Performance Criteria

**Phase 1: Text-to-SQL MVP Benchmarks**

**(A) Response Time Targets:**
- **Database Connection**: < 100ms for initial connection
- **Schema Retrieval**: < 200ms for dynamic schema generation
- **SQL Generation**: < 2 seconds for LLM-based SQL creation
- **SQL Execution**: < 5 seconds for complex analytical queries
- **End-to-End Response**: < 8 seconds total for 95th percentile

**(B) Accuracy Metrics:**
- **SQL Validity Rate**: > 95% of generated SQL should execute without syntax errors
- **Result Relevance**: > 90% of results should be semantically relevant to user queries
- **Schema Understanding**: > 95% accuracy in table/column selection for given queries

**(C) Security Compliance:**
- **Permission Validation**: 100% success rate for read-only constraint verification
- **SQL Injection Prevention**: 0% tolerance for dangerous SQL patterns in generated queries
- **Query Validation**: 100% of SQL must pass through validation pipeline

**Phase 2: Vector Search Implementation Benchmarks**

**(A) Indexing Performance:**
- **PII Detection Speed**: < 500ms per text field (average 200 characters)
- **Sentiment Analysis**: < 100ms per text chunk using local model
- **Embedding Generation**: < 2 seconds per batch of 50 text chunks
- **Vector Storage**: < 1 second for batch insertion of 100 embeddings

**(B) Search Performance:**
- **Vector Query Response**: < 1 second for semantic similarity search
- **Metadata Filtering**: < 500ms additional overhead for complex filters
- **Result Ranking**: < 200ms for re-ranking top 20 results

**(C) Quality Metrics:**
- **Retrieval Accuracy**: > 85% relevant results in top 5 for domain queries
- **PII Leak Rate**: 0% tolerance for PII in anonymized text
- **Sentiment Accuracy**: > 80% agreement with human evaluators on sentiment polarity

**Phase 3: Hybrid Query Routing Benchmarks**

**(A) Classification Performance:**
- **Query Classification Speed**: < 1 second for LLM-based classification
- **Rule-Based Pre-filter**: < 50ms for keyword-based routing
- **Classification Accuracy**: > 90% correct routing decisions
- **Confidence Calibration**: Confidence scores should correlate with actual accuracy

**(B) End-to-End Response Times:**
- **Simple SQL Queries**: < 10 seconds total response time
- **Vector Search Queries**: < 8 seconds total response time  
- **Hybrid Queries**: < 15 seconds total response time
- **Clarification Loops**: < 3 seconds for clarification presentation

**(C) User Experience Metrics:**
- **Task Completion Rate**: > 85% of user queries result in satisfactory answers
- **Clarification Rate**: < 20% of queries should require clarification
- **User Satisfaction**: > 4.0/5.0 average rating on answer quality

**Phase 4: Production Web Service Benchmarks**

**(A) Scalability Targets:**
- **Concurrent Users**: Support 100+ concurrent users
- **Request Throughput**: > 10 requests/second sustained load
- **Memory Usage**: < 2GB RAM per instance under normal load
- **CPU Utilization**: < 70% average CPU under peak load

**(B) Reliability Metrics:**
- **Service Uptime**: > 99.5% availability
- **Error Rate**: < 1% of requests result in unhandled errors
- **Recovery Time**: < 30 seconds for automatic error recovery
- **Circuit Breaker**: < 5 failures before service degradation activation

**(C) Security & Compliance:**
- **Authentication Response**: < 100ms for JWT validation
- **PII Scanning Coverage**: 100% of user inputs and outputs scanned
- **Audit Log Completeness**: 100% of queries logged with anonymized content
- **Error Sanitization**: 0% raw error exposure to end users

### Quality Assurance Testing Framework

**Automated Testing Suite (src/rag/tests/):**

**(A) Unit Test Coverage:**
```bash
# Target: >90% code coverage across all modules
pytest --cov=src/rag --cov-report=html --cov-fail-under=90
```

**(B) Integration Test Categories:**
- **Database Integration**: Verify read-only constraints and query execution
- **LLM Integration**: Test prompt reliability and response parsing
- **Vector Store Integration**: Validate embedding storage and retrieval
- **End-to-End Workflows**: Complete user journey testing

**(C) Performance Test Implementation:**
```python
# Example performance test structure
@pytest.mark.performance
async def test_sql_generation_performance():
    queries = load_benchmark_queries()
    start_time = time.time()
    
    for query in queries:
        result = await sql_tool.ainvoke(query)
        assert result is not None
    
    total_time = time.time() - start_time
    avg_time = total_time / len(queries)
    assert avg_time < 2.0, f"Average SQL generation time {avg_time:.2f}s exceeds 2s threshold"
```

**Load Testing Specifications:**

**(A) Gradual Load Increase:**
```bash
# Using locust or similar tool
# Start: 1 user, increase by 10 every 30 seconds until 100 users
# Duration: 10 minutes sustained load at peak
# Monitor: Response times, error rates, resource utilization
```

**(B) Stress Testing Scenarios:**
- **Database Connection Pool Exhaustion**: Verify graceful degradation
- **LLM API Rate Limiting**: Test fallback mechanisms
- **Memory Pressure**: Ensure garbage collection effectiveness
- **Vector Store Query Load**: Validate index performance under load

**Monitoring & Alerting Thresholds:**

**(A) Real-time Metrics:**
- **Response Time P95**: Alert if > 10 seconds for any query type
- **Error Rate**: Alert if > 2% over 5-minute window
- **Database Connection**: Alert if connection failures > 1% 
- **Memory Usage**: Alert if > 85% of allocated memory

**(B) Business Logic Metrics:**
- **Classification Accuracy**: Daily report on routing decisions
- **User Satisfaction**: Weekly aggregation of feedback scores
- **Query Complexity**: Track distribution of query types and processing times
- **Security Events**: Real-time alerts for any PII detection failures

**Performance Optimization Guidelines:**

**(A) Caching Strategy:**
- **Schema Descriptions**: Cache for 1 hour, refresh automatically
- **Frequent Queries**: Cache SQL results for 15 minutes
- **Embeddings**: Permanent cache with version-based invalidation
- **Classification Rules**: In-memory cache for regex patterns

**(B) Resource Optimization:**
- **Connection Pooling**: Maintain 5-10 DB connections per instance
- **Batch Processing**: Group embedding generation requests
- **Async Operations**: Use asyncio for all I/O operations
- **Memory Management**: Implement result streaming for large datasets

**(C) Continuous Improvement Process:**
- **Weekly Performance Reviews**: Analyze metrics and identify bottlenecks
- **Monthly Optimization Sprints**: Focus on worst-performing components
- **Quarterly Architecture Reviews**: Assess scalability and technology choices
- **User Feedback Integration**: Incorporate satisfaction data into optimization priorities

---

## 7. Future Enhancement Roadmap

### 7.1 Advanced Query Classification Features

**Objective**: Expand beyond MVP query classification to sophisticated, learning-based systems

#### 7.1.1 Machine Learning-Based Classification
- **Implementation**: Fine-tuned BERT model trained on Australian Public Service query patterns
- **Benefits**: Improved accuracy for complex and domain-specific queries beyond rule-based patterns
- **Timeline**: Phase 4 consideration
- **Technical Requirements**: Training dataset curation, model fine-tuning infrastructure

#### 7.1.2 Dynamic Pattern Learning
- **Objective**: Automatically identify and incorporate new classification patterns from usage data
- **Implementation**: Pattern mining algorithms to extract successful classification signals
- **Benefits**: Continuous improvement without manual pattern maintenance
- **Privacy Considerations**: Pattern extraction from anonymised query logs only

#### 7.1.3 Multi-Language Support
- **Objective**: Support queries in multiple languages relevant to Australian Public Service
- **Implementation**: Multi-language PII detection and classification with language-specific patterns
- **Benefits**: Enhanced accessibility for diverse user base
- **Scope**: Initially focus on English variations, expand to other languages as needed

#### 7.1.4 Advanced Confidence Calibration
- **Objective**: Sophisticated confidence scoring using ensemble classification methods
- **Implementation**: Combine multiple classification signals (rule-based, LLM, historical accuracy) with learned weights
- **Benefits**: More accurate confidence estimation leading to better routing decisions
- **Approach**: Implement weighted voting system with performance-based weight adjustment

#### 7.1.5 User Feedback Integration
- **Objective**: Incorporate user feedback to improve classification accuracy through active learning
- **Implementation**: Feedback collection system integrated with classification results
- **Benefits**: Continuous improvement based on real user interactions and corrections
- **Privacy**: Feedback collection with mandatory PII anonymisation

### 7.2 Enhanced Resilience Features

#### 7.2.1 Predictive Failure Detection
- **Objective**: Predict and prevent classification failures before they occur
- **Implementation**: Monitor classification patterns and system health metrics
- **Benefits**: Proactive error prevention rather than reactive error handling

#### 7.2.2 Adaptive Fallback Strategies
- **Objective**: Intelligently choose fallback mechanisms based on failure type and context
- **Implementation**: Failure classification system with context-aware fallback selection
- **Benefits**: More effective error recovery with minimal user impact

#### 7.2.3 Load Balancing and Scaling
- **Objective**: Distribute classification load across multiple instances for high availability
- **Implementation**: Microservice architecture with intelligent load distribution
- **Benefits**: Improved system reliability and performance at scale

### 7.3 Advanced Integration Capabilities

#### 7.3.1 External System Integration
- **Objective**: Integrate with existing Australian Public Service learning management systems
- **Implementation**: API-based integration with standardised data exchange formats
- **Benefits**: Seamless integration with existing workflows and data sources

#### 7.3.2 Real-Time Analytics Dashboard
- **Objective**: Provide real-time insights into classification performance and system health
- **Implementation**: Dashboard with live metrics, performance trends, and alert systems
- **Benefits**: Operational visibility and proactive system management

#### 7.3.3 Federated Learning Capabilities
- **Objective**: Enable cross-agency learning while maintaining data privacy
- **Implementation**: Federated learning protocols for shared model improvement
- **Benefits**: Improved accuracy through collective learning without data sharing

---

## 8. Technical Debt and Maintenance Considerations

### 8.1 Code Quality and Maintainability

#### 8.1.1 Refactoring Priorities
- **Pattern Management**: Consolidate pattern definitions into configuration-driven system
- **Error Handling**: Standardise error handling patterns across all classification components
- **Testing Infrastructure**: Expand test coverage for edge cases and failure scenarios

#### 8.1.2 Performance Optimization
- **Caching Strategy**: Implement intelligent caching for frequently classified queries
- **Pattern Compilation**: Optimise regex compilation and matching performance
- **Memory Management**: Reduce memory footprint through efficient data structures

#### 8.1.3 Documentation and Knowledge Transfer
- **Architecture Documentation**: Maintain comprehensive technical documentation
- **Operational Runbooks**: Create detailed operational procedures for common scenarios
- **Training Materials**: Develop training materials for system administrators and developers

### 8.2 Scalability Considerations

#### 8.2.1 Horizontal Scaling
- **Stateless Design**: Ensure classification components can scale horizontally
- **Database Optimization**: Implement efficient database connection pooling and query optimization
- **API Rate Limiting**: Implement intelligent rate limiting for external service calls

#### 8.2.2 Monitoring and Observability
- **Health Checks**: Comprehensive health monitoring for all system components
- **Performance Metrics**: Detailed performance tracking and alerting
- **Audit Logging**: Complete audit trail for compliance and troubleshooting
