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
Based on **Hybrid Approach – SQL Execution with Semantic Search over Serialised Data**, this module provides:

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

### Phase 1: Minimal Text-to-SQL MVP

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

### Phase 2: Vector Search Implementation & Unstructured Data Ingestion

#### Objectives
- Implement semantic search capabilities over the three free-text evaluation fields
- Create secure content ingestion pipeline with mandatory PII anonymisation
- Integrate vector search with existing sentiment analysis data
- Build testable vector search tool for terminal application integration

#### Target Data Sources
Based on Learning Content Evaluation table schema, Phase 2 will process these specific free-text fields:
- **`did_experience_issue_detail`**: Detailed issue descriptions (conditional on `did_experience_issue` selection)
- **`course_application_other`**: Custom application descriptions (conditional on "Other" selection)
- **`general_feedback`**: Open-ended course feedback and comments

These fields are already processed by the sentiment analysis module and contain rich contextual information for semantic search.

#### Key Tasks

**2.1 Data Source Analysis & PII Detection Strategy**
- Document the three target free-text fields and their data characteristics
- Implement Australian-specific PII detection using Microsoft Presidio:
  - Tax File Numbers (TFN), Medicare numbers, Australian addresses
  - Names, email addresses, phone numbers
  - APS-specific identifiers and sensitive references
- Create `src/rag/core/privacy/pii_detector.py` with Australian privacy compliance
- Establish PII anonymisation as **mandatory step before embedding generation**

**2.2 pgVector Infrastructure & Schema Design**
- Create dedicated pgvector table with explicit schema:
  ```sql
  CREATE TABLE rag_embeddings (
      embedding_id SERIAL PRIMARY KEY,
      response_id INTEGER NOT NULL REFERENCES evaluation(response_id),
      field_name VARCHAR(50) NOT NULL, -- 'did_experience_issue_detail', 'course_application_other', 'general_feedback'
      chunk_text TEXT NOT NULL,        -- Anonymised text chunk
      chunk_index INTEGER NOT NULL,    -- Chunk position within original text
      embedding VECTOR(1536) NOT NULL, -- OpenAI ada-002 or equivalent
      metadata JSONB,                  -- {user_level, agency, sentiment_scores, course_type}
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(response_id, field_name, chunk_index)
  );
  
  CREATE INDEX ON rag_embeddings USING ivfflat (embedding vector_cosine_ops);
  CREATE INDEX ON rag_embeddings (response_id);
  CREATE INDEX ON rag_embeddings (field_name);
  ```
- Implement `src/rag/data/embeddings_manager.py` for vector operations and metadata management

**2.3 Text Processing & Anonymisation Pipeline**
- Develop `src/rag/core/vector_search/text_processor.py`:
  - Extract the three target free-text fields from evaluation table
  - Apply PII detection and anonymisation before chunking
  - Implement sentence-level chunking (appropriate for evaluation responses)
  - Preserve metadata linkage (response_id, field_name, user context)
- Integration with existing sentiment analysis results via response_id mapping

**2.4 Embedding Generation & Storage System**
- Implement `src/rag/core/vector_search/embedder.py`:
  - Batch processing for efficient embedding generation
  - Support for OpenAI `text-embedding-ada-002` (primary) and Sentence-BERT (fallback)
  - Error handling and retry logic for API failures
  - Metadata augmentation with sentiment scores from existing sentiment analysis
- Store embeddings with rich metadata for filtering and retrieval

**2.5 Vector Search Tool Development**
- Create `src/rag/core/tools/vector_search_tool.py` as standalone LangChain tool:
  - Similarity search with configurable distance metrics (cosine, euclidean)
  - Metadata filtering capabilities (sentiment, user_level, agency, field_name)
  - Result ranking with combined similarity and metadata relevance
  - Integration interface for future LangGraph router
- Design for modularity and testability independent of router implementation

**2.6 Initial Data Ingestion Script**
- Implement `src/rag/data/ingestion_script.py` for one-off data loading:
  - Extract all evaluation records with non-empty target fields
  - Process through PII detection → chunking → embedding → storage pipeline
  - Comprehensive logging and error handling for batch processing
  - Integration with existing sentiment analysis data
  - Monitoring and progress reporting for large dataset processing

#### Components to be Developed/Modified

**New Components:**
- `src/rag/core/privacy/pii_detector.py` - Presidio-based PII detection
- `src/rag/core/vector_search/text_processor.py` - Text extraction and anonymisation
- `src/rag/core/vector_search/embedder.py` - Embedding generation system
- `src/rag/core/tools/vector_search_tool.py` - Standalone vector search tool
- `src/rag/data/embeddings_manager.py` - Vector storage and retrieval
- `src/rag/data/ingestion_script.py` - Initial data loading pipeline

**Database Schema Extensions:**
- Add `rag_embeddings` table to pgvector database
- Create appropriate indexes for vector similarity search
- Establish foreign key relationships with evaluation table

**Modified Components:**
- Update `src/rag/config/settings.py` with vector search configuration
- Extend `requirements.txt` with Presidio, vector search libraries
- Update `docker-compose.yml` if local embedding models are used

#### Architecture Documentation Updates
- Document vector search architecture with explicit data flow
- Add PII anonymisation pipeline diagrams
- Include metadata filtering and hybrid ranking explanations
- Document integration patterns with existing sentiment analysis data
- Create testing strategy for vector search accuracy and security

#### Data Privacy/Governance Steps (MANDATORY)
- **PII Anonymisation**: Implement comprehensive PII detection before embedding
- **Data Lineage**: Track anonymisation steps and embedding generation
- **Access Control**: Maintain read-only database access for RAG operations
- **Audit Logging**: Log all PII detection events and anonymisation actions
- **Retention Policy**: Establish policies for embedding storage and lifecycle management

#### Testing/Validation Strategy
- **Unit Tests**: PII detection accuracy, embedding generation, vector operations
- **Integration Tests**: End-to-end text processing pipeline with real evaluation data
- **Security Tests**: PII leakage prevention, anonymisation effectiveness
- **Performance Tests**: Embedding generation speed, similarity search latency
- **Quality Tests**: Embedding relevance using evaluation against known similar responses
- **Terminal Integration**: Test vector search tool independently via terminal interface

#### Modular Design for Future Integration
- Vector search tool designed as standalone LangChain component
- Clear interfaces for LangGraph router integration in Phase 3
- Separation of concerns: PII detection, text processing, embedding, search
- Configurable similarity thresholds and metadata filtering
- Extensible metadata schema for future enhancement

#### Components to be Developed/Modified

**New Components:**
- `src/rag/core/privacy/pii_detector.py` - Presidio-based PII detection
- `src/rag/core/vector_search/text_processor.py` - Text extraction and anonymisation
- `src/rag/core/vector_search/embedder.py` - Embedding generation system
- `src/rag/core/tools/vector_search_tool.py` - Standalone vector search tool
- `src/rag/data/embeddings_manager.py` - Vector storage and retrieval
- `src/rag/data/ingestion_script.py` - Initial data loading pipeline

**Database Schema Extensions:**
- Add `rag_embeddings` table to pgvector database
- Create appropriate indexes for vector similarity search
- Establish foreign key relationships with evaluation table

**Modified Components:**
- Update `src/rag/config/settings.py` with vector search configuration
- Extend `requirements.txt` with Presidio, vector search libraries
- Update `docker-compose.yml` if local embedding models are used

#### Architecture Documentation Updates
- Document vector search architecture with explicit data flow
- Add PII anonymisation pipeline diagrams
- Include metadata filtering and hybrid ranking explanations
- Document integration patterns with existing sentiment analysis data
- Create testing strategy for vector search accuracy and security

#### Data Privacy/Governance Steps (MANDATORY)
- **PII Anonymisation**: Implement comprehensive PII detection before embedding
- **Data Lineage**: Track anonymisation steps and embedding generation
- **Access Control**: Maintain read-only database access for RAG operations
- **Audit Logging**: Log all PII detection events and anonymisation actions
- **Retention Policy**: Establish policies for embedding storage and lifecycle management

#### Testing/Validation Strategy
- **Unit Tests**: PII detection accuracy, embedding generation, vector operations
- **Integration Tests**: End-to-end text processing pipeline with real evaluation data
- **Security Tests**: PII leakage prevention, anonymisation effectiveness
- **Performance Tests**: Embedding generation speed, similarity search latency
- **Quality Tests**: Embedding relevance using evaluation against known similar responses
- **Terminal Integration**: Test vector search tool independently via terminal interface

#### Modular Design for Future Integration
- Vector search tool designed as standalone LangChain component
- Clear interfaces for LangGraph router integration in Phase 3
- Separation of concerns: PII detection, text processing, embedding, search
- Configurable similarity thresholds and metadata filtering
- Extensible metadata schema for future enhancement

**Example Queries Enabled:**
```bash
# Direct vector search testing via terminal
> "Find courses with negative feedback about facilitator communication"
> "What do users say about technical difficulties in virtual courses?"
> "Show similar feedback patterns across different agencies"
> "Locate evaluation responses mentioning accessibility issues"
```

#### Success Criteria
- **Security**: Zero PII leakage in embeddings with 99.9%+ detection accuracy
- **Performance**: Vector search returns relevant results with >80% precision@10
- **Scalability**: Embedding generation processes full evaluation dataset in <2 hours
- **Integration**: Seamless metadata filtering with sentiment analysis data
- **Modularity**: Vector search tool functions independently for testing and future router integration
- **Compliance**: Full anonymisation pipeline meets Australian Privacy Principles (APP) requirements

#### Implementation Priority
1. **PII Detection & Anonymisation** (Security Critical)
2. **pgVector Schema & Infrastructure** (Foundation)
3. **Text Processing Pipeline** (Core Functionality)
4. **Embedding Generation** (Core Functionality)
5. **Vector Search Tool** (Testable Interface)
6. **Data Ingestion Script** (MVP Completion)

---

### Phase 3: Hybrid Query Routing (LangGraph) & Answer Synthesis

#### Objectives
- Implement **core LangGraph agent** as central intelligence for the architecture
- Create robust query classification system to route between SQL, Vector Search, and Hybrid approaches
- Build comprehensive answer synthesis using retrieved context from multiple sources
- Integrate Phase 1 and Phase 2 tools into unified LangGraph workflow

#### Core LangGraph Architecture
The LangGraph agent implements a **classify → route → execute → synthesise** workflow:

```python
# State definition for LangGraph agent
class RAGState(TypedDict):
    query: str
    query_type: Literal["sql", "vector", "hybrid"]
    classification_confidence: float
    sql_results: Optional[List[Dict]]
    vector_results: Optional[List[Dict]]
    context: str
    final_answer: str
    error_message: Optional[str]
```

#### Key Tasks

**3.1 LangGraph Agent Development (CORE IMPLEMENTATION)**
- Create `src/rag/core/agent.py` as the main LangGraph orchestrator:
  - **`classify_query_node`**: Determine if query requires SQL, Vector Search, or Hybrid approach
  - **`sql_tool_node`**: Execute Text-to-SQL using Phase 1 tools
  - **`vector_search_tool_node`**: Execute semantic search using Phase 2 tools
  - **`synthesis_node`**: Generate coherent answer using LLM with retrieved context
  - **Conditional edges**: Route based on classification to appropriate tool nodes
  - **Error handling nodes**: Graceful failure management and fallback strategies

**3.2 Query Classification Component**
- Implement `src/rag/core/routing/query_classifier.py`:
  - **LLM-based classification** using prompt engineering for query intent detection
  - Classification categories:
    - **SQL**: Aggregations, filters, joins, statistical queries ("How many users completed courses?")
    - **Vector**: Semantic search, thematic analysis ("Find feedback about technical issues")
    - **Hybrid**: Combined structured + unstructured queries ("Which courses with low completion rates have negative feedback?")
  - **Confidence scoring** for classification reliability
  - **Fallback logic** for uncertain classifications

**3.3 LangGraph State Management & Flow Control**
- Define `RAGState` TypedDict for information passing between nodes
- Implement state transitions and data flow management
- Create conditional edge logic based on query classification
- Handle state persistence for multi-step processing
- Error state management and recovery workflows

**3.4 Tool Integration into LangGraph Workflow**
- **SQL Tool Integration**: Wrap Phase 1 Text-to-SQL functionality as LangGraph-compatible tool
- **Vector Search Tool Integration**: Wrap Phase 2 vector search functionality as LangGraph-compatible tool
- **Tool execution nodes** with proper error handling and result formatting
- **Result standardisation** for consistent data structure across tools

**3.5 Answer Synthesis System**
- Implement `src/rag/core/synthesis/answer_generator.py`:
  - **Context aggregation** from SQL and/or vector search results
  - **LLM-based synthesis** using structured prompts for coherent answer generation
  - **Source attribution** and citation linking to original data
  - **Answer formatting** with confidence indicators and supporting evidence
  - **Multi-format outputs**: Summary, detailed, tabular presentation

**3.6 Prompt Engineering for LangGraph Components**
- **Query Classification Prompts**: Optimised prompts for accurate query type detection
- **Answer Synthesis Prompts**: Templates for coherent response generation with proper source attribution
- **Error Handling Prompts**: User-friendly error explanations and suggested query refinements
- **Confidence Assessment Prompts**: LLM-based confidence scoring for query classification and answer quality

#### Components to be Developed/Modified

**New Components:**
- `src/rag/core/agent.py` - **Main LangGraph agent with classify → route → execute → synthesise workflow**
- `src/rag/core/routing/query_classifier.py` - **LLM-based query classification component**
- `src/rag/core/synthesis/answer_generator.py` - **Context aggregation and answer synthesis**
- `src/rag/core/nodes/` - **Individual LangGraph node implementations**
  - `classify_node.py` - Query classification node
  - `sql_node.py` - SQL tool execution node
  - `vector_node.py` - Vector search execution node
  - `synthesis_node.py` - Answer generation node
  - `error_handling_node.py` - Error management and fallback
- `src/rag/core/state/rag_state.py` - **LangGraph state definition and management**

**Tool Integration Wrappers:**
- `src/rag/core/tools/langgraph_sql_tool.py` - **Phase 1 SQL tool wrapper for LangGraph**
- `src/rag/core/tools/langgraph_vector_tool.py` - **Phase 2 vector search tool wrapper for LangGraph**

**Enhanced Terminal Application:**
- Update `src/rag/interfaces/terminal_app.py` - **Integration with LangGraph agent as primary entry point**
- Modify `src/rag/runner.py` - **Route all queries through LangGraph agent**

#### Terminal Application Integration (CRITICAL UPDATE)
The terminal application must be updated to use the LangGraph agent as the **primary entry point**:

```python
# Updated terminal_app.py approach
class TerminalApp:
    def __init__(self):
        self.langgraph_agent = compile_rag_agent()  # Main LangGraph agent
    
    def process_query(self, user_query: str):
        # Route ALL queries through LangGraph agent
        result = self.langgraph_agent.invoke({"query": user_query})
        return result["final_answer"]
```

#### LangGraph Agent Architecture
```python
# Core agent structure in src/rag/core/agent.py
def create_rag_agent() -> StateGraph:
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("sql_tool", sql_tool_node)
    workflow.add_node("vector_search_tool", vector_search_tool_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("error_handler", error_handling_node)
    
    # Conditional routing edges
    workflow.add_conditional_edges(
        "classify_query",
        route_based_on_classification,
        {
            "sql": "sql_tool",
            "vector": "vector_search_tool", 
            "hybrid": ["sql_tool", "vector_search_tool"],
            "error": "error_handler"
        }
    )
    
    # Synthesis edges
    workflow.add_edge("sql_tool", "synthesis")
    workflow.add_edge("vector_search_tool", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()
```

#### Error Handling Strategy
- **Classification uncertainty**: Fallback to hybrid approach when confidence < threshold
- **Tool execution failures**: Graceful degradation and alternative tool usage
- **LLM API failures**: Local fallback models and cached responses
- **User-friendly error messages**: Explain failures and suggest query modifications

#### Prompt Engineering Strategy

**Query Classification Prompt Template:**
```python
CLASSIFICATION_PROMPT = """
Analyze this query and classify it as SQL, VECTOR, or HYBRID:

SQL: Queries about counts, statistics, aggregations, filtering structured data
VECTOR: Queries about themes, sentiment, finding similar content
HYBRID: Queries combining both structured analysis and content search

Query: {query}

Respond with classification and confidence (0-1):
Classification: [SQL|VECTOR|HYBRID]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation]
"""
```

**Answer Synthesis Prompt Template:**
```python
SYNTHESIS_PROMPT = """
Based on the retrieved information, provide a comprehensive answer to the user's query.

Query: {query}
SQL Results: {sql_results}
Vector Search Results: {vector_results}

Requirements:
- Provide clear, accurate answer based on the data
- Include specific numbers and evidence when available
- Cite sources with response_id references
- Acknowledge limitations if data is incomplete

Answer:
"""
```

#### Architecture Documentation Updates
- **Document LangGraph workflow architecture** with explicit node and edge definitions
- **Add query classification decision trees** with confidence threshold mapping
- **Include answer synthesis pipeline diagrams** showing context aggregation
- **Document state management** and data flow between LangGraph nodes
- **Create prompt engineering documentation** with template versioning

#### Data Privacy/Governance Steps
- **Query audit logging** with classification tracking and confidence scores
- **Result provenance tracking** through LangGraph state management
- **PII detection integration** in synthesis node before answer generation
- **Access control enforcement** at agent level for query restrictions
- **Error logging and monitoring** for security incident detection

#### Testing/Validation Strategy
- **Unit Tests**: Query classification accuracy, individual node functionality
- **Integration Tests**: End-to-end LangGraph workflow with Phase 1 and Phase 2 tools
- **LangGraph Flow Tests**: State transitions, conditional routing, error handling
- **Terminal Application Tests**: Complete user journey through LangGraph agent
- **Performance Tests**: Multi-step query processing latency and resource usage
- **Prompt Engineering Tests**: Classification accuracy and answer quality assessment

#### Modular Design for LangGraph Integration
- **Tool-agnostic design**: LangGraph agent works with any SQL or vector search implementation
- **Configurable routing thresholds**: Adjustable confidence levels for classification
- **Extensible node architecture**: Easy addition of new processing nodes
- **State schema versioning**: Backward compatibility for state structure changes
- **Error handling hierarchy**: Multiple levels of fallback and recovery

**Example LangGraph Query Processing:**
```bash
# Terminal interaction routed through LangGraph agent
> "Which courses with completion rates under 70% have negative feedback about workload?"

[LangGraph Flow]
1. classify_query_node → "HYBRID" (confidence: 0.9)
2. Parallel execution:
   - sql_tool_node → Query completion rates < 70%
   - vector_search_tool_node → Find "workload" negative sentiment
3. synthesis_node → Combine results with source attribution
4. Return comprehensive answer with citations

Response: "Based on the data, 3 courses meet your criteria:
- Course X (65% completion, 15 negative workload mentions)  
- Course Y (58% completion, 8 negative workload mentions)
- Course Z (69% completion, 12 negative workload mentions)

Sources: SQL analysis from attendance table, sentiment analysis from evaluation responses [response_id: 1234, 1567, 1890...]"
```

#### Success Criteria
- **Query Classification**: 95%+ accuracy on predefined test query set
- **LangGraph Integration**: Seamless routing between Phase 1 and Phase 2 tools
- **Answer Synthesis**: Coherent responses with proper source attribution
- **Terminal Integration**: All queries processed through LangGraph agent
- **Error Handling**: Graceful degradation with <5% system failures
- **Performance**: End-to-end query processing under 15 seconds for complex queries
- **User Experience**: Natural language interaction with confidence indicators

#### Implementation Priority
1. **LangGraph Agent Core** (`agent.py` with basic workflow) - Foundation
2. **Query Classification** (`query_classifier.py`) - Routing Intelligence  
3. **Tool Integration Wrappers** (Phase 1 + Phase 2 tools) - Functionality
4. **Answer Synthesis** (`answer_generator.py`) - User Experience
5. **Terminal App Integration** (Updated `terminal_app.py`) - Complete MVP
6. **Error Handling & Prompt Engineering** - Production Readiness

#### Critical Dependencies
- **Phase 1 Complete**: Text-to-SQL tool must be functional and testable
- **Phase 2 Complete**: Vector search tool must be functional and testable  
- **LangGraph Framework**: Installation and configuration in requirements.txt
- **Prompt Templates**: Well-engineered prompts for classification and synthesis

---

### Phase 4: FastAPI Web Service & Advanced RAG Techniques

#### Objectives
- Develop production-ready FastAPI service with asynchronous architecture
- Implement advanced RAG techniques for improved answer quality
- Create comprehensive SQL query validation and safety mechanisms
- Establish robust monitoring, logging, and evaluation frameworks

#### Key Tasks

**4.1 FastAPI Service Architecture**

Create a production-ready web service that reuses core logic from the LangGraph agent:

```
src/api/
├── __init__.py
├── main.py                 # FastAPI application entry point
├── routers/
│   ├── __init__.py
│   ├── query.py           # Query processing endpoints
│   ├── admin.py           # Administrative endpoints
│   └── health.py          # Health check endpoints
├── middleware/
│   ├── __init__.py
│   ├── auth.py            # Authentication middleware
│   ├── rate_limit.py      # Rate limiting middleware
│   └── logging.py         # Request logging middleware
├── models/
│   ├── __init__.py
│   ├── request.py         # Pydantic request models
│   └── response.py        # Pydantic response models
└── core/
    ├── __init__.py
    ├── agent_wrapper.py   # Async wrapper for LangGraph agent
    └── config.py          # API-specific configuration
```

**Core Implementation Strategy:**
- **Reuse LangGraph Agent**: Wrap existing `src/rag/core/agent.py` with async interface
- **Async Architecture**: All endpoints use `async def` with `ainvoke()` methods
- **FastAPI Best Practices**: Dependency injection, middleware, and proper error handling

**Priority Implementation Tasks:**
1. Create async wrapper for LangGraph agent (`src/api/core/agent_wrapper.py`)
2. Implement core endpoints with Pydantic models for validation
3. Add authentication and rate limiting middleware
4. Create comprehensive OpenAPI documentation

**4.2 Advanced RAG Techniques Implementation**

Enhance the RAG system with sophisticated retrieval and generation methods:

**Re-ranking Pipeline (`src/rag/core/reranker.py`):**
```python
from sentence_transformers import CrossEncoder
from typing import List, Tuple

class DocumentReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = CrossEncoder(model_name)
    
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Re-rank documents using cross-encoder for relevance"""
        # Implementation details...
```

**Query Transformation (`src/rag/core/query_transformer.py`):**
- **Hypothetical Document Embeddings (HyDE)**: Generate hypothetical answers and use for retrieval
- **Query Decomposition**: Break complex queries into sub-questions
- **Query Expansion**: Add synonyms and related terms for better retrieval

**Multi-Query Generation (`src/rag/core/multi_query.py`):**
```python
async def generate_multiple_queries(original_query: str) -> List[str]:
    """Generate multiple query variations for comprehensive retrieval"""
    prompt = f"""
    Generate 3 different ways to ask this question for better document retrieval:
    Original: {original_query}
    
    Variations:
    1.
    2.
    3.
    """
    # Implementation with LLM call...
```

**4.3 SQL Query Validation and Safety**

Implement comprehensive SQL validation using LangChain's QuerySQLCheckerTool:

**SQL Validator (`src/rag/tools/sql_validator.py`):**
```python
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langchain_community.utilities import SQLDatabase
from typing import Dict, Any

class SQLQueryValidator:
    def __init__(self, db_uri: str):
        self.db = SQLDatabase.from_uri(db_uri)
        self.validator = QuerySQLCheckerTool(db=self.db, llm=llm)
    
    async def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate SQL query for safety and correctness"""
        try:
            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']
            query_upper = query.upper()
            
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return {
                        "valid": False,
                        "error": f"Dangerous operation detected: {keyword}",
                        "query": query
                    }
            
            # Use LangChain validator
            validation_result = await self.validator.ainvoke({"query": query})
            
            return {
                "valid": True,
                "validated_query": validation_result,
                "original_query": query
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "query": query
            }
```

**Integration with Text-to-SQL Tool:**
- All generated SQL queries must pass validation before execution
- Implement query complexity limits (max joins, subqueries, etc.)
- Add query timeout and resource usage monitoring
- Create SQL injection prevention measures

**4.4 Monitoring, Logging, and Evaluation Framework**

**Structured Logging (`src/rag/utils/logging.py`):**
```python
import structlog
from typing import Dict, Any
import time

class RAGLogger:
    def __init__(self):
        self.logger = structlog.get_logger()
    
    async def log_query(
        self, 
        query: str, 
        user_id: str, 
        query_type: str,
        response_time: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ):
        """Log query with structured data for analysis"""
        self.logger.info(
            "rag_query",
            query=query,
            user_id=user_id,
            query_type=query_type,
            response_time=response_time,
            success=success,
            timestamp=time.time(),
            metadata=metadata or {}
        )
```

**Performance Monitoring (`src/rag/monitoring/metrics.py`):**
```python
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Metrics definitions
QUERY_COUNTER = Counter('rag_queries_total', 'Total RAG queries', ['query_type', 'status'])
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query processing time', ['query_type'])
ACTIVE_QUERIES = Gauge('rag_active_queries', 'Currently processing queries')

def monitor_query_performance(query_type: str):
    """Decorator to monitor query performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            ACTIVE_QUERIES.inc()
            
            try:
                result = await func(*args, **kwargs)
                QUERY_COUNTER.labels(query_type=query_type, status='success').inc()
                return result
            except Exception as e:
                QUERY_COUNTER.labels(query_type=query_type, status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                QUERY_DURATION.labels(query_type=query_type).observe(duration)
                ACTIVE_QUERIES.dec()
        
        return wrapper
    return decorator
```

**Answer Quality Evaluation (`src/rag/evaluation/evaluator.py`):**
```python
from typing import Dict, List, Any
import asyncio

class RAGEvaluator:
    def __init__(self):
        self.llm = # Initialize LLM for evaluation
    
    async def evaluate_answer_quality(
        self, 
        query: str, 
        answer: str, 
        retrieved_docs: List[str]
    ) -> Dict[str, float]:
        """Evaluate answer quality on multiple dimensions"""
        
        # Relevance evaluation
        relevance_prompt = f"""
        Query: {query}
        Answer: {answer}
        
        Rate the relevance of the answer to the query on a scale of 1-5:
        """
        
        # Faithfulness evaluation (answer supported by retrieved docs)
        faithfulness_prompt = f"""
        Retrieved Documents: {retrieved_docs}
        Answer: {answer}
        
        Rate how well the answer is supported by the documents on a scale of 1-5:
        """
        
        # Run evaluations concurrently
        relevance_task = self._evaluate_dimension(relevance_prompt)
        faithfulness_task = self._evaluate_dimension(faithfulness_prompt)
        
        relevance_score, faithfulness_score = await asyncio.gather(
            relevance_task, faithfulness_task
        )
        
        return {
            "relevance": relevance_score,
            "faithfulness": faithfulness_score,
            "overall": (relevance_score + faithfulness_score) / 2
        }
```

#### Components to be Developed/Modified

**New Components:**
- FastAPI application with async architecture (`src/api/`)
- Advanced RAG techniques implementation (`src/rag/core/reranker.py`, etc.)
- SQL query validation system (`src/rag/tools/sql_validator.py`)
- Comprehensive monitoring and logging framework (`src/rag/monitoring/`, `src/rag/utils/`)
- Answer quality evaluation system (`src/rag/evaluation/`)

**Modified Components:**
- Update LangGraph agent to support async operations (`src/rag/core/agent.py`)
- Enhance text-to-SQL tool with validation integration (`src/rag/tools/text_to_sql.py`)
- Modify vector search tool for re-ranking support (`src/rag/tools/vector_search.py`)
- Update configuration management for production settings (`src/rag/config.py`)

**Integration Requirements:**
- All FastAPI endpoints must use async/await patterns
- LangGraph agent wrapper must maintain state consistency
- Monitoring must be integrated into all major components
- SQL validation must be mandatory for all generated queries

#### Architecture Documentation Updates
- Document async architecture patterns and best practices
- Add comprehensive API reference with OpenAPI specification
- Include monitoring and observability setup guides
- Document advanced RAG techniques and their use cases
- Create SQL validation and security documentation

#### Data Privacy/Governance Steps
- Implement request/response logging with PII detection
- Create audit trails for all API access and SQL queries
- Establish data retention policies for logs and metrics
- Document security controls for FastAPI deployment

#### Testing/Validation
- **Unit Tests**: All new components with >90% coverage
- **Integration Tests**: FastAPI endpoints with realistic workloads
- **Performance Tests**: Load testing with concurrent async requests
- **Security Tests**: SQL injection and API security validation
- **Evaluation Tests**: Answer quality assessment on test dataset

**Testing Strategy:**
```python
# Example async test for FastAPI endpoint
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_query_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/query", 
            json={"query": "What are common course issues?"}
        )
    assert response.status_code == 200
    assert "answer" in response.json()
```

#### Success Criteria
- **Performance**: API handles 200+ concurrent requests with <3s median response
- **Quality**: Advanced RAG techniques improve answer relevance by 25%
- **Security**: 100% of SQL queries pass validation without false positives
- **Monitoring**: Comprehensive metrics collection with <1% overhead
- **Reliability**: 99.9% uptime with proper error handling and recovery
- **Code Quality**: >90% test coverage with comprehensive documentation

---

### Phase 5: Production Operations & Advanced Analytics

#### Objectives
- Establish production deployment and operations infrastructure
- Implement comprehensive security and compliance frameworks
- Create advanced analytics and business intelligence capabilities
- Develop continuous improvement and optimization processes

#### Key Tasks

**5.1 Production Deployment Infrastructure**

**Containerization and Orchestration (`deployment/`):**
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-api:
    build: 
      context: .
      dockerfile: Dockerfile.api
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  redis_data:
```

**CI/CD Pipeline (`.github/workflows/deploy.yml`):**
```yaml
name: Deploy RAG System

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          python -m pytest tests/ --cov=src --cov-report=xml
          
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security scan
        run: |
          bandit -r src/
          safety check
          
  deploy:
    needs: [test, security]
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml up -d
```

**5.2 Security and Compliance Framework**

**Authentication and Authorization (`src/api/security/`):**
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List
import jwt

class RoleBasedAuth:
    def __init__(self):
        self.security = HTTPBearer()
        self.roles = {
            "analyst": ["query", "explain"],
            "admin": ["query", "explain", "admin", "metrics"],
            "readonly": ["query"]
        }
    
    async def verify_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Dict[str, str]:
        """Verify JWT token and extract user info"""
        try:
            payload = jwt.decode(
                credentials.credentials, 
                SECRET_KEY, 
                algorithms=["HS256"]
            )
            return {
                "user_id": payload["sub"],
                "role": payload["role"],
                "permissions": self.roles.get(payload["role"], [])
            }
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = await self.verify_token()
                if permission not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                return await func(*args, **kwargs)
            return wrapper
        return decorator
```

**Data Privacy Controls (`src/rag/privacy/`):**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import Dict, List, Any

class PrivacyController:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
    async def scan_and_anonymize_response(
        self, 
        response: str, 
        anonymization_level: str = "strict"
    ) -> Dict[str, Any]:
        """Scan response for PII and anonymize if necessary"""
        
        # Analyze for PII
        results = self.analyzer.analyze(
            text=response,
            language='en',
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "AU_TFN", "AU_ABN"]
        )
        
        if results:
            # Anonymize detected PII
            anonymized_response = self.anonymizer.anonymize(
                text=response,
                analyzer_results=results
            )
            
            return {
                "original_response": response,
                "anonymized_response": anonymized_response.text,
                "pii_detected": True,
                "pii_entities": [r.entity_type for r in results],
                "anonymization_applied": True
            }
        
        return {
            "response": response,
            "pii_detected": False,
            "anonymization_applied": False
        }
```

**5.3 Advanced Analytics and Business Intelligence**

**Query Pattern Analytics (`src/analytics/query_patterns.py`):**
```python
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

class QueryAnalytics:
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def analyze_query_patterns(
        self, 
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze query patterns and user behavior"""
        
        # Query frequency analysis
        query_freq = await self._get_query_frequency(days)
        
        # Topic clustering
        topics = await self._cluster_query_topics()
        
        # Performance trends
        performance = await self._analyze_performance_trends(days)
        
        # User behavior analysis
        user_behavior = await self._analyze_user_behavior(days)
        
        return {
            "query_frequency": query_freq,
            "popular_topics": topics,
            "performance_trends": performance,
            "user_behavior": user_behavior,
            "recommendations": await self._generate_recommendations()
        }
    
    async def _cluster_query_topics(self) -> List[Dict[str, Any]]:
        """Cluster queries by topic using embeddings"""
        # Implementation using vector similarity clustering
        pass
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate system improvement recommendations"""
        return [
            "Add FAQ section for top 10 most common queries",
            "Optimize vector search for slow query patterns",
            "Create additional training data for low-confidence topics"
        ]
```

**Performance Optimization Dashboard (`src/analytics/performance.py`):**
```python
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

class PerformanceDashboard:
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Query latency distribution
        latency_data = await self._get_latency_metrics()
        latency_fig = px.histogram(
            latency_data, 
            x='response_time', 
            title='Query Response Time Distribution'
        )
        
        # Success rate trends
        success_data = await self._get_success_metrics()
        success_fig = px.line(
            success_data, 
            x='date', 
            y='success_rate',
            title='Query Success Rate Over Time'
        )
        
        # Resource utilization
        resource_data = await self._get_resource_metrics()
        resource_fig = go.Figure()
        resource_fig.add_trace(go.Scatter(
            x=resource_data['timestamp'],
            y=resource_data['cpu_usage'],
            name='CPU Usage'
        ))
        resource_fig.add_trace(go.Scatter(
            x=resource_data['timestamp'],
            y=resource_data['memory_usage'],
            name='Memory Usage'
        ))
        
        return {
            "latency_distribution": latency_fig.to_json(),
            "success_trends": success_fig.to_json(),
            "resource_utilization": resource_fig.to_json(),
            "summary_metrics": await self._calculate_summary_metrics()
        }
```

**5.4 Continuous Improvement Framework**

**A/B Testing for RAG Improvements (`src/experimentation/ab_testing.py`):**
```python
from typing import Dict, Any, Optional
import random
from enum import Enum

class ExperimentVariant(Enum):
    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"

class RAGExperimentManager:
    def __init__(self):
        self.active_experiments = {}
    
    async def assign_variant(
        self, 
        user_id: str, 
        experiment_name: str
    ) -> ExperimentVariant:
        """Assign user to experiment variant"""
        
        # Consistent assignment based on user_id hash
        hash_value = hash(f"{user_id}_{experiment_name}") % 100
        
        if experiment_name == "reranking_model":
            if hash_value < 33:
                return ExperimentVariant.CONTROL  # No re-ranking
            elif hash_value < 66:
                return ExperimentVariant.TREATMENT_A  # Cross-encoder re-ranking
            else:
                return ExperimentVariant.TREATMENT_B  # Dual-encoder re-ranking
        
        return ExperimentVariant.CONTROL
    
    async def log_experiment_result(
        self,
        user_id: str,
        experiment_name: str,
        variant: ExperimentVariant,
        query: str,
        response_quality: float,
        user_satisfaction: Optional[float] = None
    ):
        """Log experiment results for analysis"""
        # Implementation for experiment result tracking
        pass
```

**Model Performance Monitoring (`src/monitoring/model_monitor.py`):**
```python
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

class ModelDriftMonitor:
    def __init__(self):
        self.baseline_metrics = {}
        self.drift_threshold = 0.1  # 10% degradation threshold
    
    async def check_model_drift(
        self, 
        model_name: str, 
        recent_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for model performance drift"""
        
        current_metrics = await self._calculate_metrics(recent_predictions)
        baseline = self.baseline_metrics.get(model_name, {})
        
        if not baseline:
            # Initialize baseline
            self.baseline_metrics[model_name] = current_metrics
            return {"drift_detected": False, "status": "baseline_established"}
        
        # Calculate drift
        drift_scores = {}
        for metric, current_value in current_metrics.items():
            baseline_value = baseline.get(metric, current_value)
            drift = abs(current_value - baseline_value) / baseline_value
            drift_scores[metric] = drift
        
        max_drift = max(drift_scores.values())
        drift_detected = max_drift > self.drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "max_drift": max_drift,
            "drift_scores": drift_scores,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline,
            "recommendation": self._get_drift_recommendation(drift_detected, max_drift)
        }
    
    def _get_drift_recommendation(
        self, 
        drift_detected: bool, 
        max_drift: float
    ) -> str:
        """Get recommendation based on drift analysis"""
        if not drift_detected:
            return "Model performance is stable"
        elif max_drift < 0.2:
            return "Minor drift detected - monitor closely"
        else:
            return "Significant drift detected - consider model retraining"
```

#### Components to be Developed/Modified

**New Components:**
- Production deployment infrastructure (`deployment/`, `docker-compose.yml`)
- Security and authentication framework (`src/api/security/`)
- Advanced analytics and BI dashboards (`src/analytics/`)
- A/B testing and experimentation framework (`src/experimentation/`)
- Model drift monitoring system (`src/monitoring/model_monitor.py`)
- Comprehensive privacy controls (`src/rag/privacy/`)

**Enhanced Components:**
- FastAPI application with production middleware and security
- Database migration and backup systems
- Monitoring dashboards with real-time alerts
- Documentation with operational runbooks

#### Architecture Documentation Updates
- Complete deployment and operations guide
- Security architecture and compliance documentation
- Analytics and business intelligence setup
- Troubleshooting and incident response procedures
- Performance tuning and optimization guide

#### Data Privacy/Governance Steps
- Complete Australian Privacy Principles (APP) compliance implementation
- Automated PII detection and anonymization in all responses
- Comprehensive audit logging with immutable records
- Data retention and deletion policies with automated enforcement
- Privacy impact assessment and regular compliance reviews

#### Testing/Validation
- **Production Testing**: Blue-green deployment with health checks
- **Security Testing**: Penetration testing and vulnerability assessment
- **Compliance Testing**: Automated APP compliance validation
- **Performance Testing**: Load testing with realistic production scenarios
- **Disaster Recovery Testing**: Backup and restore procedures validation

**Comprehensive Test Suite:**
```python
# Production readiness test suite
@pytest.mark.production
async def test_production_deployment():
    """Test production deployment health"""
    # Health check endpoints
    # Database connectivity
    # External service dependencies
    # Performance benchmarks
    pass

@pytest.mark.security
async def test_security_controls():
    """Test security implementations"""
    # Authentication and authorization
    # PII anonymization
    # SQL injection prevention
    # Rate limiting
    pass

@pytest.mark.compliance
async def test_privacy_compliance():
    """Test privacy compliance"""
    # APP requirement validation
    # Data retention policies
    # User consent handling
    # Audit trail completeness
    pass
```

#### Success Criteria
- **Deployment**: Zero-downtime deployments with automated rollback
- **Security**: 100% compliance with security requirements and APP regulations
- **Performance**: Sub-2s response times for 95th percentile under production load
- **Reliability**: 99.9% uptime with comprehensive monitoring and alerting
- **Analytics**: Business intelligence dashboard providing actionable insights
- **Compliance**: Successful third-party security and privacy audit
- **Operations**: Complete operational runbooks and incident response procedures

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
