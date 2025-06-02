# AI-Enhanced Survey Analysis

## 1. Overview
This project aims to augment our survey-analysis workflow by applying modern AI techniques to free-text feedback. Instead of manually reviewing thousands of open-ended comments, stakeholders will be able to query an AI system and receive context-aware summaries, sentiment trends, and actionable insights.

## 2. Problem Statement
- We collect a variety of data (course information, user profiles, attendance, completions, and course evaluation from surveys), but most of our qualitative insights are buried in multiple Excel files.
- Manually reading and coding hundreds or thousands of free-text responses is time-consuming, error-prone, and prone to human bias.
- Without semantic context, business users may overlook patterns or misinterpret feedback.

## 3. Proposed Solution
1. **Sentiment Analysis**  
   - Automatically classify each free-text comment as positive, neutral, or negative.  
   - Surface recurring themes (e.g., common praise or complaints) across courses.  

2. **Retrieval-Augmented Generation (RAG)**  
   - Store every free-text response (and related metadata such as course ID, user ID, attendance status, completion rate) in a relational database.  
   - When a stakeholder asks a natural-language query, e.g., “Which courses had the most negative feedback about workload?”, the system performs a semantic lookup to find the most relevant comments, then uses an LLM to generate a concise, context-aware summary.

3. **Data Privacy & Governance**  
   - Enforce strict privacy controls at every stage (see Section 6).  
   - Ensure all personal identifiers are pseudonymised, and only authorised users have access.  

## 4. Key Benefits
- **Faster Insight Generation**: Business users get instant sentiment dashboards and AI-generated summaries.  
- **Informed Decision-Making**: Stakeholders can ask pointed questions (e.g., “Why did completion rates drop in Course X?”) and rely on AI-grounded answers.  
- **Reduced Bias**: Automated sentiment and semantic search reduce the chance that a single loud voice skews interpretation.  
- **Scalable Architecture**: Once proven, the pipeline can be extended to other surveys or integrated with additional data sources (e.g., LMS logs).

## 5. Technical Approach

### 5.1 High-Level Architecture
1. **Database (PostgreSQL + pgvector)**  
   - Store structured data tables:  
     - `courses` (course_id, course_name, course_type, …)  
     - `users` (user_id (pseudonymised), demographic fields…)  
     - `responses` (response_id, user_id, course_id, submitted_at, …)  
     - `response_texts` (text_id, response_id, question_key, text)  
   - Store vector embeddings for each text chunk using the `pgvector` extension, enabling fast semantic similarity search.

2. **AI Processing Worker**  
   - Background job runner (e.g., Celery, RQ, or Prefect) that:  
     1. **Cleans & Chunks** each free-text comment (200–300 tokens per chunk).  
     2. **Generates Embeddings** via the OpenAI Embeddings API (e.g., `text-embedding-ada-002`).  
     3. **Runs Sentiment Analysis** using HuggingFace’s RoBERTa model (fine-tuned for sentiment).  
     4. **Stores** embeddings and sentiment scores back in the database.

3. **Query API (FastAPI)**  
   - Exposes endpoints for:  
     - **Sentiment Dashboard**: Aggregates positive/negative/neutral counts per course or module.  
     - **RAG Q&A**:  
       1. Accepts a plain-language question (e.g., “Which courses with low completion rates had negative feedback on workload?”).  
       2. Uses vector similarity search (`pgvector`) to fetch top-K relevant text chunks.  
       3. Assembles a prompt (including retrieved snippets) and calls the OpenAI Chat Completion API (e.g., `gpt-o4-mini`) to generate a concise answer.  

4. **Presentation Layer**  
   - A simple front-end or dashboard (e.g., Streamlit, a static React page, or a BI tool) that:  
     - Displays aggregated sentiment graphs (e.g., positive vs. negative trends).  
     - Provides a text box for ad-hoc RAG queries.  

### 5.2 Tooling & Frameworks
- **Data Preprocessing & Evaluation**: Scikit-learn (for any custom model training or evaluation).  
- **Text Cleaning**: NLTK or spaCy (tokenisation, lemmatisation, stop-word removal).  
- **LLM Orchestration**: LangChain or LlamaIndex (optional, to standardise prompt templates and retrieval chains).  
- **Workflow Orchestration**: Apache Airflow, Prefect, or Dragster (to schedule ETL, AI-processing, and monitoring).  
- **ORM / Database Interaction**: SQLAlchemy (for Python scripts to read/write PostgreSQL).  
- **Web Framework**: FastAPI (to expose REST endpoints).  
- **Containerisation**: Docker (each service in its own container, plus `docker-compose` for local development).

---

## 6. Data Privacy & Governance
We will implement a layered, “defence-in-depth” approach:

1. **Data Minimisation**  
   - Only store fields essential for analysis:  
     - Pseudonymised user ID  
     - Course ID  
     - Free-text responses (no names, email addresses)  
     - Timestamps (submission date only)  
   - Other optional fields (e.g., attendance, completion status) are included only if they add direct analytical value.

2. **Pseudonymisation & Anonymisation**  
   - Replace any direct identifiers (e.g., student numbers) with hashed or randomly generated IDs.  
   - Remove or redact any PII within free-text entries (if a student types “My name is John,” the ETL pipeline will detect and mask “John”).

3. **Encryption**  
   - **In Transit**: All API calls (including Qualtrics → Google Sheets → ETL) use TLS/HTTPS.  
   - **At Rest**: PostgreSQL data files and automated backups are encrypted on disk.  

4. **Access Controls & Least Privilege**  
   - Database credentials are stored in a secure vault (e.g., AWS Secrets Manager, Azure Key Vault).  
   - Application and worker containers authenticate with minimal privileges—only allowed to read/write necessary tables.  
   - Admin-only roles can access raw data; analysts see only aggregated results.  

5. **Audit Logging & Monitoring**  
   - Every read/write to the `responses` table generates an audit log entry (`user`, `timestamp`, `operation`).  
   - API access logs record the origin IP, endpoint called, and timestamp—reviewed weekly.  

6. **Data Retention & Deletion Policies**  
   - **Active Data**: Keep survey data for the current academic year in the live database.  
   - **Archival**: Move older free-text entries (12+ months) to an encrypted archive (e.g., AWS S3 with server-side encryption) and delete them from the live system.  
   - **User Requests**: If a student requests deletion of their own comment, we will locate and permanently remove it (or replace it with “User requested redaction”).

7. **Regulatory Compliance**  
   - Conducted an initial Data Privacy Impact Assessment (DPIA) to identify potential risks.  
   - Aligns with internal policies and external regulations (e.g., GDPR, if applicable).  
   - Quarterly reviews by the Privacy & Ethics Committee.
