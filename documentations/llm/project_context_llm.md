# AI-Driven Survey Analysis – Consolidated LLM Context

This document is a curated, token‑efficient knowledge base for downstream LLMs. It compresses architecture, data model, workflows, privacy controls, and extension guidance so an assisting model can reason about, modify, or extend the system without rereading the entire repository. Maintain this as the single source of high‑level truth; link out to detailed READMEs when depth is needed.

---
## 0. TL;DR (High‑Impact Snapshot)
A privacy‑first hybrid RAG + analytics platform for Australian Public Service (APS) learning & evaluation data. It: 1) ingests structured (attendance, users, learning content, evaluations) and unstructured free‑text fields, 2) performs sentiment + PII anonymisation, 3) builds pgvector embeddings with rich metadata, 4) routes natural language queries through an async LangGraph agent (Text‑to‑SQL, Vector, or Hybrid), 5) synthesises answers, collects user feedback (1–5 rating + anonymised comment), and 6) enforces Australian Privacy Principles (APP) via mandatory PII detection & masking pre‑LLM. Async, test‑heavy, production‑ready foundations prepared for future FastAPI service.

---
## 1. Repository Topology (Logical View)
```
root
├─ documentations/ (architecture, governance, iterations, llm context)
├─ src/
│  ├─ rag/              # Hybrid RAG system (core intelligence)
│  │   ├─ config/       # Pydantic settings & validation
│  │   ├─ core/         # Agent, routing, privacy, synthesis, tools
│  │   ├─ data/         # Content processing & embeddings mgmt
│  │   ├─ interfaces/   # Terminal app (interactive CLI)
│  │   ├─ utils/        # DB, LLM, logging utilities
│  │   └─ tests/        # Extensive test suite
│  ├─ db/               # Table creation & data load scripts
│  ├─ sentiment-analysis/ # (Legacy -> now a reusable analyser)
│  └─ csv/              # Mock / seed data & dictionaries
├─ docker-compose.yml   # pgvector service definition
├─ requirements.txt     # Python deps
└─ README.md            # Global project overview
```

---
## 2. Domain & Data Model
Core analytical entities (simplified):
- users(user_id, user_level[L1–L6, EL1–EL2], agency)
- learning_content(surrogate_key, content_id, content_type, target_level, governing_bodies, name)
- attendance(user_id → users, learning_content_surrogate_key → learning_content, date_start, date_end, status[Enrolled|In-progress|Completed|Withdrew])
- evaluation(response_id, surrogate_key, user_id, course_end_date, delivery_type, agency, multi-choice fields, free-text fields: did_experience_issue_detail, course_application_other, general_feedback)
- rag_embeddings(embedding_id, response_id→evaluation, field_name, chunk_text (anonymised), chunk_index, embedding VECTOR, model_version, metadata JSONB{user_level, agency, sentiment, delivery_type ...}, created_at)
- sentiment (logical output; numeric probabilities per free text) – persisted separately or embedded in metadata.
- user_feedback(session_id, rating 1–5, anonymised_comment, sentiment, timestamps) [Implemented per docs]

Free-text pipeline targets exactly three evaluation columns (above). Metadata allows multi-dimensional filtering.

---
## 3. Processing Pipelines (End‑to‑End)
### 3.1 Ingestion + Embedding (Unified 6‑Stage)
Extract → Anonymise (Presidio + AU recognisers) → Sentiment (local HF RoBERTa) → Chunk (sentence/heuristic) → Embed (OpenAI or Sentence Transformers) → Store (pgvector + metadata).

### 3.2 Query Handling (LangGraph Agent)
1. Receive user natural language query.
2. PII Anonymisation (mandatory) before any classification or embedding.
3. Classification (rules pre-filter + LLM classifier) → SQL | VECTOR | HYBRID (+ confidence).
4. Conditional execution:
   - SQL: Text-to-SQL tool (schema introspection + validation + safe execution via read-only role).
   - VECTOR: Semantic retrieval (similarity + optional metadata filters + potential re-ranking planned).
   - HYBRID: Parallel SQL + VECTOR then merge context.
5. Synthesis node: Structured prompt → coherent answer (evidence, numbers, qualitative themes, limitations).
6. Feedback request: rating + optional comment (again anonymised + sentiment → analytics).

### 3.3 Feedback Analytics
Aggregations: avg rating, distribution, sentiment trend, route confidence correlation, failure patterns.

---
## 4. Core Modules & Responsibilities
| Module | Purpose | Key Artifacts |
|--------|---------|---------------|
| rag/config | Typed settings, env validation, security toggles | settings.py |
| rag/core/agent.py | LangGraph orchestrator (async) | classify → route → execute → synthesise |
| rag/core/routing | QueryClassifier (rules + LLM, confidence, fallback) | query_classifier.py |
| rag/core/privacy | Australian PII detection & anonymisation | pii_detector.py |
| rag/core/text_to_sql | Schema introspection, SQL generation, validation, execution | schema_manager.py, sql_tool.py |
| rag/core/vector_search | Embedding retrieval tool (async) | vector_search_tool.py, embeddings_manager (in data) |
| rag/core/synthesis | Answer generation & formatting | answer_generator.py |
| rag/core/conversational | Pattern-based friendly responses | handler.py |
| rag/data | ContentProcessor & EmbeddingsManager | content_processor.py |
| rag/interfaces | Terminal interactive UX | terminal_app.py, runner.py |
| rag/utils | DB connections, LLM abstraction, logging | db_utils.py, llm_utils.py, logging_utils.py |
| rag/tests | Extensive unit/integration/security tests | test_* suites |
| sentiment-analysis | Reusable SentimentAnalyser (lean) | analyser.py |

---
## 5. Privacy & Governance (Australian Focus)
Non-negotiables:
- Mandatory PII anonymisation BEFORE: classification, embedding, SQL prompt, synthesis, feedback storage.
- Australian entities: ABN, ACN, TFN, Medicare, names, emails, phones, addresses, agency identifiers.
- Read-only DB role: no data mutation; startup permission verification.
- Audit logging: queries, classification result + confidence, errors (sanitised), feedback events.
- APP Alignment: APP 3 (minimal collection), APP 6 (purpose-bound), APP 8 (anonymised-only cross-border), APP 11 (security controls).
- No raw PII leaves trust boundary; embeddings built from anonymised text only.

---
## 6. Security & Safety Controls
- SQL Guardrails: Generation → static validation (keywords blacklist) → LangChain checker → timeout & complexity caps.
- Error Handling: node-level retries, exponential backoff for transient DB/LLM failures, safe user messages.
- Logging: PII-stripped structured events (operation, latency, route, success, counts).
- Least Privilege: dedicated read-only DB user for analytical runtime.
- Model Abstraction: central llm_utils enables provider switching without leaking secrets.
- Fallback Paths: Rule-based classification fallback; hybrid default if low confidence; safe degradation if vector/SQL fails.

---
## 7. Testing Philosophy
(Reported: 150+ tests across phases.)
- Unit: classifier patterns, PII recognisers, schema summaries, embedding chunk logic, sentiment normalisation.
- Integration: end-to-end agent flows (SQL, VECTOR, HYBRID), feedback lifecycle, anonymisation before storage.
- Security: permission enforcement, SQL injection attempts, PII leakage assertions.
- Performance: latency budget (simple SQL <5s, complex hybrid <15s), memory & concurrency sanity.
- Regression: test fixtures representative of APS data distribution.

---
## 8. Configuration (Representative ENV Keys)
- RAG_DATABASE_URL / (RAG_DB_HOST, RAG_DB_PORT, RAG_DB_NAME, RAG_DB_USER, RAG_DB_PASSWORD)
- LLM_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE
- MAX_QUERY_RESULTS, QUERY_TIMEOUT_SECONDS
- ENABLE_SQL_VALIDATION=true, MAX_SQL_COMPLEXITY_SCORE
- RAG_LOG_LEVEL, RAG_DEBUG_MODE, MOCK_LLM_RESPONSES

Settings loaded via Pydantic; safe dict export hides secrets.

---
## 9. Typical Workflows
### 9.1 Initialisation
1. Load settings & validate.
2. Verify DB role is read-only.
3. Warm PII detectors + LLM client(s).
4. Instantiate agent graph (nodes registered, conditional edges compiled).

### 9.2 Embedding Ingestion (Batch)
for each evaluation row with non-empty target free-text:
  anonymise → sentiment → chunk → embed(batch) → store(metadata)

### 9.3 Query Execution (HYBRID example)
User query → anonymise → classify (HYBRID, 0.87) → parallel SQL + vector → aggregate contexts → synthesise answer (numbers + thematic cites) → solicit feedback → store anonymised feedback (with sentiment) → analytics update.

### 9.4 Feedback Analytics Query
/feedback-stats → aggregates precomputed or on-demand summarised via SQL + vector-based thematic surfacing.

---
## 10. Prompting Patterns (Summaries)
- Classification Prompt: enforces SQL|VECTOR|HYBRID taxonomy + confidence + reasoning.
- Synthesis Prompt: Includes query, abbreviated SQL results (tables → JSON-like), vector snippets (chunk_text + sentiment + metadata), instructions for citing and acknowledging uncertainty.
- Clarification Prompt: Multiple-choice disambiguation if confidence below threshold.
- Error Recovery Prompt: Guides user to rephrase with statistical vs qualitative intent cues.

(Full templates: see `architecture.md` & `src/rag/core` READMEs.)

---
## 11. Extension Guidelines (For Future LLM Assistance)
| Goal | Recommended Hook Points | Key Considerations |
|------|------------------------|--------------------|
| Add FastAPI service | new `src/api/` wrapper around agent | Preserve async; reuse classification logic |
| Add re-ranking | Introduce cross-encoder module post-retrieval | Cache model; batch score to limit latency |
| Add HyDE / query expansion | Prepend expansion node before retrieval | Ensure expansions anonymised too |
| Add dashboards | Expose metrics (Prometheus) & feedback stats | Mask PII, aggregate only |
| Swap embedding model | Update EmbeddingsManager + store new model_version | Write migration script; dual-write if needed |
| Add new free-text field | Extend content_processor + metadata schema + ingestion test | Update PII & sentiment coverage |
| Introduce caching | Layer at SQL result + vector retrieval + classification | Invalidate on schema or embedding model change |

---
## 12. Constraints & Assumptions
- Python 3.13 baseline.
- Async-first design: blocking calls wrapped or refactored.
- Only anonymised text leaves trust boundary to external LLM APIs.
- Vector dimension currently aligned with chosen embedding provider (e.g., 1536 for ada-002); must update table schema if provider changes dimensionality.
- Feedback comments may be empty; rating alone still stored.

---
## 13. Performance Targets (Current / Design)
| Stage | Target |
|-------|--------|
| PII Anonymisation | <150ms typical short query |
| Classification | <600ms (LLM) or <10ms (rule-only) |
| Simple SQL Exec | <2s |
| Vector Top-K Retrieval | <1s (indexed, k<=10) |
| Hybrid Synthesis (full) | <15s upper bound |

---
## 14. Failure Modes & Mitigations
| Failure | Mitigation |
|---------|------------|
| LLM timeout | Retry w/ reduced context; fallback to rule classification or cached result |
| DB unavailable | Graceful message; skip SQL path; vector-only answer with caveat |
| Empty vector hits | Suggest refined phrasing; show alternative metadata filters |
| Low classification confidence | Clarification prompt; default hybrid | 
| PII detection failure (rare) | Block processing; log compliance alert | 
| Embedding model change | Version tagging + backfill job | 

---
## 15. Analytics & Observability (Planned / Partial)
- Structured logs: query_id, route, confidence, latency, token counts.
- Metrics (planned): query_count{type,status}, latency histograms, active_queries gauge, feedback_rating_avg.
- Evaluation (planned): answer relevance & faithfulness multi-metric via LLM evaluator.

---
## 16. Glossary
| Term | Meaning |
|------|---------|
| HYBRID | Combined SQL + vector retrieval path |
| PII | Personally Identifiable Information (Australian + generic) |
| Sentiment Scores | Probabilities (neg, neutral, pos) from local transformer |
| Chunk | Text slice of an anonymised free-text field (usually sentence-level) |
| Metadata Filtering | Restrict similarity results by structured attributes (agency, sentiment polarity, user level, etc.) |
| Confidence | Classifier numeric (0–1) or band (HIGH/MED/LOW) for routing certainty |

---
## 17. Representative Examples
Query (SQL): "How many users completed courses in each agency?" → SQL route → GROUP BY agency → table + summary.
Query (VECTOR): "What did learners say about technical issues in virtual sessions?" → vector route → retrieve negative sentiment chunks referencing virtual delivery.
Query (HYBRID): "Which agencies have low completion but positive feedback about facilitators?" → parallel: completion stats + thematic facilitator praise → merged ranked list.

---
## 18. Risks & Future Mitigations
| Risk | Planned Mitigation |
|------|--------------------|
| Prompt drift / degraded accuracy | Versioned prompt registry + regression evals |
| Embedding store bloat | TTL or archival for stale model_version entries |
| Privacy regressions | CI test gating for PII leakage patterns |
| Over-reliance on single provider | Multi-provider abstraction already present; add automatic failover | 
| Latency under load | Introduce async pool + caching + streaming synthesis |

---
## 19. How To Use This Context (For Other LLMs)
When generating code changes:
1. Identify module boundary (see Section 4) – avoid cross-cutting edits without doc updates.
2. Enforce privacy invariant: anonymise BEFORE any LLM/embedding call.
3. Maintain async signatures; never introduce new blocking I/O inside hot paths.
4. Update this file if adding: new free-text fields, new pipeline stage, security control, or config key.
5. Add tests mirroring existing patterns (stateful agent integration + privacy assertions).
6. Preserve metadata schema stability; version if schema changes.

When answering user analytics questions:
- Determine if question is numeric/statistical, qualitative, or hybrid.
- Provide answer with provenance (table references or response_id citations where available) and note limitations.

---
## 20. Update Protocol
- Minor update: append diff summary under a new subheading at end with date stamp.
- Major architectural change: revise affected sections + increment a semantic version marker.
- Always verify tests & privacy checks pass before committing doc changes.

Version: 1.0.0 (Initial consolidated LLM context)  
Last Generated: 2025-08-10  
Maintainer: (Update with contact if required)

---
## 21. Quick Reference Indices
- Data Fields: Section 2
- Pipelines: Section 3
- Routing Logic: Sections 3.2 / 10
- Privacy Rules: Section 5
- Extension Hooks: Section 11
- Failure Modes: Section 14
- Examples: Section 17

End of context.
