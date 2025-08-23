# Frontend MVP Plan – Custom Single‑Page Chat Interface

Status: Draft  
Version: 1.0  
Date: 2025-08-10  
Audience: Engineers, Product, Governance, Supporting LLMs

---
## 0. Purpose
Define a minimal, production-capable single page app (React/Vite) integrating with existing backend RAG components (e.g. `RAGAgent`, `AsyncSQLTool`, `VectorSearchTool`, `AustralianPIIDetector`, `AnswerGenerator`) to deliver governed natural-language analytics with role-aware evidence and feedback capture. Provides LOW-LEVEL build instructions so an autonomous LLM can implement safely.

Guiding Principles:
- Server Authority: Client never fabricates classification/evidence. Everything sourced from backend JSON / SSE events.
- Privacy First: All outbound user text passes through `AustralianPIIDetector` before classification or tool use.
- Deterministic Answer Shape: Stable contract enabling safe UI rendering & future automation tests.
- Progressive Enhancement: Start with non-stream POST; add SSE once stable.

---
## 1. Scope & Explicit De‑scope
Scope (MVP): Submit query, receive answer (SQL / VECTOR / HYBRID), show route + confidence band, conditional evidence panels (role), feedback (rating+comment), correlation ID + timings, basic error surface, minimal session context (last N pairs in memory only).

De‑scoped: Clarification UI automation, reranking, dashboards, multi-session management, transcript export, caching, prompt editing, teams integration.

---
## 2. User Roles
Same role set reused from backend: `exec_view`, `analyst_full`, `admin_ops`, `compliance_audit` (latter won’t use chat in MVP). Evidence row/snippet limits enforced server side; client just hides collapsed sections when absent.

---
## 3. Backend Integration Points (Concrete)
Module / Class | Purpose | Call Pattern
---------------|---------|------------
`rag.core.agent.RAGAgent` | Orchestration | `agent = await create_rag_agent()` or manual: `agent = RAGAgent(); await agent.initialize(); state = await agent.ainvoke({"query": q, "session_id": sid})`
`rag.core.text_to_sql.sql_tool.AsyncSQLTool` | SQL generation & execution | Indirect via agent (do NOT call directly in API handler for MVP)
`rag.core.vector_search.vector_search_tool.VectorSearchTool` | Semantic feedback search | Indirect via agent
`rag.core.privacy.pii_detector.AustralianPIIDetector` | PII anonymisation | `detector = AustralianPIIDetector(); await detector.initialise(); result = await detector.process_text(query)` (create helper wrapper)
`rag.core.synthesis.answer_generator.AnswerGenerator` | Answer synthesis | Indirect via agent (already used in agent)
`rag.core.synthesis.feedback_collector` | Feedback persistence (to be wrapped) | Provide simple API calling underlying collector (if implemented) else stub DB insert

NOTE: Only the agent should coordinate tool invocation to preserve routing logic & future hybrid enhancements. API layer shapes response + filters evidence based on role.

---
## 4. API Contract (Authoritative JSON Schema)
Endpoint: `POST /api/chat`
Request: `{ "query": string, "session_id": optional string }`
Response JSON fields (non-stream first iteration):
Field | Type | Description | Source
------|------|-------------|-------
`query_id` | string(uuid4) | New per request | `uuid.uuid4()` in handler
`route` | enum(SQL|VECTOR|HYBRID|CLARIFICATION_NEEDED) | Classification result | `AgentState.classification`
`confidence` | string(HIGH|MEDIUM|LOW) | Classification confidence | `AgentState.confidence`
`answer` | string | Natural language answer | `AgentState.final_answer`
`sql` | object/null | Raw structured SQL tool result | `AgentState.sql_result`
`vector` | object/null | Raw structured vector result | `AgentState.vector_result`
`evidence` | object | Normalised evidence (see Section 5) | Derived
`timings_ms` | object | Stage timings (classify/sql/vector/synth/total) | Handler instrumentation + `AgentState.processing_time`
`model_versions` | object | LLM + embed id (static from settings now) | settings / llm object
`anonymised` | bool | True if detector changed text OR verified clean | detector result
`role` | string | User role | Auth middleware
`correlation_id` | string | Propagated for logs | generate UUID
`requires_clarification` | bool | If agent flagged follow-up needed | `AgentState.requires_clarification`
`error` | string/null | Error description | error path

Endpoint: `POST /api/feedback`
Request: `{ "query_id": string, "rating": int(1-5), "comment": optional string }`
Response: `{ "status": "ok" }`

Health: `GET /api/health` → `{ "status": "ok", "db":true, "pii":true }`

---
## 5. Evidence Normalisation Layer (Server)
Create helper `normalize_evidence(state: AgentState, role: str) -> Dict[str, Any]` in new module `rag/rag/formatter/evidence_formatter.py` (path suggestion):
Algorithm:
1. Start `evidence = {"sql": None, "vectors": []}`.
2. If `state["sql_result"]` and `success`, derive:
	 - rows = first N based on role (exec_view: 5, others: 20) from `sql_result.result` if iterable.
	 - `row_count = len(sql_result.result)` (fallback 0)
	 - `truncated = row_count > len(rows)`
	 - Include original query text as `query` (sanitise line breaks).
3. If `state["vector_result"]` present:
	 - Map each item with keys: `text`, `score`, plus optional metadata if available (sentiment, agency). Limit to K=5.
4. Return object; never include raw PII (assume already anonymised by the pipeline).
5. Unit test: ensure truncation flags / lengths correct.

---
## 6. FastAPI Implementation Steps (Low-Level)
Add new file (suggest): `src/rag/api/endpoints/chat.py` performing:
1. Dependency injection: global `AGENT: RAGAgent | None`.
2. Startup event: instantiate agent if None `agent = RAGAgent(); await agent.initialize()`; instantiate `AustralianPIIDetector` once (`await detector.initialise()`).
3. POST `/api/chat` handler:
	 - Parse body (Pydantic model `ChatRequest(query: str, session_id: Optional[str])`).
	 - `raw_query = req.query.strip()`; reject empty.
	 - `pii_result = await detector.process_text(raw_query)` (implement `process_text` wrapper calling `analyzer.analyze` + anonymiser).
	 - `query_for_agent = pii_result.anonymised_text`.
	 - `start = time.perf_counter()`; call `state = await agent.ainvoke({"query": query_for_agent, "session_id": req.session_id or short_uuid()})`.
	 - Build timings: `total = (time.perf_counter()-start)*1000` plus stage timings (initially only `total`; others stub 0 until instrumented inside agent nodes — add TODO comments referencing `_classify_query_node`, `_sql_tool_node`, `_vector_search_tool_node`, `_synthesis_node`).
	 - Evidence = `normalize_evidence(state, role)`.
	 - Compose response JSON exactly per Section 4.
	 - Log structured dict.
4. POST `/api/feedback` handler:
	 - Validate rating 1–5; sentiment optional (if `feedback_collector` has helper, else stub insert into `rag_user_feedback` table via existing DB connector in `src/db/db_connector.py`).
	 - Return status ok.
5. GET `/api/health` simple checks: DB readonly query (SELECT 1), `detector._initialised` flag, maybe agent not None.

Instrumentation TODO: Add per-stage timestamps by wrapping internal agent node methods via monkey patch or (preferred) extend `RAGAgent` with optional callback to collect node duration; safe to defer.

---
## 7. React Frontend Structure (Filesystem)
```
frontend/
	package.json
	vite.config.ts
	src/
		main.tsx
		App.tsx
		context/
			ChatContext.tsx
			ConfigContext.tsx
		components/
			ChatContainer.tsx
			MessageList.tsx
			Message.tsx
			Composer.tsx
			RouteBadge.tsx
			EvidencePanel.tsx
			FeedbackBar.tsx
			ErrorToast.tsx
			FooterMeta.tsx
			PrivacyBanner.tsx
		hooks/
			useChat.ts
			useConfig.ts
			useRetry.ts
		types/
			api.ts
			app.ts
		utils/
			sanitize.ts
			validation.ts
```

Key Types (api.ts):
```typescript
export interface ChatResponse { 
  query_id: string; 
  route: 'SQL'|'VECTOR'|'HYBRID'; 
  confidence: 'HIGH'|'MEDIUM'|'LOW'; 
  answer: string; 
  evidence: Evidence; 
  timings_ms: Record<string, number>; 
  role: string; 
  correlation_id: string; 
  anonymised: boolean; 
  error?: string; 
}

export interface Evidence { 
  sql?: { 
    query: string; 
    rows: any[][]; 
    row_count: number; 
    truncated: boolean 
  } | null; 
  vectors: { 
    text: string; 
    score: number; 
    sentiment?: string; 
    agency?: string 
  }[] 
}
```

**State Management Architecture**:
```typescript
// Context + Reducer for predictable state updates
interface ChatState {
  messages: Message[];
  currentSession: string;
  isLoading: boolean;
  error: ChatError | null;
}

type ChatAction = 
  | { type: 'SEND_MESSAGE'; payload: string }
  | { type: 'RECEIVE_MESSAGE'; payload: ChatResponse }
  | { type: 'SET_ERROR'; payload: ChatError }
  | { type: 'CLEAR_ERROR' };

const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  // Immutable state updates with type safety
};
```

Hook `useChat` responsibilities:
1. Maintain `messages` array `{ id, role: 'user'|'assistant', content, route?, confidence?, evidence?, status }`.
2. `send(query)` pushes pending user + assistant placeholder then POST fetch with retry logic.
3. On response: update assistant message fields; handle error path (status = error).
4. SSE upgrade later: if `Accept: text/event-stream` chosen, process events (meta/token/evidence/done) updating same message object.
5. **Error Recovery**: Exponential backoff retry with circuit breaker pattern.

---
## 8. Answer Rendering Template (Client)
```
<Answer>
	<div class="summary">{answer}</div>
	<RouteBadge route={route} confidence={confidence} />
	{role === 'analyst_full' && <EvidencePanel evidence={evidence} />}
	<FeedbackBar queryId={query_id} />
	<FooterMeta id={correlation_id} totalMs={timings_ms.total} anonymised={anonymised} />
</Answer>
```

---
## 9. Command Parsing (Client Minimal)
Parser maps leading slash commands before submit:
| Command | Action |
|---------|-------|
| /clear | Clear local message state (doesn’t call server) |
| /route sql|vector|hybrid | Add header `X-Route-Override` to chat POST (server may honour) |
| /mode full|summary | Add header `X-Mode` for potential future gating |
| /feedback <1-5> [comment] | Direct feedback call; skip if no previous answer |

Server MUST validate all overrides; client hints only.

---
## 10. Logging & Metrics (Server MVP)
Structured Log JSON fields: `timestamp, level, correlation_id, query_id, role, route, confidence, total_ms, row_count, vector_k, anonymised, error`.

Prometheus initial metrics (extend later):
- `rag_query_total{route}`
- `rag_query_latency_seconds_bucket{route}`
- `rag_feedback_rating_total{rating}`
- `rag_query_error_total{stage}` (stage=classification|sql|vector|synthesis|api)
- `rag_classification_confidence_bucket`

---
## 11. Testing (Granular Checklist)
Layer | Test | Implementation Detail
------|------|----------------------
Unit | Evidence truncation | Feed synthetic sql_result with 30 rows, expect exec_view returns 5 |
Unit | PII detector wrapper | Inject ABN string, expect token replaced `[ABN]` |
Unit | Route badge mapping | Confidence to colour mapping stable |
Unit | Feedback validation | Reject rating 0 or 6 |
Integration | /api/chat SQL path | Query containing "count" triggers SQL; sql section present |
Integration | /api/chat Vector path | Query containing "feedback" returns vector evidence list |
Integration | Role filtering | Simulate exec_view header vs analyst_full; row counts differ |
Integration | Feedback lifecycle | Submit feedback then verify DB row (or mock) |
Performance | 20 concurrent hybrid queries | p95 within SLO (record) |
Compliance | Synthetic PII queries | None bypass anonymisation (log anonymised:true) |

---
## 12. Phased Build Plan (Context Partitioning for LLMs)
Purpose: Enable incremental loading of ONLY the active phase + global sections (0–5) to minimise context token usage. Each phase lists: Objectives, File Touchpoints, Minimal Inputs, Exit Criteria, and Next-Phase Handoff Payload (recommended summary the LLM should emit for continuity).

Phase Overview Table:
| Phase | Title | Primary Goal | Key Risks Mitigated |
|-------|-------|-------------|---------------------|
| P0 | Skeleton & Contracts | Establish API stubs + types + logging baseline | Scope drift before contract freeze |
| P1 | Core Orchestration | Real agent integration + PII + evidence normaliser | Data / privacy mismatch |
| P2 | UI Foundations | React scaffold + chat send/receive + route badge | Frontend architectural rework |
| P3 | Evidence & Feedback | Role gating + evidence panels + /feedback | Leakage of oversized evidence |
| P4 | Observability & Quality | Metrics, structured logs, core tests | Blind spots in prod triage |
| P5 | Hardening & Compliance | Error taxonomy, PII regression tests, rate limiting | Undetected compliance regressions |
| P6 | Streaming & Performance | SSE channel + perf tuning + basic load test | Latency SLO breach lingering |

---
### Phase P0 – Skeleton & Contracts
Objectives:
- Create FastAPI app module with placeholder `/api/chat`, `/api/feedback`, `/api/health` returning stub JSON matching Section 4 schema fields (dummy values).
- Add shared Pydantic models: `ChatRequest`, `ChatResponse` (initial minimal fields), `FeedbackRequest`.
- Create `frontend/` scaffold (Vite + TS) with `useChat` hook performing POST to stub.
- Define TypeScript `ChatResponse` & `Evidence` interfaces (Section 7 types) EXACTLY; keep extra fields optional.
File Touchpoints:
- `src/rag/api/__init__.py` (new) & `src/rag/api/endpoints/chat.py` (stubs)
- `frontend/package.json`, `frontend/src/types/api.ts`, basic components.
Minimal Inputs Required for LLM:
- Sections 0,4,7 of this doc + this phase block.
Exit Criteria:
- `POST /api/chat` returns HTTP 200 with fixed `query_id`, `route`, `answer`, `timings_ms.total`.
- Frontend renders answer text & route badge placeholder.
Handoff Payload (Emit Summary): `contract_hash` (hash of response schema field names), endpoint paths, TS interface checksum.

### Phase P1 – Core Orchestration
Objectives:
- Instantiate `RAGAgent` & `AustralianPIIDetector` at startup.
- Implement PII wrapper `process_text(query)->(anonymised_text, changed:bool)`.
- Replace stub logic with `agent.ainvoke` integration; map `AgentState` to `ChatResponse`.
- Implement `normalize_evidence` (Section 5) returning shape `{ sql, vectors }`.
File Touchpoints:
- `src/rag/api/endpoints/chat.py` (real handler)
- `src/rag/formatter/evidence_formatter.py` (new)
Minimal Inputs:
- Sections 3–6 + Phase P0 summary.
Exit Criteria:
- Real data flows end-to-end for sample queries (SQL keyword vs feedback keyword) with classification fields.
- PII detection flag toggles `anonymised` correctly for a synthetic ABN.
Handoff Payload: counts (rows returned, vectors length), anonymisation flag stats.

### Phase P2 – UI Foundations
Objectives:
- Implement `ChatContainer`, `MessageList`, `Composer`, `RouteBadge`.
- `useChat.send` handles pending state and error state with retry logic.
- Basic theming + accessibility roles (landmark regions).
- **State Management**: Implement Context + Reducer pattern for scalable state updates.
- **Performance**: Add React.memo for expensive components, code splitting for evidence panels.
File Touchpoints:
- `frontend/src/components/*.tsx`, `frontend/src/hooks/useChat.ts`.
- `frontend/src/context/ChatContext.tsx` (new state management).
- `frontend/src/utils/validation.ts` (input sanitization helpers).
Minimal Inputs:
- Sections 7–9 + P1 summary.
Exit Criteria:
- Single query lifecycle visible, correlation ID & timing shown.
- Error from backend (inject manually) renders distinct styling with retry option.
- State updates are predictable and testable via reducer pattern.
- Components are properly memoized to prevent unnecessary re-renders.
Handoff Payload: component registry list + prop signatures + state management patterns.

### Phase P3 – Evidence & Feedback
Objectives:
- Add `EvidencePanel` with collapsible SQL (table) + vector snippet list.
- Role gating test: simulate `exec_view` vs `analyst_full` header.
- Implement `FeedbackBar` posting to `/api/feedback` & optimistic UI.
- **Component Optimization**: Implement lazy loading for evidence components and memoization.
- **Error Handling**: Add comprehensive error boundary with categorized error display.
File Touchpoints:
- `frontend/src/components/EvidencePanel.tsx`, `FeedbackBar.tsx`
- `frontend/src/components/ErrorBoundary.tsx` (new global error handling)
- Backend feedback handler (persist or stub).
Minimal Inputs:
- Sections 5,8,9 + P2 summary.
Exit Criteria:
- Truncation flags surface; feedback stored & returns 200.
- Evidence panels load lazily and render efficiently for large datasets.
- Error boundary catches and displays user-friendly error messages.
- Optimistic UI updates provide immediate feedback for user actions.
Handoff Payload: evidence shape sample, feedback success ratio, error handling patterns.

### Phase P4 – Observability & Quality
Objectives:
- Add Prometheus metrics (query total, latency histogram, confidence bucket).
- Structured logging JSON with required fields.
- Implement unit tests for evidence truncation & PII replacement.
- Basic integration test for `/api/chat` 3 routes.
File Touchpoints:
- `src/rag/api/metrics.py` (optional helper), test modules under `tests/`.
Minimal Inputs:
- Sections 8,10,11 + P3 summary.
Exit Criteria:
- Metrics endpoint scrape returns non-zero counters after sample queries.
- All tests pass in CI.
Handoff Payload: metric names list, test counts.

### Phase P5 – Hardening & Compliance
Objectives:
- Implement rate limiting (Redis token bucket placeholder or in‑memory fallback) at `/api/chat`.
- Error taxonomy mapping exceptions to stable `error` codes.
- Add synthetic PII regression suite (ABN, ACN, TFN, Medicare) ensuring anonymisation.
- Add security headers (CSP, HSTS) via middleware.
File Touchpoints:
- `src/rag/api/middleware/security.py`, `rate_limiter.py`, tests.
Minimal Inputs:
- Sections 4,8,10 + P4 summary.
Exit Criteria:
- All synthetic PII tests green; rate limit returns 429 after threshold.
Handoff Payload: list of error codes + rate limit config.

### Phase P6 – Streaming & Performance
Objectives:
- Implement SSE variant at `/api/chat/stream` emitting events `meta`, `token`, `evidence`, `done`.
- Client SSE consumption path updating existing message (feature flag `ENABLE_SSE`).
- Load test 20 concurrent hybrid queries; record p95.
- **Performance Optimization**: Virtual scrolling for long conversations, component memoization.
- **Monitoring**: Add frontend performance metrics (FCP, LCP, TTI).
- **Configuration**: Environment-aware feature flags and configuration validation.
File Touchpoints:
- `src/rag/api/endpoints/chat_stream.py`, frontend SSE handling in `useChat.ts`.
- `frontend/src/hooks/useConfig.ts` (configuration management).
- `frontend/src/utils/performance.ts` (performance monitoring helpers).
Minimal Inputs:
- Sections 7,8 + P5 summary.
Exit Criteria:
- SSE path functional; fallback still works; latency SLOs met.
- Virtual scrolling handles 1000+ messages without performance degradation.
- Configuration is type-safe and validated at startup.
- Frontend performance metrics are captured and reportable.
Handoff Payload: measured p95s & streaming enable flag state, performance benchmarks.

LLM Usage Guidance:
- When implementing a phase: load Sections 0–5 (global contracts) + the specific phase block + previous phase handoff payload summary ONLY.
- Avoid re-reading earlier phase code unless modifying; rely on handoff payload.

Progression Guard:
- Do not start P(n+1) until Pn exit criteria satisfied & handoff payload recorded in commit message.

Failure Handling:
- If exit criteria fail, emit remediation plan (diff summary + root cause) before retry.

---
## 13. Risks & Mitigations
Risk | Mitigation
-----|-----------
PII detector latency | Single instance reused; initialise at startup; async batch future optimisation
Route misclassification | Provide `/route` override header path; log override usage
Overfetch SQL rows | Enforce server truncation before response; never send excess to client
Frontend state inconsistency | Single source-of-truth messages array + immutable updates
Low feedback volume | Inline stars always visible + ephemeral toast asking for rating after answer

---
## 14. Acceptance Criteria
- API endpoints respond per schema; JSON contract validated in tests.
- Evidence truncation & role gating verified by tests.
- No PII leakage in synthetic ABN/ACN/TFN test set.
- p95 latency SLOs (SQL ≤4s, Vector ≤6s, Hybrid ≤12s) in performance test.
- Feedback entries persisted with correct query_id linkage.
- Governance sign-off on logging fields.

---
## 15. Post-MVP Backlog
SSE streaming, clarification automation, reranking, caching layer, transcript export, dashboards, Teams integration, prompt version registry UI, conversation persistence store.

---
## 16. Change Log
2025-08-10: Replaced OpenWebUI-centric MVP with custom single-page app plan including low-level integration instructions (v1.0).

---
End of custom SPA Frontend MVP plan.

