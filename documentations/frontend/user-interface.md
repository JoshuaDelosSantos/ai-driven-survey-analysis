# User Interface Strategy – Single‑Page Chat Analytics MVP (Custom Web App)

Status: Planning (supersedes prior OpenWebUI plan)  
Decision Date: 2025-08-10  
Intended Audience: Product, Engineering, Governance, Supporting LLMs  
Version: 1.0 (First custom SPA plan)

---
## 0. Executive Summary
We will implement a lean, single-page, ChatGPT‑style web application (React + Vite + FastAPI backend) as the primary MVP interface for querying hybrid (SQL / Vector / Hybrid) learning data. OpenWebUI will NOT be used for production; all privacy, routing, evidence formatting, and feedback logic remain server-controlled. Focus: fast delivery of secure natural-language analytics with minimal surface area and clear extensibility.

Key MVP Outcomes:
- Natural language to governed analytics with route transparency
- Role-aware evidence (summary vs full detail) – may land Phase 2
- Feedback loop for continuous improvement
- Privacy-first (server anonymisation), auditable (correlation IDs) operation

Initial Non-Functional Targets (MVP):
- p95 latency: SQL ≤4s, Vector ≤6s, Hybrid ≤12s  
- Availability (business hours): 99.0% (MVP)  
- Error budget: <5% failed queries (excluding user cancel)  
- Zero confirmed PII leakage (automated CI test + runtime guard)  

---
## 1. Scope & De-scope
In Scope (MVP): Single conversation view, query submission, streamed or buffered answer, route + confidence badge, collapsible evidence (if analyst), feedback stars, basic error handling, correlation ID display.

Explicitly Deferred: Multi-conversation management, export transcripts, clarification buttons, reranking, dashboards, advanced auth claims introspection UI, prompt editing, caching layer.

---
## 2. High-Level Architecture
```
Browser (React SPA) → /api/chat (POST | SSE) → FastAPI Controller → Anonymiser → Classifier → LangGraph Agent → (SQL / Vector) → Synthesis
																															│
																															├─ Feedback (/api/feedback)
																															└─ Metrics (/metrics)
```
Response Path: Streaming events (meta → tokens → evidence → done) OR single JSON fallback.

---
## 3. Component Model (Client)
| Component | Responsibility | State Management |
|-----------|---------------|------------------|
| ChatContainer | Layout, scroll, holds message list | Context provider for chat state |
| MessageList / Message | Renders user & assistant messages, states (pending/streaming/done/error) | Memoized rendering for performance |
| Composer | Multiline input + send; handles debounce + submit | Local form state with validation |
| RouteBadge | Displays route (SQL / VECTOR / HYBRID) + confidence band | Props-based, memoized |
| EvidencePanel | Collapsible SQL table + vector snippets (analyst role) | Lazy-loaded with code splitting |
| FeedbackBar | 1–5 stars + optional comment modal (post-answer) | Optimistic updates with retry |
| ErrorToast | Ephemeral error notifications | Global error boundary integration |
| FooterMeta | Correlation ID + latency + anonymised flag | Props-based metadata display |
| PrivacyBanner | APP compliance notice | Static component |

### State Architecture
```typescript
// Context + Reducer pattern for scalable state management
interface ChatState {
  messages: Message[];
  currentSession: string;
  isLoading: boolean;
  error: ChatError | null;
}

// Environment-aware configuration
interface AppConfig {
  apiUrl: string;
  enableSSE: boolean;
  maxRetries: number;
}
```

---
## 4. Backend Endpoints
### POST /api/chat
Request: `{ "query": "How many users completed courses in each agency?", "session_id": "optional" }`
Streaming (SSE) events: `meta`, multiple `token`, optional `evidence`, `done`.
Non-stream fallback JSON:
```
{
	"query_id": "uuid",
	"route": "SQL",
	"confidence": 0.91,
	"confidence_band": "HIGH",
	"answer": "...summary...",
	"evidence": {"sql": {"query": "SELECT ...", "rows": [["Agency A",150]], "truncated": true}, "vectors": []},
	"timings_ms": {"classify":320,"sql":180,"synth":640,"total":1180},
	"model_versions": {"llm":"gpt-4o","embed":"ada-002-v1"},
	"anonymised": true,
	"role": "analyst_full"
}
```

### POST /api/feedback
Body: `{ "query_id": "uuid", "rating": 4, "comment": "Helpful" }` → `{ "status": "ok" }`

### GET /api/health
Readiness (DB read-only, anonymiser loaded, LLM provider reachable).

### (Optional Later) GET /api/chat/{id}
Polling fallback if SSE unsupported.

---
## 5. Data Contracts (Evidence)
`evidence.sql.rows`: first N (role-based; exec_view default 5, analyst 20). Include `row_count` & `truncated` boolean.  
`evidence.vectors`: array of `{ text, sentiment, agency, score, idx }` capped (default 5).  
Future extension: `evidence.disclaimers`, `evidence.metrics`.

---
## 6. State & Data Handling
**State Management Strategy**: Context + Reducer pattern for maintainable state updates and predictable data flow.

```typescript
// Chat state with optimistic updates
const ChatStateProvider: React.FC = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  
  const sendMessage = useCallback(async (query: string) => {
    // Optimistic update + error recovery
  }, []);
  
  return (
    <ChatContext.Provider value={{ state, sendMessage }}>
      {children}
    </ChatContext.Provider>
  );
};
```

**Session Management**: Minimal in-memory array of messages. Each assistant message life cycle: pending → streaming → done (or error). No persistent storage MVP. Context passed to backend limited to last N (default 3) anonymised user+assistant pairs.

**Performance Optimizations**: 
- Memoized components for evidence panels
- Virtual scrolling for long conversations (future)
- Code splitting for non-critical components

---
## 7. Streaming Strategy
Phase 0: Non-stream single response (fast to build).  
Phase 1: Enable SSE; server sends meta first (route/confidence), then incremental `token` events, culminating with `evidence` & `done` to avoid flashing layout.

Event Format (SSE):
```
event: meta
data: {"query_id":"...","route":"HYBRID","confidence":0.82}

event: token
data: {"delta":"partial text"}

event: evidence
data: {"sql":{...},"vectors":[...]}

event: done
data: {"final":true}
```

---
## 8. Security & Privacy (MVP)
| Control | Implementation | Frontend Considerations |
|---------|---------------|------------------------|
| Authentication | Reverse proxy OIDC (Azure AD) injecting `X-User-Role`, `X-User-Id` | Token validation on route changes |
| Authorisation | Backend enforces role gating for evidence sizes | UI adapts based on user role |
| PII Protection | Server anonymises prior to classification; rejects if failure | Client-side input sanitization (defense in depth) |
| Logging | Store anonymised_query_hash, route, confidence, timings, NO raw text | Frontend error tracking with sanitized data |
| Rate Limiting | Simple Redis token bucket (10 req / 60s / user) | UI feedback for rate limit status |
| CSP | `default-src 'self'`; disallow inline scripts | Strict CSP with nonce-based inline styles |
| Transport | HTTPS only; HSTS enabled | Secure cookie handling |
| Error Sanitisation | Mask internal traces; user-facing code categories | Global error boundary with safe error display |
| Session | Ephemeral; context trimmed to last N; no localStorage PII | Session storage for non-PII UI state only |

**Configuration Management**:
```typescript
// Type-safe environment configuration
const config = {
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  enableSSE: import.meta.env.VITE_ENABLE_SSE === 'true',
  maxRetries: parseInt(import.meta.env.VITE_MAX_RETRIES || '3')
};
```

---
## 9. Non-Functional Targets
| Axis | Target |
|------|--------|
| p95 Latency SQL | ≤4s |
| p95 Latency Vector | ≤6s |
| p95 Latency Hybrid | ≤12s |
| Availability (BH) | 99% |
| Error Rate | <5% |
| PII Leak Incidents | 0 |
| Feedback Response Rate | >25% of answered queries |

---
## 10. Feature Matrix (MVP vs Later)
| Feature | MVP | Phase 2 | Phase 3 |
|---------|-----|---------|---------|
| Chat submit + answer | ✔ | – | – |
| Evidence panel (analyst) | ✔ (collapsed) | Enhanced filters | Re-ranking indicators |
| Role-based truncation | ✔ | Adjustable UI | Dynamic by usage patterns |
| Feedback stars | ✔ | Comment modal | Adaptive follow-up survey |
| Streaming | (optional) | ✔ | Token-level analytics |
| Clarification buttons | – | ✔ | Adaptive auto-refine |
| Caching layer | – | ✔ | Smart prefetch |
| Reranking | – | – | ✔ |
| Export transcript | – | ✔ | – |
| Dashboards | – | – | ✔ |

---
## 11. Development Sequence (Recommended)
1. Scaffolding: Vite + Tailwind + basic layout.  
2. POST /api/chat (non-stream) integration + render answer.  
3. Route/confidence badge + timing extraction.  
4. Evidence panel & truncation logic (analyst vs exec stub).  
5. Feedback POST & inline UI.  
6. Logging/metrics (timings, route counts).  
7. SSE upgrade (feature flag).  
8. Harden: rate limit, CSP, anonymisation failure path.  
9. Performance test (synthetic concurrency).  
10. Polishing (error toasts, privacy banner).

---
## 12. Answer Rendering Template
```
<div class="answer">
	<p class="summary">{summary}</p>
	<div class="meta">Route: {route} • Confidence: {band} • Total: {ms}ms • ID: {query_id}</div>
	<details class="evidence"> <summary>Evidence</summary>
		{sql_table?}
		{vector_snippets?}
	</details>
	<div class="feedback">Rate: ★★★★★ (click)</div>
</div>
```

---
## 13. Metrics (Initial Set)
Prometheus: `rag_query_total{route}`, `rag_query_latency_seconds_bucket{route}`, `rag_stage_latency_seconds_bucket{stage}`, `rag_feedback_rating_total{rating}`, `rag_query_error_total{stage}`, `rag_classification_confidence_bucket`, `rag_token_usage_total{phase}`.

Frontend (optional batch): TTFAnswer, answer_length, user_cancelled.

---
## 14. Testing Strategy
| Layer | Tests |
|-------|-------|
| Unit | markdown formatter, role gating, anonymisation guard, feedback validator |
| Integration | chat end-to-end (routes), evidence truncation, feedback lifecycle |
| Performance | 20 concurrent hybrid queries within SLO |
| Security | OIDC header spoof rejection, rate limit, CSP presence |
| Compliance | PII leak synthetic cases blocked |

---
## 15. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Scope creep | Freeze MVP matrix; backlog gating review |
| Auth delays | Temporary internal API key fallback (remove before prod) |
| Latency overruns | Stage timing instrumentation early; add streaming only if needed |
| Evidence overexposure | Server enforces truncation; client only reveals allowed rows/snippets |
| Low feedback uptake | Inline prompt + tooltip reminder |
| PII regression | CI anonymisation test + runtime abort w/ alert |

---
## 16. Backlog (Post-MVP)
Clarification buttons, reranking, caching layer, transcript export, dashboards, adaptive prompt tuning, conversation persistence, Teams channel integration.

---
## 17. Acceptance Criteria (Go/No-Go)
- MVP feature list implemented & tested.
- Latency SLOs met on synthetic + sample real queries.
- No PII leakage in automated tests.
- Feedback events stored & retrievable.
- Governance sign-off on logging & privacy controls.

---
## 18. Change Log
2025-08-10: Replaced OpenWebUI production plan with custom single-page web app strategy (v1.0).

---
End of single-page chat analytics MVP plan.

