# User Interface Strategy – OpenWebUI Production Interface Plan

Status: Planning (Transitioning OpenWebUI from exploratory to production interface)  
Decision Date: 2025-08-10  
Intended Audience: Executives, Analysts, Learning Programme Owners, Platform Engineers  
Version: 0.2 (scope updated to production candidate)

---
## 0. Executive Summary (Scope Adjustment)
OpenWebUI will serve as the primary production user interface enabling executives and analysts to query structured + unstructured learning data via the hybrid RAG agent. This plan formalises production-grade requirements: security (SSO, RBAC), auditability, performance SLOs, reliability, data governance, and controlled change management while retaining rapid iteration for retrieval and prompting.

Key Production Non‑Functional Targets (initial draft):
- Availability (business hours): 99.5% (Phase 1), moving to 99.9% (Phase 3).  
- p95 Latency Targets: SQL route ≤ 4s, Vector ≤ 6s, Hybrid ≤ 12s.  
- Error Budget: <2% failed queries (excludes user cancellations).  
- Data Protection: 100% queries anonymised pre-LLM; 0 known PII leakage incidents (auto gating in CI).  
- Audit Retention: 13 months rolling (configurable) for governance review.  
- Recovery: RTO 4h / RPO 1h (initial); optimise later.

---
## 1. Rationale & Positioning
OpenWebUI is adopted as the primary production interface (Phase 1) delivering:
- Executive self‑service analytics (natural language)  
- Analyst deep dives (SQL + vector evidence)  
- Governance visibility (audit / feedback / routing confidence)  
- Feedback loop for continuous model & prompt optimisation  

Rationale versus custom SPA (short term): Faster time‑to‑value, leverage existing chat interaction model, reduce initial front‑end engineering cost. A future custom UI or Teams integration can layer on top while OpenWebUI remains the stable core until parity is reached.

Production Constraints:
- APP compliance & Australian PII protection remain inviolable.  
- Auth via enterprise SSO (OIDC / Azure AD) – no anonymous access.  
- RBAC: roles = exec_view (high-level summarised results), analyst_full (detailed evidence), admin_ops (config & metrics), compliance_audit (read-only logs).  
- Observability & traceability required for every query (correlation_id).  
- Change management: prompt / routing rule changes versioned + reviewed (two-person approval for high-risk changes).  

---
## 2. High-Level Architecture
```
Browser (OpenWebUI) ──> OpenWebUI Server ──> /chat (FastAPI RAG Facade) ──> LangGraph Agent
																										 │
																										 ├─ Text-to-SQL (read-only DB)
																										 ├─ Vector Search (pgvector)
																										 └─ PII / Sentiment / Synthesis / Feedback
```

Return Path: Agent → structured JSON (role-aware filtering) → Response adapter → Markdown sections / future evidence side panel → OpenWebUI render.

Feedback Path: Inline action (/feedback or button in future plugin) → /feedback → stored (with role, latency, route, confidence_band) → analytics dashboards.

Audit & Telemetry Path: Each request emits structured log + metrics event → central store (e.g. Loki / ELK + Prometheus) → governance dashboards.

---
## 3. Phased Implementation Roadmap
| Phase | Goal | Key Deliverables | Exit Criteria |
|-------|------|------------------|---------------|
| P0 | Production Baseline Hardening | SSO (OIDC), network isolation, logging redaction, RBAC scaffolding | Auth enforced, no raw PII in logs |
| P1 | Core Query & Feedback (Prod) | /chat, /feedback, role-filtered evidence, correlation IDs | p95 latency targets (draft) met; ≥95% success |
| P2 | Evidence & Role Views | Full vs summarised view separation, row truncation logic, vector snippet classification labels | Exec vs Analyst outputs validated |
| P3 | Clarification & Reliability | Low-confidence clarification, retry/backoff, circuit breakers | Misroute rate <5%; improved success metrics |
| P4 | Observability & Governance | Metrics, audit export, prompt version registry, configuration change log | Governance report template delivered |
| P5 | Performance & Relevance | Reranking, multi-query expansion, caching layer, warm pools | precision@10 +X% (define), cost per query reduced |
| P6 | Evaluation Automation | Ragas / Phoenix integration nightly + regression gates in CI | Failing metrics block deploys |
| P7 | Executive Dashboards | High-level KPI cards (usage, satisfaction, top themes) inside OWUI extension tab | Executives self-serve KPIs |
| P8 | DR & Scaling | Backup / restore scripts, horizontal scaling tests, load test results | RTO/RPO validated; scale to target concurrency |
| P9 | Continuous Optimisation | Automated prompt A/B, dynamic routing thresholds | Sustained improvement in answer quality trend |

---
## 4. API Surface (Initial)
### 4.1 /chat (POST)
Request:
```
{ "query": "What did people say about virtual sessions?", "session_id": "optional-uuid" }
```
Response (baseline mode):
```
{
	"query_id": "uuid",
	"route": "VECTOR",
	"confidence": 0.82,
	"answer_md": "## Summary...\n...",
	"timings_ms": {"classify":420, "retrieve":610, "synth":910, "total":1960},
	"model_versions": {"llm":"gpt-4o", "embed":"ada-002-v1"}
}
```
Full mode adds: `sql`, `vector_snippets[]`, `limits` object.

### 4.2 /feedback (POST)
```
{ "query_id": "uuid", "rating": 4, "comment": "Helpful insights on issues" }
```
Response: `{ "status": "ok" }`

### 4.3 /stats/feedback (GET)
Returns aggregates: avg_rating, total_feedback, sentiment_distribution, route_breakdown.

---
## 5. Answer Markdown Structure (Baseline)
```
## Summary
<concise synthesized answer>

### Route
HYBRID (confidence 0.84) | SQL rows: 3 | Vector k: 5 | Time: 2.1s

### Key Figures
| Agency | Completed |
|--------|-----------|
| Dept A | 150 |
| Dept B | 127 |

### Insights (Vector Snippets)
> "Technical difficulties in virtual delivery..." (neg, Agency A)
> "Great facilitator pacing..." (pos, Agency B)

### Next Steps
Try refining by agency or user level. For feedback: /feedback <1-5> <optional comment>

---
_Model: gpt-4o • Embeddings: ada-002-v1 • Query ID: abc123_
```

---
## 6. Security & Privacy Controls
Control | Implementation (Prod Scope)
--------|---------------------------
Authentication | OIDC (Azure AD / enterprise IdP) – short-lived access tokens; refresh via silent flow
Authorisation (RBAC) | Role claims mapped to exec_view / analyst_full / admin_ops / compliance_audit
PII Anonymisation | Mandatory pre-processing; enforcement test in CI + runtime guard raising 500 + alert on failure
Log Redaction | Structured logging (JSON) scrubs raw query; store anonymised_query & pii_entity_counts
Rate Limiting | Per user & global token bucket (redis) + burst window controls
Output Sanitisation | Markdown renderer whitelist; escape fallback for unexpected tokens
Least Privilege DB | Read-only user validated each startup; fail-fast if privilege drift
Config Isolation | Secrets via environment / vault; secrets never echoed in diagnostics endpoints
Telemetry Privacy | Metrics export excludes raw user text; only hashed IDs & classification metadata
Change Control | Prompt & routing config commits require code review + semantic version bump
Incident Response | On-call rotation doc; correlation IDs enable trace triage; severity matrix defined (appendix TBD)
Session Management | Session_id rotation after inactivity threshold; conversation truncation (N latest) to bound context risk

Additional Governance Enhancements:
- Data Retention Policy: purge or archive audit logs > retention_period (default 13 months).  
- Access Review: quarterly RBAC membership certification.  
- Threat Modelling: revisit on any new data field ingestion.

---
## 7. Clarification Flow (Phase 4)
Trigger: classification confidence < threshold (e.g. 0.55). Server returns markdown:
```
I need a bit more detail. Are you after:
A) 📊 Statistical summary
B) 💬 Feedback / comments
C) 📈 Both
Reply with A, B, or C.
```
Parser maps A→SQL, B→VECTOR, C→HYBRID (override classification & record override_reason="user_choice").

---
## 8. Feedback Lifecycle
1. User submits /feedback command.
2. API validates (1–5), anonymises comment, sentiment-scores comment.
3. Persist (query_id, rating, sentiment, timestamp, route, confidence_band).
4. Nightly job computes trend metrics (optional early: on-demand aggregation only).

---
## 9. Metrics (Phase 5)
Metric | Description (Prod KPIs)
-------|-----------------------
rag_query_total{route} | Count by route (capacity planning)
rag_query_latency_seconds_bucket | End-to-end latency histograms
rag_stage_latency_seconds_bucket{stage} | Stage-level latency decomposition
rag_classification_confidence_bucket | Confidence distribution (tune thresholds)
rag_misroute_total | Queries manually clarified / misclassified
rag_feedback_rating_total{rating} | Raw satisfaction counts
rag_feedback_avg_rating | Rolling mean satisfaction
rag_feedback_response_rate | Feedback submissions / total answers
rag_pii_entities_detected_total | Compliance monitoring trend
rag_pii_blocked_total | Queries blocked due to anonymisation failure
rag_query_error_total{stage} | Failures segmented by stage
rag_cost_tokens_total{phase} | Token usage for cost governance
rag_cache_hit_ratio | Effectiveness of caching (post Phase 5)
rag_rerank_gain | Delta in retrieval score vs baseline (Phase 5+)

---
## 10. Production Hardening Checklist
- [ ] OIDC SSO (token validation + role extraction)
- [ ] Enforced HTTPS (TLS certs managed; HSTS header)
- [ ] Web security headers (CSP, X-Frame-Options, Referrer-Policy)
- [ ] Raw query logging disabled (verified via integration test)
- [ ] PII anonymisation CI guard (fails on leak fixture)
- [ ] p95 latency dashboards (SQL / Vector / Hybrid)
- [ ] Error budget tracking panel
- [ ] SQL row & token truncation guards active
- [ ] Vector snippet length clamp applied
- [ ] Feedback endpoint schema + rate limit (5/min/user)
- [ ] Correlation ID end-to-end propagation
- [ ] Audit log retention job configured
- [ ] Backup & restore procedure documented & tested
- [ ] Prompt version registry + rollback script
- [ ] Load test report (target concurrency achieved)
- [ ] RTO/RPO drills executed (recorded outcomes)

---
## 11. Risks & Mitigations (Production Focus)
Risk | Mitigation
-----|-----------
PII leakage (runtime) | Mandatory anonymisation guard + blocked query metric + alerting
Privilege escalation (RBAC misconfig) | Startup role validation + quarterly access review
Latency SLO breach | Stage timing metrics + adaptive caching + autoscale plan
Token cost overrun | Token budget monitor + dynamic truncation + prompt compression
Model drift / answer quality regression | Nightly evaluation (Ragas) gating deploys
Misclassification (routing) | Confidence threshold tuning + clarification flow + retraining prompts
Single point failure (agent service) | Horizontal replicas + health probes + circuit breakers
Data corruption (embeddings) | Model_version tagging + checksum & periodic validation
Incident response delays | Runbook + on-call roster + synthetic canary queries

---
## 12. Evolution Path (Superseding Prior Decommission Section)
Future Scenario: A bespoke SPA or Teams app achieves full parity & adds richer dashboards.
Transition Steps:
1. Run dual-interface period (≥ 4 weeks) with traffic split & comparative quality metrics.
2. Confirm feature parity (feedback, clarification, evidence, RBAC, metrics).
3. Announce migration timeline; provide user training materials.
4. Freeze OpenWebUI feature development; security patches only.
5. Export and archive final anonymised interaction dataset.
6. Sunset OpenWebUI after adoption threshold (>90% queries via new UI).

---
## 13. Outstanding Decisions
1. Identity Provider (Azure AD vs internal IdP)?
2. Final SLO values (p95 & p99 per route)?
3. SQL row preview default (25 vs 50) & safe override rules.
4. Conversation retention policy (per session vs 7-day rolling window)?
5. Executive dashboard metric set (initial KPI list)?
6. Evaluation metric thresholds (faithfulness, relevance) to gate deploys.
7. Backup storage location & encryption standard confirmation.

---
## 14. Next Immediate Actions (Production Track – Week 1)
1. Integrate OIDC middleware & role claim mapping.
2. Implement /chat with correlation IDs + stage timing instrumentation.
3. Build markdown formatter (summary / route / metrics / evidence placeholders) with role filtering.
4. Implement /feedback (rating, comment, anonymisation, sentiment scoring) + table migration.
5. Add anonymisation enforcement test + CI gate.
6. Configure structured logging + initial Prometheus metrics exposition.
7. Draft RBAC policy document & initial role assignment list.
8. Load synthetic data & run baseline latency measurement.

---
End of production-scope OpenWebUI interface plan.

