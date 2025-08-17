# Frontend Phase P0 – Skeleton & Contracts Implementation

**Objective**: Establish FastAPI stub endpoints and React scaffold with exact contract compliance per `mvp1.md` Section 4.

**Context Sources Required**:
- `documentations/llm/project_context_llm.md` (Sections 1-5: topology, privacy, modules)
- `documentations/frontend/mvp1.md` (Sections 0,4,7: purpose, API contracts, React structure)
- `src/rag/core/agent.py` (AgentState TypedDict for integration reference)
- `src/rag/config/settings.py` (configuration patterns)
- `src/rag/utils/db_utils.py` (database connection patterns)

**Exit Criteria**: 
- POST /api/chat returns HTTP 200 with schema-compliant JSON
- Frontend renders answer text + route badge from real backend response
- All mandatory fields present per Section 4 contract
- CORS configured for local development

---
## Backend Implementation (FastAPI Layer)

### File 1: `src/api/__init__.py`
```python
"""
FastAPI API layer for RAG system.
Entry point for MVP frontend integration.
Separate from src/rag package to avoid import conflicts.
"""
```

### File 2: `src/api/main.py`
**Purpose**: FastAPI app factory with environment-aware CORS and route registration
**Required imports**: 
- `from fastapi import FastAPI`
- `from fastapi.middleware.cors import CORSMiddleware`
- `import os`
**Key functions**:
- `create_app() -> FastAPI`: Factory pattern
- Include routes: `app.include_router(chat_router, prefix="/api")`
- CORS middleware: `origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")` 
- Health check: DB-aware endpoint with graceful degradation
- Error handling: Global exception handler for HTTPException

**Integration points**: Import settings via `from src.rag.config.settings import get_settings`

### File 3: `src/api/schemas/chat.py`
**Purpose**: Pydantic models matching mvp1.md Section 4 contract exactly
**Required models**:
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class RouteType(str, Enum):
    SQL = "SQL"
    VECTOR = "VECTOR" 
    HYBRID = "HYBRID"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None

class SQLEvidence(BaseModel):
    query: str
    rows: List[List[Any]]
    row_count: int
    truncated: bool

class VectorEvidence(BaseModel):
    text: str
    score: float
    sentiment: Optional[str] = None
    agency: Optional[str] = None

class Evidence(BaseModel):
    sql: Optional[SQLEvidence] = None
    vectors: List[VectorEvidence] = []

class ChatResponse(BaseModel):
    query_id: str
    route: RouteType
    confidence: ConfidenceLevel
    answer: str
    evidence: Evidence
    timings_ms: Dict[str, int]
    model_versions: Dict[str, str]
    anonymised: bool
    role: str
    correlation_id: str
    requires_clarification: bool = False
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    correlation_id: str
    timestamp: float
```

### File 4: `src/api/routes/chat.py`
**Purpose**: Stub chat endpoint returning hardcoded response matching schema
**Required imports**: 
- `from fastapi import APIRouter, HTTPException`
- `from ..schemas.chat import ChatRequest, ChatResponse, RouteType, ConfidenceLevel, Evidence, ErrorResponse`
- `import uuid, time`

**Key function**: `async def chat_endpoint(request: ChatRequest) -> ChatResponse`
**Implementation**: Return hardcoded ChatResponse with:
- `query_id`: `str(uuid.uuid4())`
- `route`: `RouteType.SQL` (fixed for now)
- `confidence`: `ConfidenceLevel.HIGH`
- `answer`: `"This is a stub response for query: {request.query}"`
- `evidence`: Empty Evidence object
- `timings_ms`: `{"classify": 100, "sql": 200, "synth": 200, "total": 500}`
- `model_versions`: `{"llm": "stub-gpt-4o", "embed": "stub-ada-002"}`
- `anonymised`: `True`
- `role`: `"analyst_full"`
- `correlation_id`: `str(uuid.uuid4())[:8]`

**Error handling**: Include try/catch with ErrorResponse for malformed requests

### File 5: `src/api/routes/feedback.py`
**Purpose**: Stub feedback endpoint
**Function**: `async def feedback_endpoint(request: FeedbackRequest) -> Dict[str, str]`
**Return**: `{"status": "ok"}`
**Error handling**: Validate rating range, return proper HTTPException if invalid

### File 6: `src/api/routes/health.py`
**Purpose**: Health check endpoint with optional database connectivity test
**Required imports**: 
- `from fastapi import APIRouter`
- `import time`
- `from src.rag.utils.db_utils import get_db_connection` (or similar)
**Function**: `async def health_check() -> Dict[str, Any]`
**Implementation**:
```python
async def health_check():
    result = {
        "status": "ok", 
        "timestamp": time.time(),
        "db": False
    }
    try:
        # Simple connection test - don't fail if DB unavailable
        # TODO: Replace with actual db connection test
        result["db"] = True
    except Exception:
        # Log error but don't fail health check
        pass
    return result
```

---
## Frontend Implementation (React + Vite)

### File 7: `frontend/package.json`
**Purpose**: Dependencies and scripts
**Required dependencies**:
```json
{
  "name": "rag-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.15",
    "@types/react-dom": "^18.2.7",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "@vitejs/plugin-react": "^4.0.3",
    "eslint": "^8.45.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.3",
    "typescript": "^5.0.2",
    "vite": "^4.4.5"
  }
}
```

### File 8: `frontend/vite.config.ts`
**Purpose**: Vite configuration with environment-aware proxy to backend
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

### File 9: `frontend/src/types/api.ts`
**Purpose**: TypeScript interfaces matching backend schemas exactly
**Required interfaces**: Mirror the Pydantic models from `schemas/chat.py`
```typescript
export type RouteType = 'SQL' | 'VECTOR' | 'HYBRID' | 'CLARIFICATION_NEEDED';
export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';

export interface ChatRequest {
  query: string;
  session_id?: string;
}

export interface SQLEvidence {
  query: string;
  rows: any[][];
  row_count: number;
  truncated: boolean;
}

export interface VectorEvidence {
  text: string;
  score: number;
  sentiment?: string;
  agency?: string;
}

export interface Evidence {
  sql?: SQLEvidence;
  vectors: VectorEvidence[];
}

export interface ChatResponse {
  query_id: string;
  route: RouteType;
  confidence: ConfidenceLevel;
  answer: string;
  evidence: Evidence;
  timings_ms: Record<string, number>;
  model_versions: Record<string, string>;
  anonymised: boolean;
  role: string;
  correlation_id: string;
  requires_clarification: boolean;
  error?: string;
}

export interface FeedbackRequest {
  query_id: string;
  rating: number;
  comment?: string;
}

export interface ErrorResponse {
  error: string;
  error_code: string;
  correlation_id: string;
  timestamp: number;
}
```

### File 10: `frontend/src/hooks/useChat.ts`
**Purpose**: Chat state management hook
**Key state**: `messages` array with `{ id, role: 'user' | 'assistant', content, status?, route?, confidence?, evidence?, error? }`
**Key function**: `send(query: string)` that:
1. Adds user message to state
2. Adds pending assistant message
3. POSTs to `/api/chat`
4. Updates assistant message with response or error state
**Error handling**: Catch fetch errors and network issues

### File 11: `frontend/src/components/RouteBadge.tsx`
**Purpose**: Display route + confidence with color coding
**Props**: `route: RouteType, confidence: ConfidenceLevel`
**Implementation**: Simple colored badge component

### File 12: `frontend/src/components/ChatContainer.tsx`
**Purpose**: Main layout container
**Children**: MessageList + Composer

### File 13: `frontend/src/components/MessageList.tsx`
**Purpose**: Render array of messages
**Props**: `messages` from useChat hook

### File 14: `frontend/src/components/Message.tsx`
**Purpose**: Individual message component
**Props**: `message` object with role, content, route, confidence
**Conditional rendering**: Show RouteBadge only for assistant messages

### File 15: `frontend/src/components/Composer.tsx`
**Purpose**: Query input form
**Props**: `onSend: (query: string) => void`
**Implementation**: Textarea + send button, handles empty validation

### File 16: `frontend/src/App.tsx`
**Purpose**: Main app component using ChatContainer + useChat

### File 17: `frontend/src/main.tsx`
**Purpose**: React app entry point

---
## Critical Integration Notes

1. **Import Path Strategy**: API layer lives in `src/api/` to avoid conflicts with existing `src/rag/` modules. Clean separation enables:
   - P1 integration: `from src.rag.core.agent import RAGAgent` 
   - No circular import risks
   - Future extraction to separate service if needed

2. **Privacy Compliance**: Even in stub mode, ensure `anonymised: true` flag is always set. This prepares for real PII detection integration in Phase P1.

3. **Error Handling Foundation**: Stubs include proper error response schemas and HTTP status codes to establish patterns for real implementation.

4. **Environment Configuration**: CORS and API URLs use environment variables to support different deployment contexts.

5. **Database Connection Strategy**: Health check includes optional DB test but doesn't fail if database unavailable - enables testing without full infrastructure.

6. **Schema Sync Validation**: Include contract verification to prevent TypeScript/Python drift.

---
## Development Workflow

1. **Backend First**: Start FastAPI server with `uvicorn src.api.main:create_app --reload --port 8000`
2. **Frontend Second**: Start Vite dev server with `npm run dev` in frontend directory (uses environment-aware proxy)
3. **Verification**: Test full cycle by typing query and seeing stub response rendered
4. **Contract Testing**: Manually verify JSON response matches schema exactly
5. **Environment Variables**: Set `CORS_ORIGINS=http://localhost:5173` and `VITE_API_URL=http://localhost:8000` if needed

---
## Phase P0 Success Metrics

- [ ] Backend returns valid JSON for `/api/chat` POST matching ChatResponse schema exactly
- [ ] Frontend successfully renders stub answer + route badge  
- [ ] Error responses return proper ErrorResponse schema with HTTP status codes
- [ ] No CORS errors in browser console with environment-aware configuration
- [ ] TypeScript compilation succeeds with no errors
- [ ] Manual query → answer flow works end-to-end
- [ ] Health check includes database connectivity test (passes even if DB unavailable)
- [ ] Schema validation test confirms TypeScript interfaces match Pydantic models

---
## Handoff Payload for Phase P1

After completing P0, provide:
- Schema hash: MD5 of ChatResponse field names (for contract stability tracking)
- Endpoint test results: successful response examples for each route stub
- Frontend component registry: list of created components and their props
- Integration readiness: confirmation that `src/rag/core/*` modules can be imported from `src/api/` without conflicts
- Error handling patterns: documented error response schemas and HTTP status code usage
- Environment configuration: validated CORS and proxy setup for development and production contexts

---
## Questions Needing Clarification

1. **Database Connection**: ✅ Resolved - Include optional database connectivity test in health check that doesn't fail if DB unavailable, enabling whitebox testing without requiring full infrastructure.

2. **Authentication**: ✅ Resolved - Defer authentication to later phases to keep P0 focused on core contracts.

3. **Logging**: ✅ Resolved - Wait for P4 observability phase to avoid scope creep.

4. **Schema Validation**: Should P0 include automated tests that verify TypeScript interfaces exactly match Pydantic models? This would catch contract drift early.

**Recommendation**: Add schema validation test to P0 success metrics - it's low complexity but high value for preventing integration issues in P1.
