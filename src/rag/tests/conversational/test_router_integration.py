import pytest
from rag.core.conversational.router import ConversationalRouter

@ pytest.mark.asyncio
async def test_router_returns_template():
    class DummyHandler:
        def is_conversational_query(self, q): return (True, None, 1.0)
        def handle_conversational_query(self, q): return type("R", (), {"content":"ok", "enhancement_used":False})
    router = ConversationalRouter(handler=DummyHandler(), llm_enhancer=None, pattern_classifier=None)
    resp = await router.route_conversational_query("hi")
    assert resp.content == "ok"
