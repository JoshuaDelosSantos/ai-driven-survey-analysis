import pytest
from rag.core.conversational.llm_enhancer import ConversationalLLMEnhancer, ConversationalResponse
from rag.core.privacy.pii_detector import PIIDetector
from rag.utils.llm_utils import LLMManager

class DummyLLM:
    async def generate(self, prompt, schema_only):
        return "enhanced response"

class DummyPIIDetector:
    async def detect_and_anonymise(self, text):
        return {"pii_detected": False}

@pytest.mark.asyncio
async def test_enhancer_returns_template_when_confidence_high():
    enhancer = ConversationalLLMEnhancer(llm_manager=LLMManager(), pii_detector=PIIDetector())
    # confidence above threshold should return template without enhancement
    resp = await enhancer.enhance_response_if_needed("test query", "template_response", 0.8)
    assert resp.content == "template_response"
    assert resp.enhancement_used is False

@pytest.mark.asyncio
async def test_bypass_llm_when_confident():
    enhancer = ConversationalLLMEnhancer(DummyLLM(), DummyPIIDetector())
    resp = await enhancer.enhance_response_if_needed("q", "template", 0.8)
    assert isinstance(resp, ConversationalResponse)
    assert resp.content == "template"
    assert not resp.enhancement_used

@pytest.mark.asyncio
async def test_trigger_llm_when_unconfident():
    enhancer = ConversationalLLMEnhancer(DummyLLM(), DummyPIIDetector())
    resp = await enhancer.enhance_response_if_needed("q", "template", 0.5)
    assert resp.content == "enhanced response"
    assert resp.enhancement_used

@pytest.mark.asyncio
async def test_pii_detection_fallback():
    class PIIDetectTrue(DummyPIIDetector):
        async def detect_and_anonymise(self, text):
            return {"pii_detected": True}
    enhancer = ConversationalLLMEnhancer(DummyLLM(), PIIDetectTrue())
    resp = await enhancer.enhance_response_if_needed("private data", "template", 0.5)
    assert resp.content == "template"
    assert not resp.enhancement_used
