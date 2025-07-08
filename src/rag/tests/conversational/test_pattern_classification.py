import asyncio
import pytest
from rag.core.conversational.pattern_classifier import ConversationalPatternClassifier


class DummyEmbedder:
    async def embed_text(self, text):
        return [0.1, 0.2, 0.3]

class DummySim:
    def __init__(self, score):
        self.score = score

class DummyVectorStore:
    def __init__(self, sims):
        self._sims = sims
    async def similarity_search(self, embedding, filter_metadata, limit):
        return self._sims

@pytest.mark.asyncio
async def test_enhance_pattern_confidence_no_similarities():
    embedder = DummyEmbedder()
    store = DummyVectorStore([])
    classifier = ConversationalPatternClassifier(embedder, store)
    result = await classifier.enhance_pattern_confidence("query", 0.5)
    assert result == pytest.approx(0.5)

@pytest.mark.asyncio
async def test_enhance_pattern_confidence_with_similarities():
    embedder = DummyEmbedder()
    store = DummyVectorStore([DummySim(0.6), DummySim(0.4)])
    classifier = ConversationalPatternClassifier(embedder, store)
    # vector_confidence*0.8 = 0.6*0.8 = 0.48 < base, so base remains
    assert await classifier.enhance_pattern_confidence("q", 0.7) == pytest.approx(0.7)
    # base lower than weighted vector
    result2 = await classifier.enhance_pattern_confidence("q", 0.3)
    assert result2 == pytest.approx(0.6*0.8)

@ pytest.mark.asyncio
async def test_pattern_classifier_returns_base_confidence():
    # stub: instantiate with dummy embedder and vector_store
    class DummyEmbedder:
        async def embed_text(self, text): return [0.0]
    class DummyStore:
        async def similarity_search(self, embedding, filter_metadata, limit): return []
    classifier = ConversationalPatternClassifier(embedder=DummyEmbedder(), vector_store=DummyStore())
    conf = await classifier.enhance_pattern_confidence("any query", 0.5)
    assert conf == 0.5
