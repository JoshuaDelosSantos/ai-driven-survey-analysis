import pytest
import asyncio
from rag.core.conversational.learning_integrator import LearningIntegrator
from rag.core.conversational.handler import ConversationalHandler, ConversationalPattern, PatternLearningData

class DummyHandler:
    def __init__(self):
        # simulate existing pattern learning
        self.pattern_learning = {"PATTERN_2": DummyData()}

class DummyData:
    def __init__(self):
        self.llm_usage_count = 1
        self.llm_effectiveness = 0.5
        self.success_rate = 0.6
        self.template_vs_llm_preference = "template"
        self.updated = False
    def update_success_rate(self, was_helpful, source):
        self.updated = True

def test_learning_integrator_updates_llm_stats():
    # setup handler with dummy pattern_learning
    handler = ConversationalHandler()
    handler.pattern_learning = {"test_2": PatternLearningData(success_rate=0.5)}
    integrator = LearningIntegrator(conversational_handler=handler)
    # simulate feedback
    integrator.update_learning_with_llm_feedback("q1 q2", "test", True, True)
    data = handler.pattern_learning["test_2"]
    assert data.llm_usage_count == 1

@pytest.mark.asyncio
async def test_update_learning_with_llm_feedback_increments_and_updates():
    handler = DummyHandler()
    integrator = LearningIntegrator(handler)
    await integrator.update_learning_with_llm_feedback("q", ConversationalPattern.PATTERN, True, True)
    data = handler.pattern_learning["PATTERN_2"]
    assert data.llm_usage_count == 2
    assert data.llm_effectiveness > 0.5
    # preference should switch to llm when llm_effectiveness > success_rate
    assert data.template_vs_llm_preference in ["llm", "hybrid"]
    assert data.updated
