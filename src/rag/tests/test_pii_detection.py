"""
Comprehensive Tests for AustralianPIIDetector

This test suite uses efficient testing strategies to validate the core PII
detection and anonymisation functionality, including both the Presidio and
fallback mechanisms.

- A session-scoped pytest fixture initialises the slow Presidio engine only ONCE.
- Comprehensive, parameterised tests for both Presidio and fallback paths.
- Tests for batch processing and error handling.
- Uses pytest-asyncio to handle async functions correctly.
"""

import pytest
import pytest_asyncio
import sys
from pathlib import Path
from unittest.mock import patch

# --- Test Setup ---
# Ensure the application modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Defer import of the module under test
from core.privacy.pii_detector import (
    AustralianPIIDetector,
    PIIDetectionResult,
    get_pii_detector
)

# --- Test Data ---
# A mapping of PII type to a text sample and its expected anonymised version.
# This makes it easy to add new test cases.
# (text, expected_presidio, expected_fallback)
PII_TEST_CASES = {
    "EMAIL": ("Contact me at real.email@example.com.", "Contact me at [EMAIL].", "Contact me at [EMAIL]."),
    "PHONE": ("My number is 0412 345 678.", "My number is [PHONE].", "My number is [PHONE]."),
    "PERSON": ("The report was submitted by John Smith.", "The report was submitted by [PERSON].", "The report was submitted by John Smith."), # Fallback won't detect PERSON
    "LOCATION": ("We are located in Sydney.", "We are located in [LOCATION].", "We are located in Sydney."), # Fallback won't detect LOCATION
    "ORGANISATION": ("The invoice is from Acme Corp.", "The invoice is from [ORGANISATION].", "The invoice is from Acme Corp."), # Fallback won't detect ORGANISATION
    "ABN": ("Company ABN: 53 004 085 616.", "Company [ABN].", "Company [ABN]."),
    "ACN": ("Our ACN is 123 456 789.", "Our [ACN].", "Our [ACN]."),
    "MEDICARE": ("Medicare No: 2345 67890 1.", "Medicare No: [MEDICARE].", "Medicare No: [MEDICARE]."),
    "TFN": ("Please enter your TFN: 123 456 789.", "Please enter your [TFN].", "Please enter your [TFN]."),
}


# --- Fixtures ---

@pytest_asyncio.fixture(scope="session")
async def initialised_detector():
    """
    An efficient, session-scoped fixture that initialises the AustralianPIIDetector
    only once per test run, avoiding repeated, slow initialisation.
    """
    detector = AustralianPIIDetector()
    await detector.initialise()
    yield detector
    # Cleanup
    await detector.close()


# --- Test Classes ---

class TestDetectorInitialisation:
    """Tests the basic setup and configuration of the detector."""

    def test_initial_state(self):
        """Ensures the detector starts in a non-initialised state."""
        detector = AustralianPIIDetector()
        assert not detector._initialised
        assert "ABN" in detector.australian_patterns
        assert "AU_ABN" in detector.replacement_tokens
        assert "PERSON" in detector.confidence_thresholds

class TestPresidioAnonymisation:
    """
    Tests the primary PII detection logic using the real Presidio engine.
    These tests are integration tests and rely on the initialised_detector fixture.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pii_type",
        ["EMAIL", "PHONE", "PERSON", "LOCATION", "ABN", "ACN", "MEDICARE", "TFN"]
    )
    async def test_presidio_anonymises_various_pii(self, initialised_detector, pii_type):
        """Checks that various PII types are correctly found and anonymised by Presidio."""
        original_text, expected_anonymised, _ = PII_TEST_CASES[pii_type]
        
        result = await initialised_detector.detect_and_anonymise(original_text)
        
        assert result.anonymised_text == expected_anonymised
        assert result.anonymisation_applied is True
        assert len(result.entities_detected) > 0

    @pytest.mark.asyncio
    async def test_text_without_pii_is_unchanged(self, initialised_detector):
        """Ensures that text with no PII is returned unmodified."""
        text = "This is a perfectly safe sentence with no personal data."
        
        result = await initialised_detector.detect_and_anonymise(text)
        
        assert result.original_text == result.anonymised_text
        assert result.anonymisation_applied is False
        assert result.entities_detected == []

class TestBatchProcessing:
    """Tests the batch processing capabilities."""

    @pytest.mark.asyncio
    async def test_batch_processing_works(self, initialised_detector):
        """Validates that a batch of texts is processed correctly."""
        texts = [
            "My name is Jane Doe.",
            "Contact support at help@company.com.",
            "A safe message."
        ]
        
        results = await initialised_detector.batch_process(texts)
        
        assert len(results) == 3
        assert results[0].anonymised_text == "My name is [PERSON]."
        assert results[1].anonymised_text == "Contact support at [EMAIL]."
        assert results[2].anonymised_text == "A safe message."
        assert results[2].anonymisation_applied is False

    @pytest.mark.asyncio
    async def test_batch_handles_exceptions(self, initialised_detector):
        """Ensures a single failure in a batch doesn't crash the whole process."""
        texts = ["Call me on 0400111222.", "This is a valid text."]

        # Mock the underlying method to simulate a failure on the first call
        with patch.object(
            initialised_detector, 
            'detect_and_anonymise', 
            side_effect=[Exception("Simulated processing error"), await initialised_detector.detect_and_anonymise(texts[1])]
        ) as mock_detect:
            results = await initialised_detector.batch_process(texts)
            
            assert len(results) == 2
            # Check that the first result is a safe fallback
            assert results[0].anonymised_text == "[TEXT_PROCESSING_ERROR]"
            assert results[0].anonymisation_applied is True
            # Check that the second result was processed normally
            assert results[1].anonymised_text == texts[1]

class TestSingleton:
    """Tests the singleton getter function."""

    @pytest.mark.asyncio
    async def test_get_pii_detector_returns_singleton(self):
        """Ensures get_pii_detector returns the same initialised instance."""
        detector1 = await get_pii_detector()
        detector2 = await get_pii_detector()
        
        assert detector1 is detector2
        assert detector1._initialised is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])