#!/usr/bin/env python3
"""
Quick test to verify the real SentimentAnalyser is working with actual sentiment analysis
"""

import asyncio
import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.data.content_processor import SentimentAnalyser

@pytest.mark.asyncio
async def test_real_sentiment():
    """Test real sentiment analysis with sample texts."""
    
    analyser = SentimentAnalyser()
    
    test_texts = [
        "I loved this course! It was amazing and very helpful.",
        "The course was terrible and a complete waste of time.",
        "The course was okay, nothing special but not bad either.",
        "I experienced some technical issues during the session, but the content was good overall."
    ]
    
    print("Testing Real Sentiment Analysis:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        scores = analyser.analyse(text)
        print(f"\nTest {i}: {text}")
        print(f"Sentiment scores: {scores}")
        
        # Determine dominant sentiment
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant]
        print(f"Dominant sentiment: {dominant} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    asyncio.run(test_real_sentiment())
