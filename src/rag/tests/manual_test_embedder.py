#!/usr/bin/env python3
"""
Manual test script for Embedder functionality.

This script provides a manual testing interface for the Embedder class,
allowing you to test different configurations and see real-time results.

Usage:
    python manual_test_embedder.py

Features:
- Test both OpenAI and Sentence Transformer models
- Interactive text input
- Batch processing demonstration
- Performance metrics display
- Configuration testing
"""

import asyncio
import logging
import time
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.core.vector_search.embedder import Embedder
from src.rag.config.settings import RAGSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_sentence_transformer_model():
    """Test the local Sentence Transformer model."""
    print("\n" + "="*60)
    print("TESTING SENTENCE TRANSFORMER MODEL")
    print("="*60)
    
    embedder = Embedder(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        batch_size=10
    )
    
    print("Initializing embedder...")
    start_time = time.time()
    await embedder.initialize()
    init_time = time.time() - start_time
    
    print(f"✅ Initialized in {init_time:.2f}s")
    
    # Display model info
    model_info = embedder.get_model_info()
    print(f"📊 Model Info:")
    print(f"   Provider: {model_info['provider']}")
    print(f"   Model: {model_info['model']}")
    print(f"   Dimension: {model_info['dimension']}")
    print(f"   Version: {model_info['model_version']}")
    
    # Test single text embedding
    print("\n📝 Testing single text embedding...")
    test_text = "This is a sample course evaluation feedback for testing."
    
    result = await embedder.embed_text(test_text)
    print(f"✅ Generated embedding for text: '{test_text[:50]}...'")
    print(f"   Embedding dimension: {len(result.embedding)}")
    print(f"   Processing time: {result.processing_time:.3f}s")
    print(f"   First 5 values: {result.embedding[:5]}")
    
    # Test batch embedding
    print("\n📦 Testing batch embedding...")
    sample_texts = [
        "The course was excellent and very informative.",
        "I experienced technical issues during the session.",
        "The facilitator was knowledgeable and engaging.",
        "The content was relevant to my work.",
        "I would recommend this course to others.",
        "The online platform was easy to use.",
        "More practical examples would be helpful.",
        "The course duration was appropriate."
    ]
    
    batch_result = await embedder.embed_batch(sample_texts)
    print(f"✅ Processed batch of {len(sample_texts)} texts")
    print(f"   Success count: {batch_result.success_count}")
    print(f"   Error count: {batch_result.error_count}")
    print(f"   Total time: {batch_result.total_processing_time:.2f}s")
    print(f"   Average per text: {batch_result.total_processing_time/len(sample_texts):.3f}s")
    
    # Display performance metrics
    metrics = embedder.get_metrics()
    print(f"\n📈 Performance Metrics:")
    print(f"   Total embeddings: {metrics['total_embeddings_generated']}")
    print(f"   Total time: {metrics['total_processing_time']:.2f}s")
    print(f"   Average per embedding: {metrics['average_time_per_embedding']:.3f}s")
    
    return embedder


async def test_openai_model():
    """Test OpenAI model (requires API key)."""
    print("\n" + "="*60)
    print("TESTING OPENAI MODEL")
    print("="*60)
    
    try:
        # Check if we have an API key
        settings = RAGSettings()
        if not settings.llm_api_key:
            print("⚠️  No OpenAI API key found. Skipping OpenAI test.")
            print("   Set LLM_API_KEY environment variable to test OpenAI embedding.")
            return None
        
        embedder = Embedder(
            provider="openai",
            model_name="text-embedding-ada-002",
            batch_size=5
        )
        
        print("Initializing OpenAI embedder...")
        start_time = time.time()
        await embedder.initialize()
        init_time = time.time() - start_time
        
        print(f"✅ Initialized in {init_time:.2f}s")
        
        # Display model info
        model_info = embedder.get_model_info()
        print(f"📊 Model Info:")
        print(f"   Provider: {model_info['provider']}")
        print(f"   Model: {model_info['model']}")
        print(f"   Dimension: {model_info['dimension']}")
        print(f"   Version: {model_info['model_version']}")
        
        # Test single text
        test_text = "Course evaluation feedback for OpenAI embedding test."
        result = await embedder.embed_text(test_text)
        
        print(f"\n✅ Generated OpenAI embedding")
        print(f"   Dimension: {len(result.embedding)}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        
        return embedder
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return None


async def test_batch_processing_performance():
    """Test batch processing performance with different batch sizes."""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING PERFORMANCE")
    print("="*60)
    
    embedder = Embedder(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2"
    )
    await embedder.initialize()
    
    # Create test data
    test_texts = [f"Evaluation feedback text number {i} for performance testing." for i in range(50)]
    
    batch_sizes = [5, 10, 25, 50]
    
    print(f"Testing with {len(test_texts)} texts using different batch sizes:")
    
    for batch_size in batch_sizes:
        # Reset metrics
        embedder._total_embeddings_generated = 0
        embedder._total_processing_time = 0
        
        start_time = time.time()
        batch_result = await embedder.embed_batch(test_texts, custom_batch_size=batch_size)
        total_time = time.time() - start_time
        
        print(f"   Batch size {batch_size:2d}: {total_time:.2f}s ({total_time/len(test_texts):.3f}s per text)")


async def interactive_test():
    """Interactive test allowing user input."""
    print("\n" + "="*60)
    print("INTERACTIVE EMBEDDING TEST")
    print("="*60)
    
    embedder = Embedder(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2"
    )
    await embedder.initialize()
    
    print("Enter texts to embed (empty line to finish):")
    texts = []
    
    while True:
        try:
            text = input("Text: ").strip()
            if not text:
                break
            texts.append(text)
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive test.")
            return
    
    if texts:
        print(f"\nProcessing {len(texts)} texts...")
        batch_result = await embedder.embed_batch(texts)
        
        print(f"✅ Results:")
        for i, result in enumerate(batch_result.results):
            print(f"   Text {i+1}: {len(result.embedding)} dimensions, {result.processing_time:.3f}s")
        
        print(f"\n📊 Batch Summary:")
        print(f"   Total time: {batch_result.total_processing_time:.2f}s")
        print(f"   Success rate: {batch_result.success_count}/{len(texts)}")


async def test_error_handling():
    """Test error handling scenarios."""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    embedder = Embedder(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2"
    )
    await embedder.initialize()
    
    # Test empty text
    print("Testing empty text handling...")
    try:
        await embedder.embed_text("")
        print("❌ Should have failed with empty text")
    except ValueError as e:
        print(f"✅ Correctly handled empty text: {e}")
    
    # Test empty batch
    print("\nTesting empty batch handling...")
    try:
        await embedder.embed_batch([])
        print("❌ Should have failed with empty batch")
    except ValueError as e:
        print(f"✅ Correctly handled empty batch: {e}")
    
    # Test invalid metadata
    print("\nTesting metadata length mismatch...")
    try:
        await embedder.embed_batch(["text1", "text2"], metadata_list=[{"meta": "data"}])
        print("❌ Should have failed with metadata mismatch")
    except ValueError as e:
        print(f"✅ Correctly handled metadata mismatch: {e}")


async def test_context_manager():
    """Test embedder as context manager."""
    print("\n" + "="*60)
    print("TESTING CONTEXT MANAGER")
    print("="*60)
    
    print("Testing embedder as async context manager...")
    
    async with Embedder(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2"
    ) as embedder:
        print("✅ Context manager initialized embedder")
        
        result = await embedder.embed_text("Context manager test text")
        print(f"✅ Generated embedding: {len(result.embedding)} dimensions")
        
        model_info = embedder.get_model_info()
        print(f"✅ Model info: {model_info['model']} ({model_info['dimension']} dims)")
    
    print("✅ Context manager completed successfully")


async def main():
    """Main test runner."""
    print("🚀 EMBEDDER MANUAL TEST SUITE")
    print("="*60)
    
    try:
        # Test sentence transformer model
        await test_sentence_transformer_model()
        
        # Test OpenAI model (if API key available)
        await test_openai_model()
        
        # Test batch processing performance
        await test_batch_processing_performance()
        
        # Test error handling
        await test_error_handling()
        
        # Test context manager
        await test_context_manager()
        
        # Interactive test (optional)
        print("\n" + "="*60)
        response = input("Run interactive test? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            await interactive_test()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
