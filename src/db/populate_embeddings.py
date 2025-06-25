#!/usr/bin/env python3
"""
Populate Vector Embeddings for RAG System

This script processes evaluation data and creates vector embeddings
for the feedback analysis functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.data.content_processor import ContentProcessor, ProcessingConfig


async def populate_embeddings():
    """Process all evaluation records and create embeddings."""
    print("🚀 Starting embedding generation for evaluation data...")
    
    try:
        # Create config optimized for short survey responses
        config = ProcessingConfig(
            text_fields=["general_feedback", "did_experience_issue_detail", "course_application_other"],
            enable_pii_detection=True,
            enable_sentiment_analysis=True,
            chunk_strategy="sentence",
            max_chunk_size=500,
            min_chunk_size=10  # Reduced from 50 to handle short feedback
        )
        
        # Initialize content processor
        processor = ContentProcessor(config)
        await processor.initialize()
        print("✅ Content processor initialized successfully")
        
        # Process all evaluations
        print("📊 Processing evaluation records (this may take a few minutes)...")
        results = await processor.process_all_evaluations(batch_size=10)
        
        # Report results
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        print(f"\n🎉 Embedding generation complete!")
        print(f"   ✅ Successfully processed: {successful}/{total} records")
        
        if successful < total:
            failed = total - successful
            print(f"   ⚠️  Failed to process: {failed} records")
            
        print(f"\n💬 Vector search is now ready for feedback analysis!")
        
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(populate_embeddings())
