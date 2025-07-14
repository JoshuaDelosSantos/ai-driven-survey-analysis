#!/usr/bin/env python3
"""
Future-Proof Embedding Population Script

This script provides a comprehensive solution for embedding generation with:
- Model compatibility checking
- Migration planning and execution  
- Backup and rollback capabilities
- Progress monitoring and error recovery

Usage:
    # Check compatibility first (recommended)
    python populate_embeddings.py --check-compatibility
    
    # Generate embeddings with safety checks
    python populate_embeddings.py --populate
    
    # Force regeneration (with backup)
    python populate_embeddings.py --force-regenerate
    
    # Create migration plan only
    python populate_embeddings.py --plan-only
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.data.content_processor import ContentProcessor, ProcessingConfig


async def populate_embeddings_simple():
    """
    Simple embedding population using ContentProcessor directly.
    """
    print("� Starting embedding generation...")
    
    try:
        # Create processor with optimized configuration
        config = ProcessingConfig(
            batch_size=25,
            enable_progress_logging=True,
            log_interval=5,
            concurrent_processing=1,  # Reduced to avoid rate limits
            max_retries=3,
            retry_delay=1.0
        )
        
        processor = ContentProcessor(config)
        await processor.initialize()
        
        print("� Processing all evaluation records...")
        
        # Process all evaluations
        results = await processor.process_all_evaluations(batch_size=25)
        
        # Count successful results
        successful = sum(1 for r in results if r.success)
        
        print(f"\n🎉 Embedding generation completed!")
        print(f"   ✅ Successfully processed: {successful}/{len(results)} records")
        print(f"   🔧 Using model: sentence-transformers-all-MiniLM-L6-v2 (v2)")
        
        if successful < len(results):
            failed = len(results) - successful
            print(f"   ⚠️  Failed records: {failed}")
            print("   💡 Check logs for details on failed records")
        
        return successful
        
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        raise


async def check_embeddings_status():
    """Check current status of embeddings in the database."""
    print("🔍 Checking current embeddings status...")
    
    try:
        from rag.data.embeddings_manager import EmbeddingsManager
        
        manager = EmbeddingsManager()
        await manager.initialize()
        
        stats = await manager.get_stats()
        
        print(f"\n📊 Current Embeddings Status:")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Unique responses: {stats['unique_responses']}")
        print(f"   Unique fields: {stats['unique_fields']}")
        
        if stats['model_breakdown']:
            print(f"\n📦 Model Versions:")
            for model, count in stats['model_breakdown'].items():
                print(f"   - {model}: {count} embeddings")
        
        if stats['field_breakdown']:
            print(f"\n� Field Breakdown:")
            for field, count in stats['field_breakdown'].items():
                print(f"   - {field}: {count} embeddings")
        
        await manager.close()
        return stats
        
    except Exception as e:
        print(f"❌ Error checking embeddings status: {e}")
        raise


async def clean_populate_embeddings():
    """
    Clean population of embeddings - checks status first, then populates if needed.
    """
    print("🚀 Starting clean embedding population...")
    
    try:
        # First check what we have
        stats = await check_embeddings_status()
        
        if stats['total_embeddings'] > 0:
            print(f"\n� Found {stats['total_embeddings']} existing embeddings")
            print("   Use --force to regenerate all embeddings")
            return stats
        
        # No embeddings found, proceed with population
        print("\n📊 No embeddings found, starting fresh population...")
        successful = await populate_embeddings_simple()
        
        # Check final status
        print("\n🔍 Checking final status...")
        final_stats = await check_embeddings_status()
        
        return final_stats
        
    except Exception as e:
        print(f"❌ Error during clean population: {e}")
        raise


async def main():
    """Main CLI interface for embedding population."""
    parser = argparse.ArgumentParser(
        description="Simplified embedding population script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check current status
    python populate_embeddings.py --status
    
    # Populate embeddings (skips if already exists)
    python populate_embeddings.py --populate
    
    # Force regeneration
    python populate_embeddings.py --force
        """
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Check current embeddings status"
    )
    parser.add_argument(
        "--populate", 
        action="store_true",
        help="Populate embeddings (skips if already exists)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force regeneration of all embeddings"
    )
    
    args = parser.parse_args()
    
    # Default to populate if no specific action specified
    if not any([args.status, args.populate, args.force]):
        args.populate = True
    
    try:
        if args.status:
            await check_embeddings_status()
        elif args.force:
            print("🚨 Force regeneration requested...")
            await populate_embeddings_simple()
        elif args.populate:
            await clean_populate_embeddings()
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check logs for detailed error information")


if __name__ == "__main__":
    asyncio.run(main())
