#!/usr/bin/env python3
"""
RAG Module Main Entry Point

Terminal application for the Text-to-SQL MVP.
This implements a minimal functional slice for querying attendance
and evaluation data using natural language.

Usage:
    python src/rag/runner.py
    python -m src.rag.runner
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.config.settings import get_settings, validate_configuration


def setup_logging(settings):
    """Configure logging for the RAG module."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rag.log', mode='a')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("RAG Module started")
    logger.info(f"Configuration: Model={settings.llm_model_name}, DB={settings.rag_db_host}")
    return logger


def main():
    """
    Main entry point for RAG module.
    
    Phase 1 MVP: Basic configuration validation and setup.
    Future phases will add Text-to-SQL functionality.
    """
    print("="*60)
    print("RAG Module - Text-to-SQL MVP (Phase 1)")
    print("="*60)
    
    try:
        # Load and validate configuration
        print("Loading configuration...")
        settings = get_settings()
        
        print("Validating configuration...")
        validate_configuration()
        
        # Setup logging
        logger = setup_logging(settings)
        
        print("\n" + "Configuration validated successfully!")
        print(f"   Database: {settings.rag_db_host}:{settings.rag_db_port}/{settings.rag_db_name}")
        print(f"   User: {settings.rag_db_user} (read-only)")
        print(f"   LLM Model: {settings.llm_model_name}")
        print(f"   Max Results: {settings.max_query_results}")
        print(f"   Debug Mode: {settings.debug_mode}")
        
        # Future implementation: Terminal interface
        print("\n" + "Terminal interface coming in Task 1.6")
        print("   Current status: Configuration management complete (Task 1.2)")
        print("   Next: Module structure setup (Task 1.3)")
        
        print("\n" + "="*60)
        print("RAG Module initialisation complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n RAG Module initialisation failed: {e}")
        logger.error(f"Initialisation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
