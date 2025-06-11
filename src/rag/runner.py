#!/usr/bin/env python3
"""
RAG Module Main Entry Point

Terminal application for the Text-to-SQL MVP.
This implements a minimal functional slice for querying attendance
and evaluation data using natural language.

- Async-first design with asyncio.run() integration
- Terminal application with LangGraph SQL workflow
- Dynamic schema provision and SQL generation
- Secure read-only database access with verification

Usage:
    python src/rag/runner.py
    python -m src.rag.runner
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.config.settings import get_settings, validate_configuration
from rag.utils.logging_utils import setup_logging, get_logger
from rag.interfaces.terminal_app import run_terminal_app


def main():
    """
    Main entry point for RAG module.
    
    **Phase 1 Complete**: Runs async terminal application with
    full Text-to-SQL functionality including LangGraph integration.
    """
    print("="*80)
    print("RAG Module - Text-to-SQL MVP (Phase 1 Complete)")
    print("   Australian Public Service Learning Analytics")
    print("="*80)
    
    try:
        # Load and validate configuration
        print("Loading configuration...")
        settings = get_settings()
        
        print("🔍 Validating configuration...")
        validate_configuration()
        
        # Setup logging system
        print("Setting up logging...")
        setup_logging()
        logger = get_logger(__name__)
        
        print("\nConfiguration validated successfully!")
        print(f"    Database: {settings.rag_db_host}:{settings.rag_db_port}/{settings.rag_db_name}")
        print(f"    User: {settings.rag_db_user} (read-only)")
        print(f"    LLM Model: {settings.llm_model_name}")
        print(f"    Max Results: {settings.max_query_results}")
        print(f"    Debug Mode: {settings.debug_mode}")
        
        print("\nStarting terminal application...")
        print("   Running in async mode with full Phase 1 functionality")
        print("   Secure read-only database access")
        print("   Dynamic schema provision and SQL generation")
        print("   LangGraph workflow integration")
        
        logger.info("RAG Module Phase 1 MVP starting")
        logger.info(f"Configuration: Model={settings.llm_model_name}, DB={settings.rag_db_host}")
        
        # Run async terminal application
        asyncio.run(run_terminal_app())
        
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user")
        if 'logger' in locals():
            logger.info("Application terminated by user (Ctrl+C)")
        
    except Exception as e:
        print(f"\n❌ RAG Module initialization failed: {e}")
        if 'logger' in locals():
            logger.error(f"Initialization error: {e}")
        else:
            # Fallback logging if logger setup failed
            logging.error(f"Critical initialization error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
