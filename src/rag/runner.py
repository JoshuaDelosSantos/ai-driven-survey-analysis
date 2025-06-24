#!/usr/bin/env python3
"""
RAG Module Main Entry Point

Production-ready terminal application with modular LangGraph agent.
This implements the complete Phase 3 system with advanced query 
classification and multi-modal analysis capabilities.

Features:
- Modular query classification system with 8 specialised components
- Advanced confidence calibration and circuit breaker resilience
- Async-first design with asyncio.run() integration
- LangGraph agent with SQL, Vector, and Hybrid processing
- Dynamic schema provision and intelligent query routing
- Secure read-only database access with verification
- Australian Privacy Principles (APP) compliance

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
    
    **Phase 3 Complete + Modular Architecture**: Runs async terminal application with
    full LangGraph agent functionality including modular query classification system.
    """
    print("="*80)
    print("RAG Module - Production Ready (Phase 3 Complete + Modular Architecture)")
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
        print("   Running in async mode with full Phase 3 functionality")
        print("   Modular query classification with 8 specialised components")
        print("   Advanced confidence calibration and circuit breaker resilience")
        print("   Secure read-only database access")
        print("   Dynamic schema provision and intelligent query routing")
        print("   LangGraph agent with SQL, Vector, and Hybrid processing")
        
        logger.info("RAG Module Production Ready (Phase 3 Complete + Modular Architecture) starting")
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
