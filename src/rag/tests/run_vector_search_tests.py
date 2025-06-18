"""
Test runner for Vector Search functionality (Phase 2 Task 2.5).

This script provides a convenient way to run all vector search related tests
with proper environment setup and clear output formatting.

Usage:
    python run_vector_search_tests.py
    python run_vector_search_tests.py --integration  # Include integration tests
    python run_vector_search_tests.py --unit-only    # Only unit tests
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_vector_search_tests(include_integration=False, unit_only=False):
    """Run the vector search tests with pytest."""
    print("🔍 Running Vector Search Tests (Phase 2 Task 2.5)")
    print("=" * 60)
    
    # Get test files
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_search_result.py",
        test_dir / "test_vector_search_tool.py",
        test_dir / "test_embeddings_manager.py"  # Updated with new methods
    ]
    
    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    # Add test files
    cmd.extend([str(f) for f in test_files])
    
    # Configure test selection
    if unit_only:
        cmd.extend(["-m", "not integration"])
        print("🎯 Running unit tests only")
    elif include_integration:
        print("🔗 Running all tests including integration tests")
    else:
        cmd.extend(["-m", "not integration"])
        print("🧪 Running unit tests (use --integration for integration tests)")
    
    print(f"📁 Test files: {[f.name for f in test_files]}")
    print("⚡ Starting test execution...")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("✅ All vector search tests passed!")
            print("🎉 Vector Search Tool (Phase 2 Task 2.5) implementation verified!")
            print("\n📊 Test Summary:")
            print("   • Search Result Structures: ✅")
            print("   • Vector Search Tool: ✅")
            print("   • Updated Embeddings Manager: ✅")
            print("   • Privacy Compliance: ✅")
            print("   • LangChain Integration: ✅")
        else:
            print("❌ Some vector search tests failed.")
            print("📝 Check the output above for details.")
            print("\n🔧 Troubleshooting:")
            print("   • Ensure database is running and accessible")
            print("   • Check that embedding models can be downloaded")
            print("   • Verify all dependencies are installed")
            print("   • Run individual test files for detailed debugging")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 Failed to run tests: {e}")
        return False

def run_specific_test_category(category):
    """Run tests for a specific category."""
    test_dir = Path(__file__).parent
    
    category_files = {
        "structures": ["test_search_result.py"],
        "tool": ["test_vector_search_tool.py"],
        "manager": ["test_embeddings_manager.py"],
        "privacy": ["test_vector_search_tool.py::TestPrivacyCompliance"],
        "integration": ["test_vector_search_tool.py::TestIntegrationWithDatabase"]
    }
    
    if category not in category_files:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {list(category_files.keys())}")
        return False
    
    print(f"🎯 Running {category} tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    for test_spec in category_files[category]:
        cmd.append(str(test_dir / test_spec))
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"💥 Failed to run {category} tests: {e}")
        return False

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Vector Search Tests")
    parser.add_argument("--integration", action="store_true", 
                       help="Include integration tests (requires database)")
    parser.add_argument("--unit-only", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--category", choices=["structures", "tool", "manager", "privacy", "integration"],
                       help="Run tests for specific category only")
    parser.add_argument("--quick", action="store_true",
                       help="Run a quick subset of tests for development")
    
    args = parser.parse_args()
    
    if args.category:
        success = run_specific_test_category(args.category)
    elif args.quick:
        print("⚡ Running quick development tests...")
        success = run_specific_test_category("structures")
        if success:
            print("🔧 Quick tests passed! Run full test suite before committing.")
    else:
        success = run_vector_search_tests(
            include_integration=args.integration,
            unit_only=args.unit_only
        )
    
    if success:
        print("\n🚀 All selected tests completed successfully!")
        if not args.integration and not args.unit_only and not args.category:
            print("💡 Tip: Run with --integration to test against real database")
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
