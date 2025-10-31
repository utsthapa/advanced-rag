#!/usr/bin/env python3
"""
Advanced RAG System - Main CLI Entry Point

This is the main entry point for the RAG system. It provides:
- Dataset loading
- Interactive chat interface
- Quick testing
- Comprehensive testing
"""
import argparse
import sys
import time

from src.config import setup_logging
from src.core import DatabaseManager
from src.pipeline import print_rag_answer, rag_pipeline
from src.retrieval import search
from src.utils import load_dataset_to_db

# Setup logging
setup_logging()


def interactive_chat():
    """Interactive chat interface for testing RAG system"""
    print("\n🤖 RAG Chat Interface")
    print("=" * 50)
    print("Commands:")
    print("  /help      - Show this help")
    print("  /stats     - Show database stats")
    print("  /quit      - Exit chat")
    print("=" * 50)

    db = DatabaseManager()
    print(f"Database stats: {db.get_stats()}")
    print("\nAsk me anything! (or type /quit to exit)")
    print("I'll retrieve relevant documents and generate answers with citations.")

    while True:
        try:
            user_input = input(f"\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command == "/quit" or command == "/q":
                    print("👋 Goodbye!")
                    break

                elif command == "/help" or command == "/h":
                    print("\nCommands:")
                    print("  /help      - Show this help")
                    print("  /stats     - Show database stats")
                    print("  /quit      - Exit chat")
                    continue

                elif command == "/stats":
                    try:
                        stats = db.get_stats()
                        print(f"\n📊 Database Stats:")
                        print(f"   Documents: {stats['documents']}")
                        print(f"   Chunks: {stats['chunks']}")
                        print(f"   Queries logged: {stats['queries']}")
                    except Exception as e:
                        print(f"❌ Error getting stats: {e}")
                    continue

                else:
                    print(f"Unknown command: {user_input}")
                    continue

            # Process the query with full RAG pipeline - all 5 steps enabled
            print(f"\n🔍 Processing query: '{user_input}'")

            result = rag_pipeline(
                query=user_input,
                k=5,
                use_rewrite=True,  # Step 2: Query Rewriting enabled
                reranker="all",  # Step 3: All three rerankers (cohere, bge, claude)
                use_compression=True,  # Step 4: Contextual Compression enabled
                verbose=True,  # Show all steps
            )

            print_rag_answer(result)

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")

    db.close()


def run_quick_test():
    """Run quick test with full RAG pipeline - all 5 steps"""
    print(f"\n{'='*80}")
    print(f"QUICK TEST - FULL RAG PIPELINE")
    print(f"{'='*80}")
    print(f"Testing with all 5 steps enabled:\n")
    print(f"  1️⃣  Classification")
    print(f"  2️⃣  Query Rewriting & Hybrid Retrieval (vector + fulltext)")
    print(f"  3️⃣  Reranking (All Three: Cohere, BGE, Claude)")
    print(f"  4️⃣  Contextual Compression")
    print(f"  5️⃣  Answer Generation")
    print(f"\n{'='*80}\n")

    test_queries = [
        "What is 2+2?",  # Should skip retrieval (non-scientific)
        "What causes COVID-19?",  # Should go through all steps
        "How does CRISPR gene editing work?",  # Should go through all steps
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/3: '{query}'")
        print(f"{'='*80}\n")

        # Use full pipeline with all features enabled
        result = rag_pipeline(
            query=query,
            k=5,
            use_rewrite=True,  # Step 2: Query Rewriting
            reranker="all",  # Step 3: All three rerankers (cohere, bge, claude)
            use_compression=True,  # Step 4: Compression
            verbose=True,  # Show all steps
        )

        print(f"\n{'─'*80}")
        print(f"✅ Test {i} Complete")
        print(f"{'─'*80}")

    print(f"\n{'='*80}")
    print(f"✅ BASIC TESTS COMPLETE")
    print(f"{'='*80}")
    print("\n💡 Results:")
    print("   • Math query: Should skip retrieval")
    print("   • Scientific queries: Should retrieve documents")
    print("\n📖 For more testing options:")
    print("   • python rag.py --chat     - Interactive chat mode")
    print("   • python rag.py --benchmark - Run performance benchmarks")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced RAG System")
    parser.add_argument(
        "--chat", action="store_true", help="Start interactive chat mode"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run comprehensive RAG analysis (answers 6 key questions)",
    )
    parser.add_argument(
        "--load-data", action="store_true", help="Load dataset into database"
    )

    args = parser.parse_args()

    # Check database
    print("Checking database...")
    db = DatabaseManager()
    stats = db.get_stats()

    # Load data if requested or if database is empty
    if args.load_data or stats["documents"] == 0:
        if stats["documents"] == 0:
            print("No data found. Loading dataset into database...")
        else:
            print("Reloading dataset...")

        try:
            result = load_dataset_to_db()
            print(f"Load result: {result}")
            # Refresh stats
            stats = db.get_stats()
        except Exception as e:
            print(f"\n❌ Error loading dataset: {e}")
            print("\n💡 If HuggingFace is timing out, you can:")
            print("   1. Try again later")
            print("   2. Run in chat mode with existing data: python main.py --chat")
            db.close()
            sys.exit(1)
    else:
        print(
            f"✅ Database already contains {stats['documents']} documents and {stats['chunks']} chunks"
        )

    # Chat mode
    if args.chat:
        if stats["documents"] == 0:
            print("❌ No documents found in database!")
            print("   Please run: python rag.py --load-data")
            db.close()
            sys.exit(1)

        print(f"✅ Found {stats['documents']} documents and {stats['chunks']} chunks")
        db.close()
        interactive_chat()
        sys.exit(0)

    # Benchmark mode
    if args.benchmark:
        if stats["documents"] == 0:
            print("❌ No documents found in database!")
            print("   Please run: python rag.py --load-data")
            db.close()
            sys.exit(1)

        print(f"✅ Found {stats['documents']} documents and {stats['chunks']} chunks\n")
        db.close()

        # Ask which benchmark to run
        print("\nSelect benchmark type:")
        print("  1. Performance benchmarks (latency, quality, compression, cost)")
        print("  2. Load test (50+ QPS)")
        print("  3. System requirements (comprehensive)")

        try:
            choice = input("\nEnter choice (1-3, default=3): ").strip() or "3"
        except (EOFError, KeyboardInterrupt):
            # Non-interactive mode, use default
            choice = "3"
            print("   (Non-interactive mode, using default: System requirements)")

        if choice == "1":
            from src.benchmarks import PerformanceBenchmarkSuite

            suite = PerformanceBenchmarkSuite()
            success = suite.run_all_benchmarks()
        elif choice == "2":
            from src.benchmarks import run_load_test_suite

            results = run_load_test_suite(target_qps=50, duration_seconds=60)
            success = results.get("passed", False)
        else:
            from src.benchmarks import SystemRequirementsTest

            test_suite = SystemRequirementsTest()
            success = test_suite.run_all_tests()

        sys.exit(0 if success else 1)

    # Analysis mode
    if args.analyze:
        if stats["documents"] == 0:
            print("❌ No documents found in database!")
            print("   Please run: python rag.py --load-data")
            db.close()
            sys.exit(1)

        print(f"✅ Found {stats['documents']} documents and {stats['chunks']} chunks\n")
        db.close()

        from src.benchmarks.rag_analysis import RAGAnalysis

        analyzer = RAGAnalysis()
        results = analyzer.run_all_analyses()
        sys.exit(0 if results else 1)

    # Default: run quick test
    if stats["documents"] == 0:
        print("❌ No documents found in database!")
        print("   Please run: python main.py --load-data")
        db.close()
        sys.exit(1)

    db.close()
    run_quick_test()

    print(f"\n🎉 Quick test complete!")


if __name__ == "__main__":
    main()
