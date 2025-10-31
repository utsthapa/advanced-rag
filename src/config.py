"""Configuration and constants for the RAG system"""

import logging
import os

# AWS Configuration
AWS_PROFILE = "artisan-dev"
AWS_REGION = "us-east-1"

# RAG Configuration
RAG_MAX_DOCS = None  # Process ALL documents
RAG_TOP_K = 5
RAG_RERANK_MULTIPLIER = 8  # Multiplier for retrieval when reranking (k * multiplier)
# Increased from 4 to 8 for better reranking performance. Higher values (8-10) give rerankers
# more diverse candidates, improving NDCG. Trade-off: slightly increased latency.
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.9

# Performance Mode Configuration
# BALANCED_MODE: Balance speed and quality (~500ms)
# QUALITY_MODE: Maximum quality (~2-3s)
BYPASS_CLASSIFICATION = False  # Enable LLM-based query classification
BYPASS_REWRITING = False  # Enable query rewriting
SKIP_RERANKING = False  # Enable reranking
SKIP_COMPRESSION = False  # Enable document compression
USE_HYBRID_SEARCH = True  # Use hybrid search (vector + fulltext)
CACHE_EMBEDDINGS = True  # Cache query embeddings

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "rag_db"),
    "user": os.getenv("DB_USER", "utsavthapa"),
    "password": os.getenv("DB_PASSWORD", "admin"),
}

# Connection Pool Settings (for high-performance mode)
DB_POOL_MIN = 5
DB_POOL_MAX = 20  # Increased for benchmarks
DB_POOL_TIMEOUT = 10

# Logging Configuration
LOG_LEVEL = os.environ.get("RAG_LOG_LEVEL", "WARNING").upper()


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING))

    # Suppress noisy library logs
    logging.getLogger("langchain_aws").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
