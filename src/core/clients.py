"""AWS Bedrock client management"""

import hashlib
import json
import logging
from functools import lru_cache

import boto3
from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.messages import HumanMessage

from src.config import AWS_PROFILE, AWS_REGION, CACHE_EMBEDDINGS
from src.models import QueryClassification, QueryRewrite

logger = logging.getLogger(__name__)

# Global client instances (lazy loading)
_bedrock_runtime = None
_embeddings = None
_classifier = None
_rewriter = None

# Embedding cache
_embedding_cache = {}

# Bedrock client configuration - optimized for low latency
BEDROCK_CLIENT_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
    connect_timeout=5,  # Reduced for faster failure
    read_timeout=30,
    max_pool_connections=50,  # Pool connections for reuse
)


def get_bedrock_runtime():
    """Get or create AWS Bedrock runtime client"""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        logger.info("Creating Bedrock client using profile='%s'", AWS_PROFILE)
        session = boto3.Session(profile_name=AWS_PROFILE)
        _bedrock_runtime = session.client(
            "bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CLIENT_CONFIG
        )
    return _bedrock_runtime


def get_embeddings():
    """Get or create embeddings model with optimized settings"""
    global _embeddings
    if _embeddings is None:
        _embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=get_bedrock_runtime(),
            # Optimize batch size for better throughput
            model_kwargs={"dimensions": 1024},
        )
    return _embeddings


def embed_query_cached(query: str) -> list:
    """Embed query with caching for performance"""
    if CACHE_EMBEDDINGS and query in _embedding_cache:
        return _embedding_cache[query]

    embeddings = get_embeddings()
    embedding = embeddings.embed_query(query)

    if CACHE_EMBEDDINGS:
        _embedding_cache[query] = embedding

    return embedding


def embed_queries_batch(queries: list) -> list:
    """Embed multiple queries in a single batch for better throughput"""
    # Check cache and collect uncached queries
    results = []
    uncached_queries = []
    uncached_positions = []

    for i, query in enumerate(queries):
        if CACHE_EMBEDDINGS and query in _embedding_cache:
            results.append(_embedding_cache[query])
        else:
            results.append(None)  # Placeholder
            uncached_queries.append(query)
            uncached_positions.append(i)

    # Batch embed uncached queries
    if uncached_queries:
        embeddings = get_embeddings()
        batch_embeddings = embeddings.embed_documents(uncached_queries)

        # Cache the results and fill in placeholders
        for query, embedding, pos in zip(
            uncached_queries, batch_embeddings, uncached_positions
        ):
            if CACHE_EMBEDDINGS:
                _embedding_cache[query] = embedding
            results[pos] = embedding

    return results


def get_classifier():
    """Get or create query classifier model"""
    global _classifier
    if _classifier is None:
        try:
            chat_model = ChatBedrock(
                model_id="amazon.nova-micro-v1:0",
                model_kwargs={"temperature": 0, "max_tokens": 200},
                region_name=AWS_REGION,
                credentials_profile_name=AWS_PROFILE,
            )
            # Test the model
            chat_model.invoke([HumanMessage(content="Test message")])
            # Create structured output model
            _classifier = chat_model.with_structured_output(QueryClassification)
        except Exception as e:
            logger.error(f"Nova Micro initialization failed: {e}")
            raise e
    return _classifier


def get_rewriter():
    """Get or create query rewriter model"""
    global _rewriter
    if _rewriter is None:
        try:
            chat_model = ChatBedrock(
                model_id="amazon.nova-micro-v1:0",
                model_kwargs={"temperature": 0.3, "max_tokens": 300},
                region_name=AWS_REGION,
                credentials_profile_name=AWS_PROFILE,
            )
            _rewriter = chat_model.with_structured_output(QueryRewrite)
        except Exception as e:
            logger.error(f"Query rewriter initialization failed: {e}")
            raise e
    return _rewriter
