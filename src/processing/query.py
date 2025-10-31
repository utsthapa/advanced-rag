"""Query classification and rewriting"""

import logging
import time

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from src.config import CLASSIFICATION_CONFIDENCE_THRESHOLD
from src.core import get_classifier, get_rewriter

logger = logging.getLogger(__name__)

# Classification prompt
CLASSIFY_PROMPT = PromptTemplate.from_template(
    """
Analyze the following query and determine if it needs document retrieval from a SCIENTIFIC LITERATURE database.

IMPORTANT CONTEXT: The available documents are from BeIR SciFact - a scientific fact verification dataset focused on biomedical research papers.

CRITICAL RULE: When in doubt, ALWAYS use retrieval. This is a scientific RAG system.

Query Types:
- factual: Simple NON-SCIENTIFIC facts (e.g., "What is the capital of France?")
- mathematical: Pure math calculations (e.g., "What's 15% of 200?")
- definition: ONLY basic everyday non-scientific definitions (e.g., "What does USA stand for?")
- complex: ANY scientific, medical, or technical question - USE RETRIEVAL

Guidelines:
✅ USE RETRIEVAL for:
- ANY question about diseases, viruses, bacteria, medical conditions
- ANY question about scientific processes or mechanisms
- ANY question about research findings
- ANY question mentioning scientific terms
- ANY "how does X work" question about scientific topics

❌ SKIP RETRIEVAL ONLY for:
- Basic geography (capitals, countries)
- Basic history (famous dates)
- Basic math calculations
- Non-scientific definitions

Query: {query}

REMEMBER: If the query mentions ANY scientific, medical, or technical term, set needs_retrieval=TRUE.
"""
)

# Rewriting prompt
REWRITE_PROMPT = PromptTemplate.from_template(
    """
Rewrite the following query to optimize it for different search methods while PRESERVING the original query's terms:

CRITICAL RULES:
- PRESERVE ALL original terms from the query - do NOT replace or expand them
- DO NOT make assumptions about abbreviations (e.g., don't expand "covid" to "COVID-19")
- DO NOT add information not present in the original query
- Only ADD synonyms alongside original terms, never replace them

VECTOR SEARCH optimization:
- Make queries more conceptual and descriptive by adding synonyms
- Preserve original terms and add semantic relationships
- Focus on meaning and context
- Example: "what causes covid" → "what causes covid disease virus pandemic" (adding synonyms, keeping "covid")

FULL-TEXT SEARCH optimization:
- Preserve all original keywords exactly as written
- ADD technical terms and synonyms alongside originals
- Include variations but keep original terms
- Example: "what causes covid" → "causes covid coronavirus sars-cov" (keeping "covid", adding related terms)

Original query: {query}

Provide optimized versions for both search methods, ensuring original terms are preserved.
"""
)


def classify_query(query: str, verbose: bool = True) -> bool:
    """
    Classify query to determine if retrieval is needed

    Args:
        query: The user's query
        verbose: Whether to print classification results

    Returns:
        bool: True if retrieval is needed, False otherwise
    """
    t0 = time.perf_counter()

    try:
        classifier = get_classifier()
        prompt_text = CLASSIFY_PROMPT.format(query=query)
        result = classifier.invoke([HumanMessage(content=prompt_text)])

        took = time.perf_counter() - t0

        # Decision logic based on confidence threshold
        should_retrieve = (
            result.needs_retrieval
            or result.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD
        )

        if verbose:
            decision = "SKIP_RETRIEVAL" if not should_retrieve else "USE_RETRIEVAL"
            print(
                f"🤖 Classification: {decision} (confidence: {result.confidence:.2f}, type: {result.query_type})"
            )
            if not should_retrieve:
                print(f"   💡 Reason: {result.reasoning}")

        return should_retrieve

    except Exception as e:
        took = time.perf_counter() - t0
        logger.error(f"Classification failed after {took:.2f}s: {e}")
        raise e


def rewrite_query(query: str, verbose: bool = True) -> dict:
    """
    Rewrite query for optimal performance across different search methods

    Args:
        query: The original query
        verbose: Whether to print rewriting results

    Returns:
        dict: Contains original, vector_optimized, fulltext_optimized queries and reasoning
    """
    try:
        rewriter = get_rewriter()
        prompt_text = REWRITE_PROMPT.format(query=query)
        result = rewriter.invoke([HumanMessage(content=prompt_text)])

        if verbose:
            print(f"🔄 Query Rewriting:")
            print(f"   📝 Original: {query}")
            print(f"   🔤 Vector: {result.vector_optimized}")
            print(f"   📝 Full-text: {result.fulltext_optimized}")
            print(f"   💡 Strategy: {result.reasoning}")

        return {
            "original": query,
            "vector_optimized": result.vector_optimized,
            "fulltext_optimized": result.fulltext_optimized,
            "reasoning": result.reasoning,
        }

    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        if verbose:
            print(f"⚠️  Query rewriting failed, using fallback strategy: {e}")

        # Fallback strategy
        return {
            "original": query,
            "vector_optimized": query + " definition meaning explanation",
            "fulltext_optimized": query + " " + " ".join(query.split()),
            "reasoning": "Rewriting failed, using fallback strategy",
        }
