"""Reranking strategies for search results"""

import json
import logging
from typing import Dict, List

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

from src.config import AWS_PROFILE, AWS_REGION
from src.core import get_bedrock_runtime
from src.models import RelevanceScore

logger = logging.getLogger(__name__)

# Global reranker instances
_bge_reranker = None
_claude_reranker = None


def get_bge_reranker():
    """Load BGE reranker model (open source)"""
    global _bge_reranker
    if _bge_reranker is None:
        logger.info("Loading BGE reranker model...")
        _bge_reranker = CrossEncoder("BAAI/bge-reranker-large")
        logger.info("BGE reranker loaded successfully")
    return _bge_reranker


def get_claude_reranker():
    """Get Claude model for reranking"""
    global _claude_reranker
    if _claude_reranker is None:
        try:
            chat_model = ChatBedrock(
                model_id="amazon.nova-micro-v1:0",
                model_kwargs={"temperature": 0, "max_tokens": 150},
                region_name=AWS_REGION,
                credentials_profile_name=AWS_PROFILE,
            )
            _claude_reranker = chat_model.with_structured_output(RelevanceScore)
        except Exception as e:
            logger.error(f"Claude reranker initialization failed: {e}")
            raise e
    return _claude_reranker


def rerank_cohere(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank documents using Cohere Rerank v3 via AWS Bedrock"""
    if not documents:
        return []

    bedrock_runtime = get_bedrock_runtime()
    texts = [doc.get("content", str(doc)) for doc in documents]

    # Cohere Rerank v3.5 API payload for Bedrock
    # Process all documents to get best ranking, then return top_k
    # Limit to 100 documents max for API constraints
    max_docs = min(100, len(documents))
    payload = {
        "api_version": 2,
        "query": query,
        "documents": texts[:max_docs],
        "top_n": max_docs,  # Get ranking for all, then slice to top_k
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId="cohere.rerank-v3-5:0", body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())

        # Rerank results
        reranked = []
        for item in result.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            doc_copy = documents[idx].copy()
            doc_copy["rerank_score"] = score
            doc_copy["original_rank"] = idx + 1
            reranked.append(doc_copy)

        # Return only top_k results (already sorted by Cohere)
        return reranked[:top_k]

    except Exception as e:
        logger.error(f"Cohere reranking failed: {e}")
        return documents[:top_k]


def rerank_bge(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank documents using BGE reranker (open source)"""
    if not documents:
        return []

    reranker = get_bge_reranker()
    texts = [doc.get("content", str(doc)) for doc in documents]
    pairs = [[query, text] for text in texts]

    try:
        scores = reranker.predict(pairs)

        # Combine documents with scores
        reranked = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_rank"] = idx + 1
            reranked.append(doc_copy)

        # Sort by score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        logger.error(f"BGE reranking failed: {e}")
        return documents[:top_k]


def rerank_claude(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank documents using Claude (LLM-based reranking)"""
    if not documents:
        return []

    reranker = get_claude_reranker()
    rerank_prompt = PromptTemplate.from_template(
        """Rate the relevance of the following document to the query on a scale of 0.0 to 1.0.

Query: {query}

Document:
{document}

Consider:
- Does the document directly answer the query?
- Is the information accurate and specific?
- How well does it match the user's intent?

Provide a relevance score between 0.0 (not relevant) and 1.0 (highly relevant)."""
    )

    try:
        reranked = []
        for idx, doc in enumerate(documents):
            content = doc.get("content", str(doc))

            # Truncate very long documents
            if len(content) > 2000:
                content = content[:2000] + "..."

            prompt_text = rerank_prompt.format(query=query, document=content)
            result: RelevanceScore = reranker.invoke(
                [HumanMessage(content=prompt_text)]
            )

            doc_copy = doc.copy()
            doc_copy["rerank_score"] = result.score
            doc_copy["rerank_reasoning"] = result.reasoning
            doc_copy["original_rank"] = idx + 1
            reranked.append(doc_copy)

        # Sort by score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        logger.error(f"Claude reranking failed: {e}")
        return documents[:top_k]


def rerank_ensemble(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
    cohere_weight: float = 0.7,
    bge_weight: float = 0.3,
) -> List[Dict]:
    """
    Ensemble reranking combining Cohere and BGE rerankers.

    Combines scores from multiple rerankers using weighted average.
    This often outperforms single rerankers by leveraging their different strengths.

    Args:
        query: Search query
        documents: List of documents to rerank
        top_k: Number of top results to return
        cohere_weight: Weight for Cohere scores (default 0.7)
        bge_weight: Weight for BGE scores (default 0.3)

    Returns:
        Reranked documents sorted by combined score
    """
    if not documents:
        return []

    if len(documents) == 1:
        # Single document - return as is
        return documents

    try:
        # Get scores from both rerankers
        # Use larger k to get scores for all candidates
        cohere_results = rerank_cohere(query, documents, len(documents))
        bge_results = rerank_bge(query, documents, len(documents))

        # Create mapping of doc_id -> scores
        # Use doc_id or id as unique identifier
        def get_doc_key(doc: Dict) -> str:
            return str(doc.get("doc_id") or doc.get("id") or "")

        cohere_scores = {}
        for result in cohere_results:
            key = get_doc_key(result)
            cohere_scores[key] = result.get("rerank_score", 0.0)

        bge_scores = {}
        for result in bge_results:
            key = get_doc_key(result)
            bge_scores[key] = result.get("rerank_score", 0.0)

        # Normalize scores to [0, 1] if needed (Cohere is already 0-1, BGE may need normalization)
        # BGE scores can be negative, so we'll use min-max normalization
        if bge_scores:
            bge_min = min(bge_scores.values())
            bge_max = max(bge_scores.values())
            if bge_max > bge_min:
                bge_scores = {
                    k: (v - bge_min) / (bge_max - bge_min)
                    for k, v in bge_scores.items()
                }

        # Combine scores with weighted average
        combined_scores = {}
        all_keys = set(cohere_scores.keys()) | set(bge_scores.keys())

        for key in all_keys:
            cohere_score = cohere_scores.get(key, 0.0)
            bge_score = bge_scores.get(key, 0.0)

            # Weighted combination
            combined_score = cohere_weight * cohere_score + bge_weight * bge_score
            combined_scores[key] = combined_score

        # Sort documents by combined score
        reranked = []
        for doc in documents:
            key = get_doc_key(doc)
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = combined_scores.get(key, 0.0)
            doc_copy["cohere_score"] = cohere_scores.get(key, 0.0)
            doc_copy["bge_score"] = bge_scores.get(key, 0.0)
            doc_copy["original_rank"] = documents.index(doc) + 1
            reranked.append(doc_copy)

        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    except Exception as e:
        logger.error(f"Ensemble reranking failed: {e}, falling back to Cohere")
        # Fallback to Cohere if ensemble fails
        return rerank_cohere(query, documents, top_k)
