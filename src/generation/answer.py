"""Answer generation using LLMs"""

import logging
import time
from typing import Dict, List

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from src.config import AWS_PROFILE, AWS_REGION

logger = logging.getLogger(__name__)

# Global answer generator
_answer_generator = None

# Answer generation prompt - STRICT: Only use provided documents, synthesize what's there
ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a document-only assistant. You MUST answer the question using ONLY information from the provided context documents below. You are FORBIDDEN from using any external knowledge, general knowledge, or information not explicitly stated in these documents.

Question: {query}

Context Documents (ONLY SOURCE OF INFORMATION):
{context}

CRITICAL RULES:
1. EVERY fact, claim, or piece of information in your answer MUST come directly from the context documents above
2. Use citations [1], [2], [3] etc. to show which document each piece of information comes from
3. Synthesize information from multiple documents when they relate to the same point - combine related facts into coherent sentences
4. Provide a clear, well-structured answer that reads naturally - don't just quote the documents verbatim
5. If the context documents contain the answer, provide a comprehensive response that addresses the question fully
6. If the context documents do NOT contain enough information, clearly state: "The provided documents do not contain sufficient information to fully answer this question." Then provide what information IS available from the documents.

STRICT PROHIBITIONS:
- DO NOT use any knowledge that is NOT in the context documents
- DO NOT infer, assume, or extrapolate beyond what is explicitly stated
- DO NOT add background information, definitions, or context unless they appear in the documents
- DO NOT combine document information with external knowledge
- DO NOT make connections between facts unless those connections are explicitly stated in the documents

WHAT TO DO:
- Read all context documents carefully
- Identify every relevant piece of information that addresses the question
- Synthesize these pieces into a coherent, natural-sounding answer
- Cite each fact with [N] where N is the document number
- Write as if explaining to someone - but ONLY using information from the documents

Example: If asked "What causes COVID-19?" and documents show:
[1] COVID-19 is caused by SARS-CoV-2
[2] SARS-CoV-2 emerged in Wuhan in late 2019

Good answer: "COVID-19 is caused by SARS-CoV-2 [1], a virus that emerged in Wuhan in late 2019 [2]."

Bad answer: "COVID-19 is caused by SARS-CoV-2 [1], which is a respiratory virus that spreads through droplets." (The droplet information is NOT in the documents, so it cannot be included.)

Now provide your answer using ONLY the information in the context documents above. Every sentence must be traceable to a specific document citation.
"""
)


def get_answer_generator():
    """Get Claude model for answer generation with strict settings to prevent hallucinations"""
    global _answer_generator
    if _answer_generator is None:
        try:
            _answer_generator = ChatBedrock(
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                # Set temperature to 0 for maximum determinism and reduced hallucination
                # This ensures the model sticks closely to the provided context
                model_kwargs={"temperature": 0, "max_tokens": 2000},
                region_name=AWS_REGION,
                credentials_profile_name=AWS_PROFILE,
            )
        except Exception as e:
            logger.error(f"Answer generator initialization failed: {e}")
            raise e
    return _answer_generator


def generate_answer(
    query: str, documents: List[Dict], use_compression: bool = True
) -> Dict:
    """
    Generate an answer to the query using retrieved documents

    Args:
        query: User's question
        documents: Retrieved documents (can be compressed or original)
        use_compression: Whether documents are compressed

    Returns:
        Dict with answer, sources, and metadata
    """
    if not documents:
        return {
            "answer": "I could not find any relevant documents in the database to answer your question.",
            "sources": [],
            "query": query,
        }

    # Check if documents have very low relevance scores
    avg_relevance = 0
    relevance_scores = []
    for doc in documents:
        score = doc.get("similarity", doc.get("rerank_score", 0))
        relevance_scores.append(score)
        avg_relevance += score

    avg_relevance = avg_relevance / len(documents) if documents else 0

    # Warn if average relevance is very low - low relevance may indicate irrelevant documents
    context_warning = ""
    if avg_relevance < 0.1:
        logger.warning(f"Low relevance scores detected: avg={avg_relevance:.3f}")
        # Add explicit warning to context that documents may not be relevant
        context_warning = "\n\n⚠️ WARNING: The retrieved documents have very low relevance scores. Use ONLY explicit information from them, and state clearly if they do not contain relevant information.\n"

    # Build context from documents
    # Only include valid documents with actual content - filter out any invalid/empty entries
    context_parts = []
    sources = []
    valid_doc_index = 1  # Start numbering from 1 for citations

    for doc in documents:
        # Use compressed content if available, otherwise original
        # Only extract the actual text content - no metadata or extra fields
        content = doc.get("compressed_content") or doc.get("content", "")

        # Ensure content is a string and not empty
        if not content or not isinstance(content, str):
            logger.warning(f"Document has invalid content, skipping")
            continue

        # Remove any potential metadata that might have leaked into content
        # Ensure we only use the actual document text
        content = content.strip()

        if not content:
            logger.debug(f"Skipping empty document")
            continue

        # Include key points if available (they often contain additional relevant information)
        key_points = doc.get("key_points", [])
        if key_points and isinstance(key_points, list) and key_points:
            key_points_text = "Key points: " + "; ".join(
                str(kp) for kp in key_points if kp
            )
            # Append key points to content for richer context
            content = f"{content}\n{key_points_text}"

        # Add document to context - only the actual content text
        # Use sequential numbering for valid documents only
        context_parts.append(f"[{valid_doc_index}] {content}")

        # Track source with cleaned content
        sources.append(
            {
                "number": valid_doc_index,
                "content": content,  # Only the actual text content
                "similarity": doc.get("similarity", doc.get("rerank_score", 0)),
                "doc_id": doc.get("doc_id", doc.get("id")),
                "key_points": doc.get("key_points", []),
            }
        )

        valid_doc_index += 1

    # Safety check: if no valid documents after filtering, return early
    if not context_parts:
        logger.warning("No valid documents with content after filtering")
        return {
            "answer": "I could not find any valid documents with content to answer your question. All retrieved documents were empty or invalid.",
            "sources": [],
            "query": query,
            "avg_relevance": avg_relevance,
        }

    context = "\n\n".join(context_parts)

    # Add warning if documents are low relevance
    if avg_relevance < 0.1:
        context = context_warning + context

    # Generate answer with strict context-only enforcement
    generator = get_answer_generator()
    prompt_text = ANSWER_PROMPT.format(query=query, context=context)

    # Log context length for debugging
    logger.debug(
        f"Answer generation context length: {len(context)} characters, {len(documents)} documents"
    )

    try:
        start = time.perf_counter()
        response = generator.invoke([HumanMessage(content=prompt_text)])
        generation_time = time.perf_counter() - start

        answer = response.content

        # Ensure answer is not None or empty
        if not answer or (isinstance(answer, str) and answer.strip() == ""):
            logger.warning("Answer generation returned empty response")
            answer = "The answer generation returned an empty response. Please try again or check the retrieved documents."

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "generation_time": generation_time,
            "num_documents": len(documents),
            "used_compression": use_compression,
            "avg_relevance": avg_relevance,
        }

    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        return {
            "answer": f"Error generating answer: {str(e)}. Please check the logs for more details.",
            "sources": sources,
            "query": query,
            "error": str(e),
            "generation_time": 0,
        }
