"""Document contextual compression"""

import logging

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from src.config import AWS_PROFILE, AWS_REGION
from src.models import CompressedDocument

logger = logging.getLogger(__name__)

# Global compressor instance
_claude_compressor = None

# Compression prompt
COMPRESSION_PROMPT = PromptTemplate.from_template(
    """You are an intelligent document compressor. Extract and preserve ALL information relevant to answering the query, even if it seems secondary or related.

Query: {query}

Original Document:
{document}

Your task:
1. Remove ONLY clearly irrelevant sentences that have NO connection to the query topic
2. Keep ALL sentences that mention or relate to the query topic, even tangentially
3. Preserve key facts, numbers, definitions, explanations, and any details that help answer the query
4. Maintain factual accuracy - don't paraphrase or summarize away important details
5. Extract 3-5 key points that directly address the query - include ALL relevant information in key points

Guidelines:
- Remove: ONLY sentences completely unrelated to the query topic (different diseases, unrelated studies, etc.)
- Keep: ANY information related to the query topic, including definitions, context, related facts
- Be conservative - when in doubt, KEEP the information
- Aim for 20-40% compression, preserving ALL relevant signal
- If a sentence mentions the query topic or related concepts, KEEP it
- Key points should be comprehensive and include all relevant details from the document

CRITICAL: If the document is relevant to the query, preserve as much information as possible. Don't over-compress.

Provide the compressed content and key points."""
)


def get_claude_compressor():
    """Get Claude model for contextual compression"""
    global _claude_compressor
    if _claude_compressor is None:
        try:
            chat_model = ChatBedrock(
                model_id="amazon.nova-micro-v1:0",
                model_kwargs={"temperature": 0, "max_tokens": 1000},
                region_name=AWS_REGION,
                credentials_profile_name=AWS_PROFILE,
            )
            _claude_compressor = chat_model.with_structured_output(CompressedDocument)
        except Exception as e:
            logger.error(f"Claude compressor initialization failed: {e}")
            raise e
    return _claude_compressor


def compress_document(query: str, document: dict) -> dict:
    """
    Compress a single document using Claude, removing irrelevant content

    Args:
        query: The search query
        document: Document dict with 'content' field

    Returns:
        Document with compressed content and metadata
    """
    compressor = get_claude_compressor()
    content = document.get("content", str(document))

    # Don't compress if already short
    if len(content) < 500:
        return {
            **document,
            "compressed_content": content,
            "compression_applied": False,
            "original_length": len(content),
            "compressed_length": len(content),
            "compression_ratio": 0.0,
        }

    try:
        prompt_text = COMPRESSION_PROMPT.format(query=query, document=content)
        result: CompressedDocument = compressor.invoke(
            [HumanMessage(content=prompt_text)]
        )

        original_len = len(content)
        compressed_len = len(result.compressed_content)
        compression_ratio = ((original_len - compressed_len) / original_len) * 100

        return {
            **document,
            "compressed_content": result.compressed_content,
            "original_content": content,
            "key_points": result.key_points,
            "compression_applied": True,
            "original_length": original_len,
            "compressed_length": compressed_len,
            "compression_ratio": compression_ratio,
            "compression_reasoning": result.reasoning,
        }

    except Exception as e:
        logger.error(f"Document compression failed: {e}")
        return {
            **document,
            "compressed_content": content,
            "compression_applied": False,
            "error": str(e),
        }


def compress_documents(query: str, documents: list, show_stats: bool = True) -> list:
    """
    Compress multiple documents using Claude

    Args:
        query: The search query
        documents: List of document dicts
        show_stats: Whether to print compression statistics

    Returns:
        List of compressed documents
    """
    if not documents:
        return []

    if show_stats:
        print(f"🗜️  Compressing {len(documents)} documents...")

    compressed = []
    total_original = 0
    total_compressed = 0

    for i, doc in enumerate(documents, 1):
        result = compress_document(query, doc)
        compressed.append(result)

        if result.get("compression_applied"):
            total_original += result["original_length"]
            total_compressed += result["compressed_length"]

            if show_stats:
                ratio = result["compression_ratio"]
                print(
                    f"   [{i}] {ratio:.1f}% reduction ({result['original_length']} → {result['compressed_length']} chars)"
                )

    if show_stats and total_original > 0:
        overall_ratio = ((total_original - total_compressed) / total_original) * 100
        token_savings = (total_original - total_compressed) // 4  # Rough token estimate
        print(f"\n   📊 Overall: {overall_ratio:.1f}% compression")
        print(f"   💰 Est. token savings: ~{token_savings:,} tokens")

    return compressed
