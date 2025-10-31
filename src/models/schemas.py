"""Pydantic models for structured outputs"""

from pydantic import BaseModel, Field


class RelevanceScore(BaseModel):
    """Structured output for Claude-based reranking"""

    score: float = Field(
        description="Relevance score between 0.0 and 1.0, where 1.0 is most relevant"
    )
    reasoning: str = Field(
        description="Brief explanation of why this score was assigned"
    )


class CompressedDocument(BaseModel):
    """Structured output for document compression"""

    compressed_content: str = Field(
        description="Compressed version containing only relevant information for the query"
    )
    removed_percentage: float = Field(
        description="Percentage of original content removed (0-100)"
    )
    key_points: list[str] = Field(
        description="List of 3-5 key points extracted from the document"
    )
    reasoning: str = Field(description="Brief explanation of compression strategy")


class QueryRewrite(BaseModel):
    """Structured output for query rewriting"""

    vector_optimized: str = Field(
        description="Query optimized for semantic/vector search - more conceptual and descriptive"
    )
    fulltext_optimized: str = Field(
        description="Query optimized for keyword/full-text search - specific terms and synonyms"
    )
    reasoning: str = Field(description="Brief explanation of the rewriting strategy")


class QueryClassification(BaseModel):
    """Structured output for query classification"""

    needs_retrieval: bool = Field(
        description="Whether this query needs document retrieval"
    )
    query_type: str = Field(
        description="Type: factual, mathematical, definition, common_knowledge, scientific_basic, or complex"
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(
        description="Brief explanation of the classification decision"
    )
