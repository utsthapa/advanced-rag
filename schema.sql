-- Simple PostgreSQL Schema for RAG System
-- No fancy dollar quoting or complex blocks

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    content_tsvector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', content), 'B')
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    content_tsvector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,
    char_start INTEGER,
    char_end INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- Query logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    original_query TEXT NOT NULL,
    rewritten_query TEXT,
    query_type VARCHAR(50),
    search_method VARCHAR(20),
    results_count INTEGER,
    top_result_ids INTEGER[],
    user_feedback INTEGER,
    embedding_time_ms INTEGER,
    search_time_ms INTEGER,
    rerank_time_ms INTEGER,
    total_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector indexes (optimized for speed and recall)
-- Using HNSW for better query performance and recall (vs IVFFlat)
-- HNSW offers superior query performance and better recall at the same speed
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Additional performance indexes for fast lookups
CREATE INDEX IF NOT EXISTS chunks_content_trgm_idx ON document_chunks
USING gin (content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS chunks_id_idx ON document_chunks(id);

-- Full-text indexes
CREATE INDEX IF NOT EXISTS documents_fts_idx ON documents USING GIN(content_tsvector);
CREATE INDEX IF NOT EXISTS chunks_fts_idx ON document_chunks USING GIN(content_tsvector);

-- Other indexes
CREATE INDEX IF NOT EXISTS documents_doc_id_idx ON documents(doc_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_composite_idx ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS query_logs_created_idx ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS query_logs_search_method_idx ON query_logs(search_method);


-- Create functions separately to avoid quoting issues

-- Hybrid search function
CREATE OR REPLACE FUNCTION hybrid_search_score(
    vector_similarity REAL,
    fts_rank REAL,
    vector_weight REAL DEFAULT 0.7,
    fts_weight REAL DEFAULT 0.3
) RETURNS REAL
LANGUAGE SQL IMMUTABLE
AS 'SELECT (vector_weight * COALESCE(vector_similarity, 0)) + (fts_weight * COALESCE(fts_rank, 0))';

-- Analytics view
CREATE OR REPLACE VIEW search_analytics AS
SELECT
    DATE(created_at) as search_date,
    search_method,
    COUNT(*) as query_count,
    AVG(total_time_ms) as avg_response_time_ms,
    AVG(results_count) as avg_results_count,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_response_time_ms
FROM query_logs
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), search_method
ORDER BY search_date DESC, search_method;
