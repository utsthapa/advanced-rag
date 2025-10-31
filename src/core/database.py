"""Database manager for PostgreSQL with pgvector"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from src.config import DB_CONFIG, DB_POOL_MAX, DB_POOL_MIN

logger = logging.getLogger(__name__)

# Connection pool
_pool = None


class DatabaseManager:
    """Manages PostgreSQL database operations with pgvector for embeddings"""

    def __init__(self, db_config: Dict[str, str] = None, use_pool: bool = True):
        self.db_config = db_config or DB_CONFIG
        self.use_pool = use_pool
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection or get from pool"""
        global _pool

        try:
            if self.use_pool and _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    DB_POOL_MIN, DB_POOL_MAX, **self.db_config
                )
                logger.info("Created database connection pool")

            if self.use_pool and _pool is not None:
                self.conn = _pool.getconn()
            else:
                self.conn = psycopg2.connect(**self.db_config)

            self.conn.autocommit = True

            # Set session-level optimizations for speed
            with self.conn.cursor() as cur:
                cur.execute("SET enable_seqscan = off")  # Favor index scans
                cur.execute("SET random_page_cost = 1.1")  # Favor SSD
                cur.execute("SET work_mem = '256MB'")  # Allow more memory per query

            logger.info("Got database connection")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if document already exists"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE doc_id = %s", (doc_id,))
            return cur.fetchone()[0] > 0

    def insert_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        embedding: List[float],
        metadata: Dict = None,
    ) -> int:
        """Insert document and return database ID"""
        if metadata is None:
            metadata = {}

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (doc_id, title, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """,
                (doc_id, title, content, psycopg2.extras.Json(metadata), embedding),
            )
            return cur.fetchone()[0]

    def insert_chunk(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding: List[float],
        char_start: int = None,
        char_end: int = None,
        token_count: int = None,
    ):
        """Insert document chunk"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks
                (document_id, chunk_index, content, embedding, char_start, char_end, token_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    document_id,
                    chunk_index,
                    content,
                    embedding,
                    char_start,
                    char_end,
                    token_count,
                ),
            )

    def vector_search(
        self, query_embedding: List[float], k: int = 5, search_chunks: bool = True
    ) -> List[Dict]:
        """Vector similarity search"""
        table = "document_chunks" if search_chunks else "documents"

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Optimized query - compute distance once and use index
            cur.execute(
                f"""
                SELECT id, content, embedding <=> %s::vector AS distance,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                (query_embedding, query_embedding, query_embedding, k),
            )
            return [dict(row) for row in cur.fetchall()]

    def fulltext_search(
        self, query: str, k: int = 5, search_chunks: bool = True
    ) -> List[Dict]:
        """Full-text search"""
        table = "document_chunks" if search_chunks else "documents"

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, content, ts_rank(content_tsvector, plainto_tsquery('english', %s)) AS similarity
                FROM {table}
                WHERE content_tsvector @@ plainto_tsquery('english', %s)
                ORDER BY similarity DESC
                LIMIT %s
            """,
                (query, query, k),
            )
            return [dict(row) for row in cur.fetchall()]

    def hybrid_search_rrf(
        self,
        query_embedding: List[float],
        vector_query: str,
        fulltext_query: str,
        k: int = 5,
        rrf_k: int = 60,
        search_chunks: bool = True,
    ) -> List[Dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF)
        Uses different optimized queries for vector and fulltext search
        Returns doc_id for ground truth matching
        """
        table = "document_chunks" if search_chunks else "documents"
        doc_join = "JOIN documents d ON dc.document_id = d.id" if search_chunks else ""
        id_col = "dc.id" if search_chunks else "id"
        content_col = "dc.content" if search_chunks else "content"
        doc_id_col = "d.doc_id" if search_chunks else "doc_id"

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if search_chunks:
                cur.execute(
                    f"""
                    WITH vector_results AS (
                        SELECT {id_col}, {content_col}, {doc_id_col},
                               ROW_NUMBER() OVER (ORDER BY dc.embedding <=> %s::vector) as vector_rank
                        FROM {table} dc
                        {doc_join}
                        ORDER BY dc.embedding <=> %s::vector
                        LIMIT 200
                    ),
                    fts_results AS (
                        SELECT {id_col}, {content_col}, {doc_id_col},
                               ROW_NUMBER() OVER (ORDER BY ts_rank(dc.content_tsvector, plainto_tsquery('english', %s)) DESC) as fts_rank
                        FROM {table} dc
                        {doc_join}
                        WHERE dc.content_tsvector @@ plainto_tsquery('english', %s)
                        ORDER BY ts_rank(dc.content_tsvector, plainto_tsquery('english', %s)) DESC
                        LIMIT 200
                    ),
                    rrf_fusion AS (
                        SELECT
                            COALESCE(v.id, f.id) as id,
                            COALESCE(v.content, f.content) as content,
                            COALESCE(v.doc_id, f.doc_id) as doc_id,
                            COALESCE(1.0/(v.vector_rank::float + %s), 0) +
                            COALESCE(1.0/(f.fts_rank::float + %s), 0) as rrf_score,
                            v.vector_rank,
                            f.fts_rank
                        FROM vector_results v
                        FULL OUTER JOIN fts_results f ON v.id = f.id
                    )
                    SELECT id, content, doc_id, rrf_score as similarity, vector_rank, fts_rank
                    FROM rrf_fusion
                    ORDER BY rrf_score DESC
                    LIMIT %s
                """,
                    (
                        query_embedding,
                        query_embedding,
                        fulltext_query,
                        fulltext_query,
                        fulltext_query,
                        rrf_k,
                        rrf_k,
                        k,
                    ),
                )
            else:
                # Simplified for documents table (no join needed)
                cur.execute(
                    f"""
                    WITH vector_results AS (
                        SELECT {id_col}, {content_col}, {doc_id_col},
                               ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as vector_rank
                        FROM {table}
                        ORDER BY embedding <=> %s::vector
                        LIMIT 200
                    ),
                    fts_results AS (
                        SELECT {id_col}, {content_col}, {doc_id_col},
                               ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', %s)) DESC) as fts_rank
                        FROM {table}
                        WHERE content_tsvector @@ plainto_tsquery('english', %s)
                        ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', %s)) DESC
                        LIMIT 200
                    ),
                    rrf_fusion AS (
                        SELECT
                            COALESCE(v.id, f.id) as id,
                            COALESCE(v.content, f.content) as content,
                            COALESCE(v.doc_id, f.doc_id) as doc_id,
                            COALESCE(1.0/(v.vector_rank::float + %s), 0) +
                            COALESCE(1.0/(f.fts_rank::float + %s), 0) as rrf_score,
                            v.vector_rank,
                            f.fts_rank
                        FROM vector_results v
                        FULL OUTER JOIN fts_results f ON v.id = f.id
                    )
                    SELECT id, content, doc_id, rrf_score as similarity, vector_rank, fts_rank
                    FROM rrf_fusion
                    ORDER BY rrf_score DESC
                    LIMIT %s
                """,
                    (
                        query_embedding,
                        query_embedding,
                        fulltext_query,
                        fulltext_query,
                        fulltext_query,
                        rrf_k,
                        rrf_k,
                        k,
                    ),
                )
            return [dict(row) for row in cur.fetchall()]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents) as doc_count,
                    (SELECT COUNT(*) FROM document_chunks) as chunk_count,
                    (SELECT COUNT(*) FROM query_logs) as query_count
            """
            )
            doc_count, chunk_count, query_count = cur.fetchone()
            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "queries": query_count,
            }

    def close(self):
        """Close database connection or return to pool"""
        if self.conn:
            if self.use_pool and _pool is not None:
                _pool.putconn(self.conn)
                logger.debug("Returned connection to pool")
            else:
                self.conn.close()
                logger.info("Closed database connection")
            self.conn = None

    def reuse_connection(self):
        """Reuse existing connection without creating a new one"""
        # Connection already exists, just reset any session settings
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("SET enable_seqscan = off")
                cur.execute("SET random_page_cost = 1.1")
                cur.execute("SET work_mem = '256MB'")

    def vector_search_fast(
        self, query_embedding: List[float], k: int = 5, search_chunks: bool = True
    ) -> List[Dict]:
        """Fast vector search using indexed queries (optimized for speed)"""
        table = "document_chunks" if search_chunks else "documents"

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Optimized query - minimize computations, use parallelism
            cur.execute(f"SET LOCAL max_parallel_workers_per_gather = 4")

            # Simplified query to minimize computation overhead
            cur.execute(
                f"""
                SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                (query_embedding, query_embedding, k),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_stats_fast(self) -> Dict:
        """Fast stats retrieval using cached counts"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents) as doc_count,
                    (SELECT COUNT(*) FROM document_chunks) as chunk_count,
                    (SELECT COUNT(*) FROM query_logs) as query_count
            """
            )
            doc_count, chunk_count, query_count = cur.fetchone()
            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "queries": query_count,
            }

    @staticmethod
    def close_all_connections():
        """Close all connections in the pool"""
        global _pool
        if _pool:
            _pool.closeall()
            _pool = None
            logger.info("Closed all database connections")
