-- ============================================================
-- Email Embeddings Table for RAG Chat
-- Purpose: Store chunked email content with vector embeddings
-- ============================================================

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create email_embeddings table
CREATE TABLE IF NOT EXISTS email_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Reference to original email
    email_id UUID REFERENCES email(id) ON DELETE CASCADE,
    
    -- Content chunk (email content may be split into multiple chunks)
    content TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    
    -- Vector embedding (1536 dimensions for OpenAI text-embedding-3-small)
    embedding vector(1536),
    
    -- Metadata for filtering and display
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for vector similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS email_embeddings_embedding_idx 
ON email_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create index for email_id lookup
CREATE INDEX IF NOT EXISTS email_embeddings_email_id_idx 
ON email_embeddings(email_id);

-- Create RPC function for similarity search
CREATE OR REPLACE FUNCTION match_email_embeddings(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    email_id UUID,
    content TEXT,
    metadata JSONB,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ee.id,
        ee.email_id,
        ee.content,
        ee.metadata,
        1 - (ee.embedding <=> query_embedding) AS similarity
    FROM email_embeddings ee
    WHERE 1 - (ee.embedding <=> query_embedding) > match_threshold
    ORDER BY ee.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Grant permissions (adjust as needed)
-- GRANT ALL ON email_embeddings TO authenticated;
-- GRANT EXECUTE ON FUNCTION match_email_embeddings TO authenticated;

COMMENT ON TABLE email_embeddings IS 'Stores email content chunks with vector embeddings for RAG search';
COMMENT ON COLUMN email_embeddings.embedding IS 'OpenAI text-embedding-3-small vector (1536 dimensions)';
COMMENT ON COLUMN email_embeddings.chunk_index IS 'Index of chunk within the email (for long emails split into multiple chunks)';
