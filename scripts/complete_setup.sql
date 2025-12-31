-- Complete setup for RAG with OpenAI text-embedding-3-small (1536 dims)
-- Run this in Supabase SQL Editor

-- 1. Drop existing table
DROP TABLE IF EXISTS alpagino_documents CASCADE;

-- 2. Create table
CREATE TABLE alpagino_documents (
  id BIGSERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding VECTOR(1536) NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create index
CREATE INDEX ON alpagino_documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 4. Create RPC function for vector search
CREATE FUNCTION match_documents(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.0,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  distance float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    alpagino_documents.id,
    alpagino_documents.content,
    alpagino_documents.metadata,
    (alpagino_documents.embedding <=> query_embedding) AS distance
  FROM alpagino_documents
  ORDER BY alpagino_documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- 5. Verify
SELECT 'Setup complete! Ready for embeddings.' as status;
