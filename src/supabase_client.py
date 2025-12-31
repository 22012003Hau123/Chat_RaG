"""
Supabase Client Module

Handles all interactions with Supabase:
- Vector operations (pgvector)
- Image storage
- Document metadata

Usage:
    client = SupabaseClient()
    client.insert_documents(chunks, embeddings)
    image_url = client.upload_image(image_bytes, "doc_page1.png")
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
from langchain_core.documents import Document
import numpy as np


class SupabaseClient:
    """Client for Supabase vector database and storage operations."""
    
    def __init__(self):
        """Initialize Supabase client with credentials from environment."""
        url = os.getenv('SUPABASE_URL')
        # Check both possible key names
        key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in environment. "
                "Check your .env file."
            )
        
        self.client: Client = create_client(url, key)
        self.storage = self.client.storage
        print(f"✓ Connected to Supabase: {url}")
    
    def insert_documents(
        self,
        chunks: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> int:
        """
        Insert document chunks with embeddings into Supabase.
        
        Args:
            chunks: List of Document objects with content and metadata
            embeddings: NumPy array of embeddings (shape: [n_chunks, dimension])
            batch_size: Number of documents to insert per batch
            
        Returns:
            Number of documents inserted
        """
        print(f"\nInserting {len(chunks)} documents to Supabase...")
        
        total_inserted = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Prepare batch data
            rows = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                row = {
                    'content': chunk.page_content,
                    'embedding': embedding.tolist(),  # Convert to list for JSON
                    'metadata': chunk.metadata
                }
                rows.append(row)
            
            # Insert batch
            try:
                result = self.client.table('alpagino_documents').insert(rows).execute()
                batch_count = len(result.data) if result.data else len(rows)
                total_inserted += batch_count
                print(f"  Inserted batch {i//batch_size + 1}: {batch_count} documents")
            except Exception as e:
                print(f"  Error inserting batch: {e}")
                continue
        
        print(f"✓ Total inserted: {total_inserted} documents")
        return total_inserted
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector (shape: [dimension])
            k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "doc.pdf"})
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        # Call Supabase RPC function for vector search
        # Note: You need to create this function in Supabase SQL Editor
        try:
            # Convert embedding to list if needed (OpenAI returns list, numpy needs conversion)
            if isinstance(query_embedding, list):
                embedding_list = query_embedding
            else:
                embedding_list = query_embedding.tolist()
            
            params = {
                'query_embedding': embedding_list,
                'match_count': k
            }
            
            if filter_metadata:
                params['filter'] = filter_metadata            
            # Call new match_document RPC function
            result = self.client.rpc('match_documents_1536', params).execute()
            
            # Convert results to Documents
            documents = []
            for row in result.data:
                doc = Document(
                    page_content=row['content'],
                    metadata=row['metadata']
                )
                # Function returns similarity directly (0-1, higher = more similar)
                similarity = row['similarity']
                
                if score_threshold is None or similarity >= score_threshold:
                    documents.append((doc, similarity))
            
            return documents[:k]
            
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []
    
    def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        bucket: str = 'alpagino'
    ) -> str:
        """
        Upload image to Supabase Storage.
        
        Args:
            image_bytes: Image file content as bytes
            filename: Name for the uploaded file
            bucket: Storage bucket name
            
        Returns:
            Public URL of uploaded image
        """
        try:
            # Try to remove existing file first (ignore error if doesn't exist)
            try:
                self.storage.from_(bucket).remove([filename])
            except:
                pass  # File doesn't exist, that's OK
            
            # Upload to storage
            self.storage.from_(bucket).upload(
                filename,
                image_bytes,
                file_options={"content-type": "image/png"}
            )
            
            # Get public URL
            url = self.storage.from_(bucket).get_public_url(filename)
            return url
            
        except Exception as e:
            print(f"Error uploading image {filename}: {e}")
            return ""
    
    def delete_image(
        self,
        filename: str,
        bucket: str = 'alpagino'
    ) -> bool:
        """
        Delete image from Supabase Storage.
        
        Args:
            filename: Name of file to delete
            bucket: Storage bucket name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.storage.from_(bucket).remove([filename])
            return True
        except Exception as e:
            print(f"Error deleting image {filename}: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in the database."""
        try:
            result = self.client.table('alpagino_documents').select('id', count='exact').execute()
            return result.count if hasattr(result, 'count') else 0
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def clear_all_documents(self) -> bool:
        """
        Delete all documents from the database.
        WARNING: This is destructive!
        """
        try:
            self.client.table('alpagino_documents').delete().neq('id', 0).execute()
            print("✓ Cleared all documents from database")
            return True
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False
