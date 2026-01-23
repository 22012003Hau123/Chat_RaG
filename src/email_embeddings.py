"""
Email Embeddings Service

Ingests email content into vector embeddings for RAG search.
- Chunks email content
- Generates embeddings using OpenAI
- Stores in email_embeddings table
"""

import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class EmailEmbeddingService:
    """Service to create and manage email embeddings."""
    
    def __init__(self):
        """Initialize with OpenAI and Supabase clients."""
        # OpenAI for embeddings
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.openai = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        
        # Supabase for storage
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE credentials required")
        self.supabase = create_client(url, key)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        
        logger.info("📧 EmailEmbeddingService initialized")
    
    def ingest_email(self, email: Dict) -> int:
        """
        Ingest a single email into embeddings.
        
        Args:
            email: Dict with id, subject, sender, recipient, date, content
            
        Returns:
            Number of chunks created
        """
        email_id = email.get("id")
        content = email.get("content", "")
        
        if not content or not content.strip():
            logger.warning(f"Email {email_id} has no content, skipping")
            return 0
        
        # Chunk the content
        chunks = self.text_splitter.split_text(content)
        logger.info(f"📧 Email {email_id[:8]}... split into {len(chunks)} chunks")
        
        # Create embeddings and insert
        inserted = 0
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                response = self.openai.embeddings.create(
                    input=chunk,
                    model=self.embedding_model
                )
                embedding = response.data[0].embedding
                
                # Prepare metadata
                metadata = {
                    "subject": email.get("subject", ""),
                    "sender_email": email.get("sender_email", ""),
                    "sent_at": email.get("sent_at", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                # Insert into database
                self.supabase.table("email_embeddings").insert({
                    "email_id": email_id,
                    "content": chunk,
                    "chunk_index": i,
                    "embedding": embedding,
                    "metadata": metadata
                }).execute()
                
                inserted += 1
                
            except Exception as e:
                logger.error(f"Error embedding chunk {i} of email {email_id}: {e}")
        
        return inserted
    
    def ingest_all_emails(self, emails: List[Dict]) -> Dict:
        """
        Ingest multiple emails.
        
        Args:
            emails: List of email dicts
            
        Returns:
            Summary dict with counts
        """
        total_emails = len(emails)
        total_chunks = 0
        successful = 0
        failed = 0
        
        for email in emails:
            try:
                chunks = self.ingest_email(email)
                if chunks > 0:
                    successful += 1
                    total_chunks += chunks
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to ingest email {email.get('id')}: {e}")
                failed += 1
        
        result = {
            "total_emails": total_emails,
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks
        }
        
        logger.info(f"📧 Ingestion complete: {result}")
        return result
    
    def clear_email_embeddings(self, email_id: Optional[str] = None) -> int:
        """
        Clear embeddings for a specific email or all emails.
        
        Args:
            email_id: Optional specific email to clear, or None for all
            
        Returns:
            Number of records deleted
        """
        try:
            if email_id:
                result = self.supabase.table("email_embeddings")\
                    .delete()\
                    .eq("email_id", email_id)\
                    .execute()
            else:
                # Delete all - be careful!
                result = self.supabase.table("email_embeddings")\
                    .delete()\
                    .neq("id", "00000000-0000-0000-0000-000000000000")\
                    .execute()
            
            count = len(result.data) if result.data else 0
            logger.info(f"🗑️ Cleared {count} email embeddings")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
            return 0
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about email embeddings."""
        try:
            result = self.supabase.table("email_embeddings")\
                .select("email_id", count="exact")\
                .execute()
            
            total_chunks = result.count or 0
            unique_emails = len(set(row["email_id"] for row in (result.data or [])))
            
            return {
                "total_chunks": total_chunks,
                "unique_emails": unique_emails
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_chunks": 0, "unique_emails": 0}


# Singleton
_embedding_service = None


def get_email_embedding_service() -> Optional[EmailEmbeddingService]:
    """Get or create EmailEmbeddingService instance."""
    global _embedding_service
    
    if _embedding_service is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        try:
            _embedding_service = EmailEmbeddingService()
        except Exception as e:
            logger.warning(f"Failed to initialize EmailEmbeddingService: {e}")
    
    return _embedding_service
