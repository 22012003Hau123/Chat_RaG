"""
Email RAG Chain

Separate RAG pipeline for email search and Q&A.
Uses email_embeddings table instead of documents table.
"""

import os
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session_manager import ConversationSession

from openai import OpenAI
from supabase import create_client
from langchain_core.documents import Document

from src.email_prompt import create_email_messages

logger = logging.getLogger(__name__)


class EmailRAGChain:
    """
    RAG pipeline for email search.
    
    Similar to RAGChain but queries email_embeddings table.
    """
    
    def __init__(self):
        """Initialize email RAG chain."""
        logger.info("📧 Initializing Email RAG Chain...")
        
        # OpenAI clients
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        self.openai = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4o"
        
        # Supabase
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE credentials required")
        
        self.supabase = create_client(url, key)
        
        logger.info("📧 Email RAG Chain initialized!")
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query."""
        response = self.openai.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def retrieve(self, question: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant email chunks.
        
        Prioritizes direct ID lookup if a UUID is present in the query.
        Falls back to vector search otherwise.
        """
        try:
            # Check for UUID in question (Direct Lookup)
            # UUID Pattern: 8-4-4-4-12 hex digits
            import re
            uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
            uuid_match = re.search(uuid_pattern, question)
            
            if uuid_match:
                email_id = uuid_match.group(0)
                logger.info(f"🔍 Detected Email ID in query: {email_id}. Performing direct lookup.")
                
                # Fetch directly from embeddings table
                result = self.supabase.table("email_embeddings")\
                    .select("*")\
                    .eq("email_id", email_id)\
                    .execute()
                
                if result.data:
                    # Found it!
                    documents = []
                    for row in result.data:
                        doc = Document(
                            page_content=row.get("content", ""),
                            metadata={
                                "email_id": row.get("email_id"),
                                "similarity": 1.0, # Exact match
                                **(row.get("metadata") or {})
                            }
                        )
                        documents.append(doc)
                    
                    logger.info(f"📧 Direct lookup found {len(documents)} chunks")
                    return documents
                else:
                    logger.warning(f"⚠️ Direct lookup for {email_id} returned no results. Falling back to vector search.")
            
            # --- Vector Search (Fallback) ---
            
            # Generate query embedding
            query_embedding = self._embed_query(question)
            
            # Call Supabase RPC for similarity search
            result = self.supabase.rpc(
                "match_email_embeddings",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.25,  # Balanced threshold for finding relevant items
                    "match_count": k
                }
            ).execute()
            
            # Convert to Document objects
            documents = []
            for row in (result.data or []):
                doc = Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "email_id": row.get("email_id"),
                        "similarity": row.get("similarity"),
                        **(row.get("metadata") or {})
                    }
                )
                documents.append(doc)
                
                # Debug log
                logger.debug(f"🔍 Found: {doc.metadata.get('subject')} (Sim: {doc.metadata.get('similarity')})")
            
            logger.info(f"📧 Retrieved {len(documents)} email chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            # Return error as document to debug
            return [Document(
                page_content=f"SYSTEM_ERROR: {str(e)}",
                metadata={"source": "system", "subject": "Error"}
            )]
    
    def query(
        self,
        question: str,
        k: int = 10,  # Increase default retrieval count
        session: Optional['ConversationSession'] = None
    ) -> Dict[str, Any]:
        """
        Complete email RAG query: Retrieve → Compose → Generate
        
        Args:
            question: User's question about emails
            k: Number of results to retrieve
            session: Optional conversation session for history
            
        Returns:
            Dict with answer and sources
        """
        logger.info(f"📧 Email Query: {question[:50]}...")
        
        # Get conversation history
        history = []
        if session:
            context = session.get_context()
            if 'chat_history' in context:
                for msg in context['chat_history']:
                    role = 'user' if hasattr(msg, 'type') and msg.type == 'human' else 'assistant'
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    history.append({'role': role, 'content': content})
        
        # Retrieve relevant emails
        documents = self.retrieve(question, k=k)
        
        # Create prompt
        messages = create_email_messages(question, documents, history)
        
        # Generate answer
        try:
            response = self.openai.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Update session if provided
            if session:
                session.add_exchange(question, answer)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"Erreur: {str(e)}"
        
        # Format sources
        sources = []
        seen = set()
        for doc in documents:
            email_id = doc.metadata.get("email_id", "")
            if email_id and email_id not in seen:
                seen.add(email_id)
                sender = doc.metadata.get("sender_email", "Inconnu")
                subject = doc.metadata.get("subject", "")
                sources.append(f"📧 {sender}: {subject}")
        
        return {
            "answer": answer,
            "sources": sources,
            "documents": documents
        }


# Singleton
_email_rag_chain = None


def get_email_rag_chain() -> Optional[EmailRAGChain]:
    """Get or create EmailRAGChain instance."""
    global _email_rag_chain
    
    if _email_rag_chain is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        try:
            _email_rag_chain = EmailRAGChain()
        except Exception as e:
            logger.warning(f"Failed to initialize EmailRAGChain: {e}")
    
    return _email_rag_chain
