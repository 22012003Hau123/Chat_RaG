"""
RAG Chain - Complete Retrieval-Augmented Generation Pipeline

This module orchestrates the full RAG workflow:
1. Retrieve relevant documents from vector store
2. Compose prompt with context
3. Generate answer using LLM

Usage:
    rag = RAGChain()
    answer = rag.query("What is RAG?")
"""

import os
import yaml
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session_manager import ConversationSession

# Import our custom modules
from src.retriever import SupabaseRetriever
from src.prompt import create_messages_format
from src.configuration import resolve_config_path

# OpenAI for LLM
from openai import OpenAI

# LangChain document structure
from langchain_core.documents import Document


class RAGChain:
    """
    Complete RAG pipeline that integrates retrieval and generation.
    
    The RAG Chain combines:
    - Document retrieval (from Supabase vector database)
    - Prompt engineering (context injection)
    - LLM generation (OpenAI)
    - Session-based memory for conversation context
    
    This is the "brain" of the RAG system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG chain with all components.
        
        What happens during initialization:
        1. Load configuration
        2. Initialize Supabase retriever and embeddings
        3. Initialize embedding model
        4. Initialize OpenAI client
        
        Args:
            config_path: Optional explicit path to configuration file
        """
        print("Initializing RAG Chain...")

        resolved_config_path = resolve_config_path(config_path)
        print(f"Loading configuration from: {resolved_config_path}")

        # Load configuration
        self.config = self._load_config(resolved_config_path)
        
        # Initialize Supabase retriever
        print("\nInitializing Supabase retriever...")
        self.retriever = SupabaseRetriever(self.config)
        
        # Initialize OpenAI client
        print("\nInitializing LLM...")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it: export OPENAI_API_KEY='your-key-here'"
            )
        self.client = OpenAI(api_key=api_key)
        self.llm_config = self.config['llm']
        
        # Retrieval settings
        self.retrieval_config = self.config['retrieval']
        
        print("\nRAG Chain initialized successfully!")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def retrieve(self, question: str, method: str = "mmr_rerank") -> List[Document]:
        """
        Retrieve relevant documents for a question.
        
        This is Step 1 of the RAG pipeline: RETRIEVE
        
        Args:
            question: User's question
            method: Search method - "similarity", "mmr", "rerank", or "mmr_rerank" (default)
            
        Returns:
            List of relevant Document objects
        """
        k = self.retrieval_config.get('top_k', 5)
        lambda_mult = self.retrieval_config.get('mmr_lambda', 0.5)
        fetch_k = self.retrieval_config.get('fetch_k', 20)
        
        if method == "similarity":
            results = self.retriever.similarity_search(
                query=question,
                k=k
            )
        elif method == "mmr":
            results = self.retriever.mmr_search(
                query=question,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
        elif method == "rerank":
            results = self.retriever.rerank_search(
                query=question,
                k=k,
                fetch_k=fetch_k
            )
        elif method == "mmr_rerank":
            # BEST: Combined MMR + Rerank for diverse AND accurate results
            results = self.retriever.mmr_rerank_search(
                query=question,
                k=k,
                fetch_k=50,  # Fetch more for better diversity
                mmr_k=20,    # MMR selects diverse subset
                lambda_mult=lambda_mult
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}. Use: similarity, mmr, rerank, or mmr_rerank")
        
        # Extract just the documents (without scores)
        documents = [doc for doc, score in results]
        return documents
    
    def _is_chitchat(self, question: str) -> bool:
        """
        Check if question is chit-chat / casual conversation (not document-related).
        Return True for greetings, self-introduction, small talk, etc.
        """
        question_lower = question.lower()
        
        # Greetings
        greetings = ['hello', 'hi', 'bonjour', 'salut', 'hey', 'good morning', 'good afternoon', 'coucou', 'xin chào']
        if any(greeting in question_lower for greeting in greetings):
            return True
        
        # Self-introduction / name exchange (Vietnamese)
        intro_patterns_vn = [
            'tên là', 'tên tôi', 'tên mình', 'bạn tên', 'tên gì',
            'bạn là ai', 'ai vậy', 'ai đó'
        ]
        if any(pattern in question_lower for pattern in intro_patterns_vn):
            return True
        
        # Self-introduction / name exchange (French/English)
        intro_patterns_fr_en = [
            'je m’appelle', 'mon nom', 'je suis', 'comment tu t’appelles',
            'my name is', 'i am', 'what is your name', 'who are you'
        ]
        if any(pattern in question_lower for pattern in intro_patterns_fr_en):
            return True
        
        # Small talk
        small_talk = [
            'comment ça va', 'ça va', 'how are you', 'khỏe không',
            'merci', 'thank you', 'cảm ơn'
        ]
        if any(talk in question_lower for talk in small_talk):
            return True
        
        return False
    
    def _rewrite_query(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Use LLM to rewrite the question into a standalone query based on history.
        Replaces manual enrichment heuristics.
        """
        if not history:
            return question
            
        try:
            # Prepare minimal history for context (last 2 turns + current question)
            # Format: User: ... \n AI: ...
            context_str = ""
            for msg in history[-4:]: # Take last 4 messages for context
                role = "User" if msg.get('role') == 'user' else "Assistant"
                context_str += f"{role}: {msg.get('content', '')}\n"
                
            system_prompt = (
                "You are an expert query refiner. Your task is to rewrite the latest user question "
                "based on the chat history to make it a standalone query for a search engine.\n"
                "Rules:\n"
                "1. Replace pronouns (it, that, he, she, this subject) with specific names, entities, or email subjects from the history.\n"
                "2. Be specific and detailed. Include relevant context like IDs, names of projects/people if mentioned previously.\n"
                "3. Keep the SAME LANGUAGE as the latest user question (if user asks in French, rewrite in French).\n"
                "4. Do NOT answer the question. Just rewrite it."
            )
            
            user_prompt = f"Chat History:\n{context_str}\nLatest Question: {question}\n\nStandalone Question:"
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # Use GPT-3.5 for fast rewriting
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # Deterministic
                max_tokens=150
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            if rewritten.lower() != question.lower():
                logger.info(f"  ✨ Query Rewritten (LLM): '{question}' → '{rewritten}'")
                return rewritten
                
        except Exception as e:
            logger.error(f"Error rewriting query with LLM: {e}")
            
        return question

    def generate_title(self, question: str) -> str:
        """
        Generate a concise 3-5 word title for the conversation based on the first question.
        """
        try:
            system_prompt = (
                "You are a helpful assistant. Generate a short, concise title (3-5 words) "
                "summarizing the user's question. "
                "If the question is about summarizing an email, use the subject/topic. "
                "Do NOT use quotes. Keep it in the same language as the question."
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.5,
                max_tokens=20
            )
            
            title = response.choices[0].message.content.strip()
            # Remove quotes if present
            title = title.strip('"').strip("'")
            return title
            
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "New Chat"

    def query(
        self, 
        question: str, 
        method: str = "mmr_rerank",
        return_context: bool = False,
        session: Optional['ConversationSession'] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query: Retrieve → Compose → Generate
        
        Now with ConversationSummaryBufferMemory support:
        - Uses session memory for automatic conversation summarization
        - Extracts entities from LLM-generated summaries
        - More efficient token usage
         
        Args:
            question: User's question
            method: Retrieval method - "similarity" or "mmr"
            return_context: If True, include retrieved documents in response
            session: ConversationSession with chat history memory
            
        Returns:
            Dictionary with:
            - answer: LLM's response
            - sources: List of source documents
            - context: Retrieved documents (if return_context=True)
        """
        print(f"\n{'='*60}")
        print("RAG QUERY PIPELINE")
        print(f"{'='*60}")
        print(f"Question: {question}")
        
        # Extract context from session memory
        if session:
            print(f"🧠 Using session memory")
            memory_context = session.get_context()
            
            # Convert memory messages to history format for enrichment
            history_for_enrichment = []
            if 'chat_history' in memory_context:
                for msg in memory_context['chat_history']:
                    role = 'user' if hasattr(msg, 'type') and msg.type == 'human' else 'assistant'
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    history_for_enrichment.append({'role': role, 'content': content})
        else:
            print(f"ℹ️  No session - standalone question")
            history_for_enrichment = []
        
        # SMART ENRICHMENT: Add context if follow-up question
        enriched_query = self._rewrite_query(question, history_for_enrichment)
        
        # Retrieval with enriched query
        print(f"\nRetrieving documents (method: {method})...")
        documents = self.retrieve(enriched_query, method=method)
        print(f"Retrieved {len(documents)} documents")
        
        # Step 2: COMPOSE prompt with context
        print("\nStep 2: Composing prompt...")
        messages = create_messages_format(question, documents, history_for_enrichment)
        
        # Step 3: GENERATE answer using LLM
        print("\nStep 3: Generating answer...")
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,  # type: ignore
                temperature=self.llm_config.get('temperature', 0.1),
                max_tokens=self.llm_config.get('max_tokens', 1000)
            )
            
            answer = response.choices[0].message.content
            print("Answer generated successfully!")
            
            # Update session memory if using sessions
            if session:
                print("💾 Updating session memory...")
                session.add_exchange(question, answer)
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Error: Could not generate answer. {str(e)}"
        
        # Deduplicate sources and add Supabase links
        unique_sources = []
        seen_sources = set()
        supabase_url_base = os.getenv('SUPABASE_URL')
        
        from urllib.parse import quote
        
        for doc in documents:
            source_name = doc.metadata.get('source', 'Unknown')
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                
                # Get Supabase Storage URL (URL encode the filename for spaces/special chars)
                storage_path = doc.metadata.get('storage_path')
                if storage_path:
                    # Use existing storage_path from metadata
                    encoded_path = quote(storage_path, safe='/')
                    file_url = f"{supabase_url_base}/storage/v1/object/public/source-documents/{encoded_path}"
                else:
                    # Construct URL from source filename
                    encoded_name = quote(source_name, safe='')
                    file_url = f"{supabase_url_base}/storage/v1/object/public/source-documents/{encoded_name}"
                
                # Create markdown link format: [name](url)
                unique_sources.append(f"[{source_name}]({file_url})")
        
        # Prepare response
        result = {
            "answer": answer,
            "sources": unique_sources
        }
        
        if return_context:
            result["context"] = documents
        
        print(f"{'='*60}\n")
        return result


def main():
    """
    CLI interface for testing the RAG chain.
    
    This allows you to:
    - Ask questions interactively
    - Test different retrieval methods
    - See retrieved sources
    - Verify the full pipeline works
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Question Answering System")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["similarity", "mmr", "rerank", "mmr_rerank"],
        default="mmr_rerank",
        help="Retrieval method (default: mmr_rerank - best quality)"
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context documents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file. Defaults to environment via RAG_ENV or RAG_CONFIG_PATH."
    )
    args = parser.parse_args()
    
    # Initialize RAG chain
    print("\n" + "="*60)
    print("RAG QUESTION ANSWERING SYSTEM")
    print("="*60)
    rag = RAGChain(config_path=args.config)
    
    # Query the system
    result = rag.query(
        question=args.question,
        method=args.method,
        return_context=args.show_context
    )
    
    # Display results
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(result['answer'])
    
    print("\n" + "="*60)
    print("SOURCES")
    print("="*60)
    for i, source in enumerate(result['sources'], 1):
        print(f"[{i}] {source}")
    
    # Show context if requested
    if args.show_context and 'context' in result:
        print("\n" + "="*60)
        print("RETRIEVED CONTEXT")
        print("="*60)
        for i, doc in enumerate(result['context'], 1):
            print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
            content = doc.page_content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"Content: {content}")
            print("-" * 60)
    
    print("\n" + "="*60)
    print("QUERY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
