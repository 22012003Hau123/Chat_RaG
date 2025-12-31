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
from typing import Dict, Any, List, Optional

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
    - Document retrieval (from FAISS)
    - Prompt engineering (context injection)
    - LLM generation (OpenAI)
    
    This is the "brain" of the RAG system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG chain with all components.
        
        What happens during initialization:
        1. Load configuration
        2. Load FAISS vector store
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
    
    def retrieve(self, question: str, method: str = "similarity") -> List[Document]:
        """
        Retrieve relevant documents for a question.
        
        This is Step 1 of the RAG pipeline: RETRIEVE
        
        Args:
            question: User's question
            method: Search method - "similarity" or "mmr"
            
        Returns:
            List of relevant Document objects
        """
        k = self.retrieval_config.get('top_k', 5)
        
        if method == "similarity":
            results = self.retriever.similarity_search(
                query=question,
                k=k
            )
        elif method == "mmr":
            lambda_mult = self.retrieval_config.get('mmr_lambda', 0.5)
            fetch_k = self.retrieval_config.get('fetch_k', 20)
            results = self.retriever.mmr_search(
                query=question,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
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
    
    def _extract_topic_keywords(self, history: List[Dict[str, str]], max_words: int = 5) -> str:
        """
        Extract main topic keywords from recent history.
        Returns a short phrase representing the topic, not full questions.
        
        Args:
            history: Recent conversation history
            max_words: Maximum words to extract
            
        Returns:
            Topic keywords string
        """
        if not history or len(history) == 0:
            return ""
        
        # Look at last 2-3 user questions to find topic
        user_questions = []
        for msg in reversed(history[-6:]):  # Last 3 turns
            if msg.get('role') == 'user':
                user_questions.append(msg.get('content', ''))
                if len(user_questions) >= 2:
                    break
        
        if not user_questions:
            return ""
        
        # Take first user question as main topic
        # Extract key nouns/entities (simple approach: take first max_words)
        topic_question = user_questions[-1]  # Oldest in our list = earliest question
        words = topic_question.split()
        
        # Filter out common question words
        stop_words = ['là', 'gì', 'the', 'what', 'is', 'are', 'c\'est', 'qu\'est-ce', 'quoi', 
                      'như', 'thế', 'nào', 'how', 'why', 'when', 'where']
        
        keywords = [w for w in words if w.lower() not in stop_words]
        
        # Return first max_words keywords
        return ' '.join(keywords[:max_words])
    
    def _is_followup_question(self, question: str) -> bool:
        """
        Detect if question is a follow-up (needs context enrichment).
        
        Args:
            question: User question
            
        Returns:
            True if follow-up, False if standalone
        """
        question_lower = question.lower()
        
        # Follow-up indicators
        followup_patterns = [
            # Vietnamese - image/content requests
            'ảnh', 'hình', 'cho', 'thêm', 'nữa', 'hiển thị',
            'xem', 'giải thích', 'chi tiết', 'cụ thể',
            # French
            'image', 'photo', 'montre', 'affiche', 'plus', 'autre',
            'détail', 'expliquer',
            # English  
            'image', 'show', 'display', 'more', 'another', 'details',
            'explain'
        ]
        
        # Check if contains any follow-up pattern
        has_pattern = any(pattern in question_lower for pattern in followup_patterns)
        
        # Also check if question is short (likely incomplete)
        is_short = len(question.split()) <= 4
        
        return has_pattern or is_short
    
    def _enrich_query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Smart query enrichment:
        - Extract topic keywords from recent history (not full questions)
        - Only enrich if question is a follow-up
        - Avoid overly long queries
        
        Args:
            question: Original user question
            history: Conversation history
            
        Returns:
            Enriched query if needed, original otherwise
        """
        # No history = no enrichment
        if not history or len(history) == 0:
            return question
        
        # Check if this is a follow-up question
        if not self._is_followup_question(question):
            # Standalone question - no enrichment needed
            return question
        
        # Extract topic keywords (short!)
        topic_keywords = self._extract_topic_keywords(history, max_words=5)
        
        if topic_keywords:
            # Enrich with topic keywords
            enriched = f"{question} {topic_keywords}"
            print(f"  🔍 Query enriched: '{question}' → '{enriched}'")
            return enriched
        
        return question

    def query(
        self, 
        question: str, 
        method: str = "similarity",
        return_context: bool = False,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query: Retrieve → Compose → Generate
        
        This is the full RAG pipeline in action:
        
        Step 1 - RETRIEVE: Get relevant documents from vector store
        Step 2 - COMPOSE: Format documents into prompt with question
        Step 3 - GENERATE: Send to LLM and get answer
        
        Why RAG works:
        - LLM gets specific, relevant context
        - Reduces hallucination (LLM can't make stuff up)
        - Answer is grounded in your documents
        - Can cite sources in the response
        
        Args:
            question: User's question
            method: Retrieval method - "similarity" or "mmr"
            return_context: If True, include retrieved documents in response
            
        Returns:
            Dictionary with:
            - answer: LLM's response
            - sources: List of source documents (optional)
            - context: Retrieved documents (if return_context=True)
        """
        print(f"\n{'='*60}")
        print("RAG QUERY PIPELINE")
        print(f"{'='*60}")
        print(f"Question: {question}")
        
        # SMART ENRICHMENT: Add context if follow-up question
        enriched_query = self._enrich_query(question, history)
        
        # Retrieval with enriched query
        print(f"\nRetrieving documents (method: {method})...")
        documents = self.retrieve(enriched_query, method=method)
        print(f"Retrieved {len(documents)} documents")
        
        # Step 2: COMPOSE prompt with context
        print("\nStep 2: Composing prompt...")
        messages = create_messages_format(question, documents, history)
        
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
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Error: Could not generate answer. {str(e)}"
        
        # Deduplicate sources and add Supabase links
        unique_sources = []
        seen_sources = set()
        supabase_url_base = os.getenv('SUPABASE_URL')
        
        for doc in documents:
            source_name = doc.metadata.get('source', 'Unknown')
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                
                # Get Supabase Storage URL
                storage_path = doc.metadata.get('storage_path')
                if storage_path:
                    # Use existing storage_path from metadata
                    file_url = f"{supabase_url_base}/storage/v1/object/public/source-documents/{storage_path}"
                else:
                    # Construct URL from source filename
                    # Assume files are in source-documents bucket
                    file_url = f"{supabase_url_base}/storage/v1/object/public/source-documents/{source_name}"
                
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
        choices=["similarity", "mmr"],
        default="similarity",
        help="Retrieval method (default: similarity)"
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
