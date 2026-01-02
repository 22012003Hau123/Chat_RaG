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
        
        NEW APPROACH:
        - Extract from BOTH user questions AND bot responses
        - Prioritize proper nouns (capitalized words like "Lucid", "iPhone")
        - Use simple NER to identify entities
        
        Args:
            history: Recent conversation history
            max_words: Maximum words to extract
            
        Returns:
            Topic keywords string (entity names)
        """
        if not history or len(history) == 0:
            return ""
        
        # Strategy 1: Extract proper nouns (entities) from recent bot responses
        # Bot responses usually contain the actual entity names clearly
        entities = []
        
        # Look at last 3 turns (6 messages)
        for msg in reversed(history[-6:]):
            content = msg.get('content', '')
            words = content.split()
            
            # Extract capitalized words (proper nouns) - simple NER
            for word in words:
                # Clean word of punctuation
                cleaned = word.strip('.,!?:;()[]{}"\'')
                
                # Check if it's a proper noun:
                # - Starts with capital letter
                # - Length > 2 (avoid "Le", "La", etc.)
                # - Not common French/English/Vietnamese articles/phrases
                # - Not common French connectors/sentence starters
                excluded_words = [
                    # Articles
                    'The', 'Les', 'Une', 'Des', 'Một', 'Các', 
                    # Prepositions
                    'Pour', 'Dans', 'Avec', 'Sans', 'Sous', 'Sur',
                    # Common French connectors/phrases (CRITICAL!)
                    "D'après", "Selon", "Voici", "Voilà", "Cependant", 
                    "Toutefois", "Néanmoins", "Ainsi", "Donc", "Ensuite",
                    # Sentence starters
                    "Bonjour", "Merci", "Désolé", "Pardon",
                    # Common Vietnamese starters
                    "Xin", "Cảm", "Chào"
                ]
                
                if (cleaned and 
                    cleaned[0].isupper() and 
                    len(cleaned) > 2 and
                    cleaned not in excluded_words and
                    not cleaned.startswith("D'") and  # Filter D'après, D'autre, etc.
                    not cleaned.startswith("L'") and  # Filter L'application, etc.
                    not cleaned.endswith("...")):     # Filter incomplete text
                    entities.append(cleaned)
            
            # Stop if we have enough entities
            if len(entities) >= 3:
                break
        
        # Remove duplicates while preserving order (most recent first)
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)
        
        # SMART PRIORITIZATION: Match entities with user questions
        if unique_entities:
            # Get recent user questions to match against
            user_questions = []
            for msg in reversed(history[-6:]):
                if msg.get('role') == 'user':
                    user_questions.append(msg.get('content', '').lower())
                    if len(user_questions) >= 2:
                        break
            
            # Find entities that appear in user questions (highest priority)
            matched_entities = []
            for entity in unique_entities:
                entity_lower = entity.lower()
                for question in user_questions:
                    if entity_lower in question:
                        matched_entities.append(entity)
                        break
            
            # Use matched entity if found, otherwise use first extracted entity
            if matched_entities:
                main_entity = matched_entities[0]
                print(f"  🎯 Extracted topic entity from history (matched with user question): '{main_entity}'")
            else:
                main_entity = unique_entities[0]
                print(f"  🎯 Extracted topic entity from history: '{main_entity}'")
            
            return main_entity
        
        # Fallback: Extract from user questions (old logic)
        user_questions = []
        for msg in reversed(history[-6:]):
            if msg.get('role') == 'user':
                user_questions.append(msg.get('content', ''))
                if len(user_questions) >= 2:
                    break
        
        if not user_questions:
            return ""
        
        # Take first user question and extract keywords
        topic_question = user_questions[-1]
        words = topic_question.split()
        
        # Filter stop words
        stop_words = ['là', 'gì', 'the', 'what', 'is', 'are', 'c\'est', 'qu\'est-ce', 'quoi', 
                      'như', 'thế', 'nào', 'how', 'why', 'when', 'where', 'về', 'của', 'cho',
                      'et', 'de', 'à', 'un', 'une', 'le', 'la']
        
        keywords = [w for w in words if w.lower() not in stop_words]
        
        if keywords:
            result = ' '.join(keywords[:max_words])
            print(f"  🎯 Extracted topic keywords (fallback): '{result}'")
            return result
        
        return ""
    
    def _is_followup_question(self, question: str) -> bool:
        """
        Detect if question is a follow-up (needs context enrichment).
        
        ENHANCED DETECTION:
        - More follow-up patterns
        - Check for pronouns ("it", "nó", "that")
        - Detect vague questions without clear subject
        
        Args:
            question: User question
            
        Returns:
            True if follow-up, False if standalone
        """
        question_lower = question.lower()
        words = question.split()
        
        # Follow-up indicators - EXPANDED
        followup_patterns = [
            # Vietnamese - image/content requests
            'ảnh', 'hình', 'cho', 'thêm', 'nữa', 'hiển thị',
            'xem', 'giải thích', 'chi tiết', 'cụ thể', 'minh họa',
            # Vietnamese pronouns/demonstratives
            'nó', 'cái đó', 'cái này', 'đó', 'này',
            # French
            'image', 'photo', 'montre', 'affiche', 'plus', 'autre',
            'détail', 'expliquer', 'illustration',
            # French pronouns
            'ça', 'cela', 'celui', 'celle',
            # English  
            'image', 'show', 'display', 'more', 'another', 'details',
            'explain', 'illustration',
            # English pronouns
            'it', 'this', 'that', 'them', 'those'
        ]
        
        # Check if contains any follow-up pattern
        has_pattern = any(pattern in question_lower for pattern in followup_patterns)
        
        # Check if question is very short (< 3 words = likely incomplete)
        is_very_short = len(words) < 3
        
        # Check if question is short and vague (< 5 words, no proper nouns)
        is_short_vague = False
        if len(words) <= 4:
            # No capitalized words = no clear subject
            has_entity = any(word[0].isupper() for word in words if len(word) > 0)
            is_short_vague = not has_entity
        
        # Detect questions starting with action verbs (command-like)
        action_starters = ['cho', 'show', 'montre', 'affiche', 'display', 'give', 'tell']
        starts_with_action = any(question_lower.startswith(verb) for verb in action_starters)
        
        return has_pattern or is_very_short or is_short_vague or starts_with_action
    
    def _enrich_query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Smart query enrichment:
        - Extract topic entity from recent history
        - ALWAYS enrich if follow-up question detected
        - Add entity context to help retrieval
        
        IMPROVED STRATEGY:
        - If follow-up detected → extract main entity from history
        - Format as "{question} về {entity}" or "{question} {entity}"
        - This helps retrieval find the right documents even with vague questions
        
        Args:
            question: Original user question
            history: Conversation history
            
        Returns:
            Enriched query if follow-up, original otherwise
        """
        # No history = no enrichment
        if not history or len(history) == 0:
            return question
        
        # Check if this is a follow-up question
        if not self._is_followup_question(question):
            # Standalone question - no enrichment needed
            print(f"  ✓ Standalone question, no enrichment needed")
            return question
        
        # Extract topic entity from history
        topic_entity = self._extract_topic_keywords(history, max_words=5)
        
        if topic_entity:
            # Smart formatting based on language
            question_lower = question.lower()
            
            # If question already contains "về" or similar, just append
            if any(prep in question_lower for prep in ['về', 'de', 'about', 'of']):
                enriched = f"{question} {topic_entity}"
            # Otherwise, add "về" for Vietnamese, nothing for others
            elif any(vn_word in question_lower for vn_word in ['ảnh', 'hình', 'cho', 'xem']):
                enriched = f"{question} về {topic_entity}"
            else:
                # Default: just append
                enriched = f"{question} {topic_entity}"
            
            print(f"  🔍 Query enriched: '{question}' → '{enriched}'")
            return enriched
        else:
            print(f"  ⚠️  Follow-up detected but no topic entity found in history")
        
        return question

    def query(
        self, 
        question: str, 
        method: str = "similarity",
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
        enriched_query = self._enrich_query(question, history_for_enrichment)
        
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
