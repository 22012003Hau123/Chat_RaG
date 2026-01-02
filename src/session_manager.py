"""
Session Manager for RAG Chatbot

Manages conversation sessions with LangChain's ConversationSummaryBufferMemory.
Each session maintains:
- Conversation history with automatic summarization
- Last access time for expiry
- Session-specific context for entity extraction

Sessions expire after 30 minutes of inactivity.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class ConversationSession:
    """
    Represents a single conversation session.
    
    Stores conversation history with messages.
    Automatically keeps only recent messages to prevent context overflow.
    """
    
    def __init__(self, session_id: str, max_messages: int = 20):
        """
        Initialize a new conversation session.
        
        Args:
            session_id: Unique identifier for this session
            max_messages: Maximum number of messages to keep (default: 20 = 10 turns)
        """
        self.session_id = session_id
        self.last_accessed = datetime.now()
        self.max_messages = max_messages
        
        # Use ChatMessageHistory for message storage
        self.history = ChatMessageHistory()
    
    def update_access_time(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """
        Check if session has expired due to inactivity.
        
        Args:
            timeout_minutes: Session timeout in minutes (default: 30)
            
        Returns:
            True if session has been inactive for longer than timeout
        """
        expiry_time = self.last_accessed + timedelta(minutes=timeout_minutes)
        return datetime.now() > expiry_time
    
    def get_messages(self) -> List:
        """
        Get conversation messages.
        
        Returns:
            List of HumanMessage and AIMessage objects
        """
        return self.history.messages
    
    def get_context(self) -> Dict:
        """
        Get conversation context in format compatible with RAG chain.
        
        Returns:
            Dictionary containing 'chat_history' with list of messages
        """
        return {'chat_history': self.history.messages}
    
    def add_exchange(self, question: str, answer: str):
        """
        Add a Q&A exchange to the conversation history.
        
        Args:
            question: User's question
            answer: Assistant's answer
        """
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        
        # Trim history if it exceeds max_messages
        if len(self.history.messages) > self.max_messages:
            # Keep only the most recent max_messages
            self.history.messages = self.history.messages[-self.max_messages:]
        
        self.update_access_time()


class SessionManager:
    """
    Manages all active conversation sessions.
    
    Responsibilities:
    - Create new sessions with unique IDs
    - Retrieve existing sessions
    - Periodically clean up expired sessions
    - Thread-safe access to sessions
    """
    
    def __init__(self, cleanup_interval_minutes: int = 10):
        """
        Initialize the session manager.
        
        Args:
            cleanup_interval_minutes: How often to clean up expired sessions
        """
        self.sessions: Dict[str, ConversationSession] = {}
        self.lock = threading.Lock()
        
        # Start background cleanup thread
        self.cleanup_interval = cleanup_interval_minutes * 60  # Convert to seconds
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        
        print(f"📋 SessionManager initialized (cleanup every {cleanup_interval_minutes} min)")
    
    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ConversationSession instance
        """
        with self.lock:
            if session_id not in self.sessions:
                # Create new session
                self.sessions[session_id] = ConversationSession(session_id)
                print(f"📝 Session {session_id[:8]}...: NEW session created")
            else:
                # Update existing session access time
                self.sessions[session_id].update_access_time()
            
            return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get existing session without creating a new one.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ConversationSession if exists, None otherwise
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.update_access_time()
            return session
    
    def delete_session(self, session_id: str):
        """
        Manually delete a session.
        
        Args:
            session_id: Session to delete
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"🗑️  Session {session_id[:8]}... deleted")
    
    def cleanup_expired_sessions(self, timeout_minutes: int = 30) -> int:
        """
        Remove expired sessions from memory.
        
        Args:
            timeout_minutes: Session timeout in minutes
            
        Returns:
            Number of sessions cleaned up
        """
        with self.lock:
            expired_ids = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(timeout_minutes)
            ]
            
            for sid in expired_ids:
                del self.sessions[sid]
            
            if expired_ids:
                print(f"🗑️  Cleaned up {len(expired_ids)} expired sessions")
            
            return len(expired_ids)
    
    def get_active_session_count(self) -> int:
        """Get number of currently active sessions."""
        with self.lock:
            return len(self.sessions)
    
    def _cleanup_loop(self):
        """
        Background thread that periodically cleans up expired sessions.
        """
        while True:
            time.sleep(self.cleanup_interval)
            self.cleanup_expired_sessions()
