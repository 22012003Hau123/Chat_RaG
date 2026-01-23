"""
Chat History Module

Manages persistent chat conversations in Supabase.
Provides CRUD operations for conversations and messages.
"""

import os
import logging
from typing import Optional, List, Dict
from datetime import datetime
from supabase import create_client

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat history persistence in Supabase."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize with Supabase connection."""
        self.supabase = create_client(supabase_url, supabase_key)
        logger.info("📜 ChatHistoryManager initialized")
    
    def create_conversation(self, title: str = "New Chat", mode: str = "document") -> Optional[str]:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            mode: Chat mode ('document' or 'email')
        
        Returns:
            Conversation ID or None if failed
        """
        try:
            result = self.supabase.table("chat_conversations").insert({
                "title": title,
                "mode": mode
            }).execute()
            
            if result.data:
                conv_id = result.data[0]["id"]
                logger.info(f"✓ Created conversation: {conv_id[:8]}... (mode={mode})")
                return conv_id
            return None
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return None
    
    def list_conversations(self, limit: int = 50, mode: Optional[str] = None) -> List[Dict]:
        """
        Get list of conversations, ordered by most recent.
        
        Args:
            limit: Maximum number of conversations to return
            mode: Optional filter by chat mode ('document' or 'email')
        
        Returns:
            List of conversation dicts with id, title, mode, created_at, updated_at
        """
        try:
            query = self.supabase.table("chat_conversations")\
                .select("id, title, mode, created_at, updated_at")\
                .order("updated_at", desc=True)\
                .limit(limit)
            
            # Filter by mode if specified
            if mode:
                query = query.eq("mode", mode)
            
            result = query.execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Get a conversation with all its messages.
        
        Returns:
            Dict with conversation info and messages list
        """
        try:
            # Get conversation - use maybe_single to avoid error when not found
            conv_result = self.supabase.table("chat_conversations")\
                .select("*")\
                .eq("id", conversation_id)\
                .maybe_single()\
                .execute()
            
            if not conv_result.data:
                return None
            
            # Get messages
            msg_result = self.supabase.table("chat_messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=False)\
                .execute()
            
            return {
                "conversation": conv_result.data,
                "messages": msg_result.data or []
            }
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None
    
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            True if successful
        """
        try:
            # Insert message
            self.supabase.table("chat_messages").insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content
            }).execute()
            
            # Update conversation's updated_at
            self.supabase.table("chat_conversations")\
                .update({"updated_at": datetime.utcnow().isoformat()})\
                .eq("id", conversation_id)\
                .execute()
            
            return True
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False
    
    def update_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        try:
            self.supabase.table("chat_conversations")\
                .update({"title": title})\
                .eq("id", conversation_id)\
                .execute()
            return True
        except Exception as e:
            logger.error(f"Error updating title: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Returns:
            True if successful
        """
        try:
            # Messages deleted automatically via CASCADE
            self.supabase.table("chat_conversations")\
                .delete()\
                .eq("id", conversation_id)\
                .execute()
            
            logger.info(f"🗑️ Deleted conversation: {conversation_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False
    
    def generate_title_from_message(self, message: str, max_length: int = 50) -> str:
        """Generate a title from the first user message."""
        # Take first sentence or first N characters
        title = message.strip()
        
        # Remove newlines
        title = title.replace('\n', ' ')
        
        # Truncate
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        return title if title else "New Chat"
    
    def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Delete conversations older than specified days.
        
        Args:
            days: Number of days (default: 30)
            
        Returns:
            Number of deleted conversations
        """
        try:
            from datetime import timedelta
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Get old conversations
            result = self.supabase.table("chat_conversations")\
                .select("id")\
                .lt("updated_at", cutoff_date)\
                .execute()
            
            old_ids = [row["id"] for row in result.data] if result.data else []
            
            if old_ids:
                # Delete old conversations (messages auto-deleted via CASCADE)
                for conv_id in old_ids:
                    self.supabase.table("chat_conversations")\
                        .delete()\
                        .eq("id", conv_id)\
                        .execute()
                
                logger.info(f"🧹 Cleaned up {len(old_ids)} conversations older than {days} days")
            
            return len(old_ids)
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")
            return 0


# Singleton instance
_chat_history_instance = None

def get_chat_history() -> Optional[ChatHistoryManager]:
    """Get or create ChatHistoryManager instance."""
    global _chat_history_instance
    
    if _chat_history_instance is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if url and key:
            _chat_history_instance = ChatHistoryManager(url, key)
        else:
            logger.warning("Supabase credentials not found")
    
    return _chat_history_instance
