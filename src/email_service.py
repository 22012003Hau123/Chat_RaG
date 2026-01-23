"""
Email Service Module

Fetches emails from Supabase 'email' table.
Provides methods to list, get, and prepare emails for ingestion.
"""

import os
import logging
from typing import Optional, List, Dict
from datetime import datetime
from supabase import create_client

logger = logging.getLogger(__name__)


class EmailService:
    """Service to interact with emails stored in Supabase."""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize with Supabase connection."""
        url = supabase_url or os.environ.get("SUPABASE_URL")
        key = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required")
        
        self.supabase = create_client(url, key)
        logger.info("📧 EmailService initialized")
    
    def _parse_email_headers(self, content: str) -> Dict:
        """Parse email headers from content to extract From, To, Sent, Subject."""
        import re
        
        result = {
            "from_name": "",
            "to_email": "",
            "sent_date": "",
            "parsed_subject": ""
        }
        
        if not content:
            return result
        
        # Extract From: line
        from_match = re.search(r'From:\s*(.+?)(?:\n|$)', content)
        if from_match:
            result["from_name"] = from_match.group(1).strip()
        
        # Extract To: line  
        to_match = re.search(r'To:\s*(.+?)(?:\n|$)', content)
        if to_match:
            result["to_email"] = to_match.group(1).strip()
        
        # Extract Sent: line
        sent_match = re.search(r'Sent:\s*(.+?)(?:\n|$)', content)
        if sent_match:
            result["sent_date"] = sent_match.group(1).strip()
        
        # Extract Subject: line from content (may differ from subject column)
        subject_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', content)
        if subject_match:
            result["parsed_subject"] = subject_match.group(1).strip()
        
        return result
    
    def list_emails(self, limit: int = 50) -> List[Dict]:
        """
        Get list of all emails, ordered by most recent.
        
        Returns:
            List of email dicts with parsed headers
        """
        try:
            result = self.supabase.table("email")\
                .select("id, subject, sender_email, sent_at, content, created_at")\
                .order("sent_at", desc=True)\
                .limit(limit)\
                .execute()
            
            # Parse headers from content for each email
            emails = []
            for email in (result.data or []):
                parsed = self._parse_email_headers(email.get("content", ""))
                email["from_name"] = parsed["from_name"]
                email["to_email"] = parsed["to_email"]
                email["sent_date"] = parsed["sent_date"]
                emails.append(email)
            
            return emails
        except Exception as e:
            logger.error(f"Error listing emails: {e}")
            return []
    
    def get_email(self, email_id: str) -> Optional[Dict]:
        """
        Get a specific email with full content.
        
        Returns:
            Email dict with all fields
        """
        try:
            result = self.supabase.table("email")\
                .select("*")\
                .eq("id", email_id)\
                .maybe_single()\
                .execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Error getting email {email_id}: {e}")
            return None
    
    def get_all_for_ingest(self) -> List[Dict]:
        """
        Get all emails with content for embedding ingestion.
        
        Returns:
            List of emails with id, subject, sender_email, sent_at, content
        """
        try:
            result = self.supabase.table("email")\
                .select("id, subject, sender_email, sent_at, content")\
                .order("sent_at", desc=True)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting emails for ingest: {e}")
            return []
    
    def get_emails_not_yet_embedded(self) -> List[Dict]:
        """
        Get emails that haven't been embedded yet.
        
        Checks email_embeddings table to find which emails need processing.
        """
        try:
            # Get all email IDs
            all_emails = self.supabase.table("email")\
                .select("id, subject, sender_email, sent_at, content")\
                .execute()
            
            if not all_emails.data:
                return []
            
            # Get already embedded email IDs
            embedded = self.supabase.table("email_embeddings")\
                .select("email_id")\
                .execute()
            
            embedded_ids = set(row["email_id"] for row in (embedded.data or []))
            
            # Filter to only non-embedded emails
            not_embedded = [
                email for email in all_emails.data 
                if email["id"] not in embedded_ids
            ]
            
            logger.info(f"📧 Found {len(not_embedded)} emails not yet embedded")
            return not_embedded
            
        except Exception as e:
            logger.error(f"Error checking embedded emails: {e}")
            return []

    def delete_email(self, email_id: str) -> bool:
        """
        Delete an email by ID.
        
        Note: email_embeddings table is set to ON DELETE CASCADE,
        so embeddings will be automatically removed by Supabase.
        """
        try:
            self.supabase.table("email").delete().eq("id", email_id).execute()
            logger.info(f"🗑️ Deleted email {email_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting email {email_id}: {e}")
            return False


# Singleton instance
_email_service_instance = None


def get_email_service() -> Optional[EmailService]:
    """Get or create EmailService instance."""
    global _email_service_instance
    
    if _email_service_instance is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        try:
            _email_service_instance = EmailService()
        except Exception as e:
            logger.warning(f"Failed to initialize EmailService: {e}")
    
    return _email_service_instance
