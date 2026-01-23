-- Add mode column to chat_conversations table
-- This allows filtering conversation history by chat mode (document/email)

ALTER TABLE chat_conversations 
ADD COLUMN IF NOT EXISTS mode TEXT DEFAULT 'document';

-- Add comment
COMMENT ON COLUMN chat_conversations.mode IS 'Chat mode: document or email';
