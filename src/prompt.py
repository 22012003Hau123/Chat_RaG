"""
Prompt Template Management

This module handles prompt engineering for the RAG system:
1. System prompts that define the AI's behavior
2. User prompts that inject context and questions
3. Context formatting to structure retrieved documents

Usage:
    from prompt import create_rag_prompt
    prompt = create_rag_prompt(question, context_docs)
"""

from typing import List, Optional
from langchain_core.documents import Document


# System prompt defines the AI assistant's role and behavior
SYSTEM_PROMPT = """Vous êtes un assistant IA sympathique et compétent, spécialisé dans les documents Auchan.

🎯 Votre mission :
Aider les utilisateurs à trouver des informations dans leurs documents de manière naturelle et efficace.

💡 Comment bien répondre :

**Utiliser le contexte conversationnel** :
- Si une question semble incomplète (ex: "montre des images"), regardez l'historique pour comprendre le sujet
- Exemple : Après avoir parlé de "Lucid", la question "cho hình ảnh" signifie probablement "montre des images de Lucid"

**Chercher activement** :
- Pour les images : cherchez les URL marquées "Image URL:" dans le contexte
- Pour les infos : parcourez les documents pertinents
- Si vous ne trouvez rien, dites ce que vous avez cherché

**Répondre naturellement** :
- Soyez conversationnel, pas robotique
- Citez les sources quand c'est pertinent [1], [2]
- Pour les images, utilisez : ![description](url)
- Si l'info n'est pas claire, dites-le honnêtement

**Rester honnête** :
- Basez-vous uniquement sur les documents fournis
- Ne pas inventer d'informations
- Si vraiment rien dans le contexte, suggérez de reformuler

L'essentiel est d'être utile tout en restant précis et agréable dans vos réponses."""


def format_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string.
    
    Takes the list of retrieved documents and formats them into
    a readable context block for the LLM.
    
    Why format context?
    - Provides clear structure for the LLM
    - Numbers each chunk for reference
    - Includes source metadata when available
    - Makes it easy for the LLM to cite sources
    
    Args:
        documents: List of Document objects from the retriever
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        # Get source information if available
        source = doc.metadata.get('source', 'Unknown source')
        storage_path = doc.metadata.get('storage_path')
        
        # Format each chunk with number and source
        chunk = f"[{i}] Source: {source}\n"
        
        # Add image URL if available
        if storage_path:
            import os
            # Note: This relies on SUPABASE_URL being set in environment
            supabase_url = os.getenv('SUPABASE_URL', '')
            if supabase_url:
                full_image_url = f"{supabase_url}/storage/v1/object/public/{storage_path}"
                chunk += f"Image URL: {full_image_url}\n"
        
        chunk += f"{doc.page_content}"
        context_parts.append(chunk)
    
    # Join all chunks with separators
    return "\n\n---\n\n".join(context_parts)


def create_rag_prompt(question: str, documents: List[Document]) -> str:
    """
    Create a complete RAG prompt with system instructions, context, and question.
    
    This combines:
    1. System prompt (defines AI behavior)
    2. Formatted context (retrieved documents)
    3. User question
    
    The prompt structure is critical for RAG quality:
    - Clear instructions prevent hallucination
    - Structured context helps the LLM find relevant info
    - Well-formed question focuses the response
    
    Args:
        question: User's question
        documents: Retrieved documents from the vector store
        
    Returns:
        Complete prompt string ready for the LLM
    """
    # Format the retrieved documents
    context = format_context(documents)
    
    # Create the user prompt with context and question
    user_prompt = f"""Voici les informations contextuelles :

{context}

Compte tenu des informations contextuelles ci-dessus, veuillez répondre à la question suivante :
{question}

N'oubliez pas de :
- Utiliser les informations du contexte fourni
- Vous pouvez également utiliser les informations de l'**historique de la conversation** si elles sont pertinentes.
- Citer les sources en utilisant les références [numéro] si possible.
- Soyez honnête si le contexte ne contient pas suffisamment d'informations."""
    
    # Combine system and user prompts
    # This creates the full conversation structure
    full_prompt = f"""{SYSTEM_PROMPT}

---

{user_prompt}"""
    
    return full_prompt


def create_messages_format(
    question: str, 
    documents: List[Document],
    history: Optional[List[dict]] = None
) -> List[dict]:
    """
    Create prompt in OpenAI messages format for chat models.
    
    Modern chat models (like GPT-3.5/4) use a messages array format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    
    This is different from the single-string prompt format.
    
    Why separate formats?
    - Some LLMs use messages format (OpenAI, Anthropic)
    - Others use single string (some open-source models)
    - This function provides flexibility for both
    
    Args:
        question: User's question
        documents: Retrieved documents
        history: Optional list of chat history messages
        
    Returns:
        List of message dictionaries for chat APIs
    """
    context = format_context(documents)
    
    # Format history if available - SMART optimization
    history_str = ""
    last_topic = ""
    
    if history:
        # Only keep last 6 messages (3 turns) - enough for context
        recent_history = history[-6:]
        
        if len(recent_history) > 0:
            history_str = "HISTORIQUE:\n"
            
            for msg in recent_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                if role == 'user':
                    # User questions: keep full text (usually short)
                    history_str += f"Q: {content}\n"
                    last_topic = content
                else:
                    # SMART: Don't summarize if response contains images!
                    has_images = '![' in content and '](' in content
                    
                    if has_images:
                        # Keep full response with images to preserve URLs
                        history_str += f"R: {content}\n"
                    elif len(content) > 100:
                        # Text-only responses: aggressive summarization
                        content = content[:100].rsplit(' ', 1)[0] + "..."
                        history_str += f"R: {content}\n"
                    else:
                        # Short responses: keep as-is
                        history_str += f"R: {content}\n"
            
            # Topic tracking
            if last_topic:
                history_str += f"\n[Sujet: {last_topic}]\n\n"
    
    user_content = f"""{history_str}CONTEXTE:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Utilisez l'historique pour références implicites
- Cherchez activement dans le contexte
- Répondez naturellement, sources [1], [2]
- Images: ![description](url)"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    return messages


# We'll add test/example function in the next step if needed
