"""
Email-Specific Prompt Templates

Prompt engineering for email RAG system.
Focused on email context: sender, recipient, subject, date.
"""

from typing import List, Optional
from langchain_core.documents import Document


# System prompt for Email Chat mode
EMAIL_SYSTEM_PROMPT = """Vous êtes un assistant IA spécialisé dans la recherche et l'analyse d'emails.

🎯 Votre mission :
Aider les utilisateurs à trouver et comprendre les informations dans leurs emails de manière efficace.

💡 Comment bien répondre :

**Comprendre le contexte email** :
- Chaque email a un expéditeur, destinataire, sujet et date
- Utilisez ces métadonnées pour contextualiser vos réponses
- Si l'utilisateur cherche un email spécifique, utilisez les critères (date, expéditeur, sujet)

**Format de réponse** :
- Citez l'expéditeur et la date de l'email
- Résumez le contenu pertinent
- Format: "[De: expéditeur, Date: date] contenu..."

**Prioriser les sources** :
- Les emails sont listés PAR ORDRE DE PERTINENCE
- [1] est le plus pertinent
- Citez les sources: "[1] Email de Jean le 15/01..."

**Rester honnête** :
- Basez-vous uniquement sur les emails fournis
- Ne pas inventer d'informations
- Si aucun email ne correspond, dites-le clairement

**Langue de réponse** :
- Répondez TOUJOURS dans la même langue que la question de l'utilisateur (Français, Vietnamien, Anglais, etc.)
- Si la question est en Vietnamien, répondez en Vietnamien.

**Conversation naturelle** :
- Soyez conversationnel et utile
- Proposez des recherches alternatives si besoin
- Résumez les points clés des emails
- Si plusieurs emails traitent du même sujet, regroupez les informations de manière synthétique."""


def format_email_context(documents: List[Document]) -> str:
    """
    Format email documents into context string.
    
    Includes email metadata (sender, recipient, date, subject) for each result.
    """
    if not documents:
        return "Aucun email pertinent trouvé."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata or {}
        
        # Extract email metadata
        email_id_meta = metadata.get('email_id', 'N/A')
        sender = metadata.get('sender_email', 'Inconnu')
        subject = metadata.get('subject', 'Sans sujet')
        date = metadata.get('sent_at', '')
        
        # Format header
        header = f"[{i}] 📧 Email (ID: {email_id_meta})"
        if sender:
            header += f" de {sender}"
        if date:
            header += f" ({date})"
        if subject:
            header += f"\nSujet: {subject}"
        
        # Content
        content = doc.page_content
        
        chunk = f"{header}\n---\n{content}"
        context_parts.append(chunk)
    
    return "\n\n" + "="*50 + "\n\n".join(context_parts)


def create_email_messages(
    question: str,
    documents: List[Document],
    history: Optional[List[dict]] = None
) -> List[dict]:
    """
    Create prompt in OpenAI messages format for email chat.
    
    Args:
        question: User's question
        documents: Retrieved email chunks
        history: Optional conversation history
        
    Returns:
        List of message dicts for OpenAI API
    """
    context = format_email_context(documents)
    
    # Format history if available
    history_str = ""
    if history:
        recent = history[-6:]  # Last 3 turns
        if recent:
            history_str = "HISTORIQUE:\n"
            for msg in recent:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    history_str += f"Q: {content}\n"
                else:
                    # Truncate long responses
                    if len(content) > 150:
                        content = content[:150] + "..."
                    history_str += f"R: {content}\n"
            history_str += "\n"
    
    user_content = f"""{history_str}EMAILS TROUVÉS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Répondez en utilisant les emails ci-dessus
- Citez l'expéditeur et la date
- Utilisez le format [1], [2] pour les sources
- Soyez précis et utile"""
    
    messages = [
        {"role": "system", "content": EMAIL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    return messages
