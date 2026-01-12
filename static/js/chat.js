// DOM elements
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const imagePreviewArea = document.getElementById('imagePreviewArea');
const imageInput = document.getElementById('imageInput');

// Image attachment state
let attachedImage = null;

// Generate UUID for session ID
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Session management (tab-scoped with sessionStorage)
let sessionId = sessionStorage.getItem('chat_session_id');
if (!sessionId) {
    sessionId = generateUUID();
    sessionStorage.setItem('chat_session_id', sessionId);
    console.log('🆔 New session created:', sessionId);
} else {
    console.log('🆔 Existing session loaded:', sessionId);
}

// Function to reset session (for new conversation)
function resetSession() {
    sessionId = generateUUID();
    sessionStorage.setItem('chat_session_id', sessionId);
    console.log('🔄 Session reset:', sessionId);
    // Clear chat UI
    chatMessages.innerHTML = '';
    // Re-add greeting
    addMessage("Bonjour! Je suis votre assistant IA pour les documents Auchan. Comment puis-je vous aider ?", false);
}

// ============================================================================
// CONVERSATION HISTORY MANAGEMENT
// ============================================================================

let currentConversationId = localStorage.getItem('current_conversation_id') || null;

// Load conversations on page load
document.addEventListener('DOMContentLoaded', loadConversations);

async function loadConversations() {
    try {
        const response = await fetch('/api/conversations');
        const data = await response.json();
        renderConversationList(data.conversations);
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

function renderConversationList(conversations) {
    const listEl = document.getElementById('conversationList');
    if (!listEl) return;
    
    // Group by date
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 86400000).toDateString();
    
    const groups = {
        'Today': [],
        'Yesterday': [],
        'Previous': []
    };
    
    conversations.forEach(conv => {
        const convDate = new Date(conv.updated_at || conv.created_at).toDateString();
        if (convDate === today) {
            groups['Today'].push(conv);
        } else if (convDate === yesterday) {
            groups['Yesterday'].push(conv);
        } else {
            groups['Previous'].push(conv);
        }
    });
    
    let html = '';
    for (const [label, convs] of Object.entries(groups)) {
        if (convs.length === 0) continue;
        
        html += `<div class="date-group">`;
        html += `<div class="date-label">${label}</div>`;
        
        convs.forEach(conv => {
            const isActive = conv.id === currentConversationId ? 'active' : '';
            html += `
                <div class="conversation-item ${isActive}" onclick="loadConversation('${conv.id}')">
                    <span class="conversation-title">${escapeHtml(conv.title)}</span>
                    <div class="conversation-actions">
                        <button class="conversation-edit" onclick="event.stopPropagation(); renameConversation('${conv.id}', '${escapeHtml(conv.title)}')" title="Rename">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                        </button>
                        <button class="conversation-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')" title="Delete">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        </button>
                    </div>
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    listEl.innerHTML = html || '<div style="color: rgba(255,255,255,0.5); padding: 16px; text-align: center;">No conversations yet</div>';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function createNewChat() {
    try {
        const response = await fetch('/api/conversations', { method: 'POST' });
        const data = await response.json();
        
        currentConversationId = data.id;
        localStorage.setItem('current_conversation_id', data.id);
        
        // Clear chat and reset session
        chatMessages.innerHTML = '';
        addMessage("Bonjour! Je suis votre assistant IA pour les documents Auchan. Comment puis-je vous aider ?", false);
        
        // Reload sidebar
        loadConversations();
    } catch (error) {
        console.error('Error creating conversation:', error);
    }
}

async function loadConversation(convId) {
    try {
        const response = await fetch(`/api/conversations/${convId}`);
        const data = await response.json();
        
        currentConversationId = convId;
        localStorage.setItem('current_conversation_id', convId);
        
        // Clear and load messages
        chatMessages.innerHTML = '';
        
        data.messages.forEach(msg => {
            addMessage(msg.content, msg.role === 'user');
        });
        
        // Update sidebar highlight
        loadConversations();
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

async function deleteConversation(convId) {
    // Find the conversation item
    const items = document.querySelectorAll('.conversation-item');
    for (const item of items) {
        if (item.onclick && item.onclick.toString().includes(convId)) {
            const actionsDiv = item.querySelector('.conversation-actions');
            if (actionsDiv) {
                // Replace with confirm buttons
                actionsDiv.innerHTML = `
                    <span style="font-size: 11px; color: rgba(255,255,255,0.7);">Delete?</span>
                    <button class="confirm-yes" onclick="event.stopPropagation(); confirmDelete('${convId}')">Yes</button>
                    <button class="confirm-no" onclick="event.stopPropagation(); loadConversations()">No</button>
                `;
            }
            break;
        }
    }
}

async function confirmDelete(convId) {
    try {
        await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
        
        if (convId === currentConversationId) {
            currentConversationId = null;
            localStorage.removeItem('current_conversation_id');
            chatMessages.innerHTML = '';
            addMessage("Bonjour! Je suis votre assistant IA pour les documents Auchan. Comment puis-je vous aider ?", false);
        }
        
        loadConversations();
    } catch (error) {
        console.error('Error deleting conversation:', error);
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
}

async function renameConversation(convId, currentTitle) {
    // Find the conversation item and make title editable
    const items = document.querySelectorAll('.conversation-item');
    for (const item of items) {
        if (item.onclick.toString().includes(convId)) {
            const titleSpan = item.querySelector('.conversation-title');
            if (titleSpan) {
                // Replace span with input
                const input = document.createElement('input');
                input.type = 'text';
                input.value = currentTitle;
                input.className = 'conversation-title-input';
                input.style.cssText = 'background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); border-radius: 4px; color: white; padding: 4px 8px; width: 100%; font-size: 14px;';
                
                titleSpan.replaceWith(input);
                input.focus();
                input.select();
                
                // Save on Enter or blur
                const saveTitle = async () => {
                    const newTitle = input.value.trim();
                    if (newTitle && newTitle !== currentTitle) {
                        try {
                            await fetch(`/api/conversations/${convId}/title`, {
                                method: 'PUT',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ title: newTitle })
                            });
                        } catch (error) {
                            console.error('Error renaming:', error);
                        }
                    }
                    loadConversations();
                };
                
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        saveTitle();
                    } else if (e.key === 'Escape') {
                        loadConversations();
                    }
                });
                
                input.addEventListener('blur', saveTitle);
            }
            break;
        }
    }
}

function addMessage(text, isUser, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    if (isUser) {
        avatar.textContent = '👤';
    } else {
        // Use custom bot avatar image
        const botImg = document.createElement('img');
        botImg.src = '/static/images/bot-avatar.jpeg';
        botImg.alt = 'Bot';
        botImg.style.width = '100%';
        botImg.style.height = '100%';
        botImg.style.objectFit = 'cover';
        botImg.style.borderRadius = '50%';
        avatar.appendChild(botImg);
    }
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Process markdown conversion if library is available
    if (!isUser && typeof marked !== 'undefined') {
        content.innerHTML = marked.parse(text);
        
        // IMPORTANT: Process all links to open in new tab
        const links = content.querySelectorAll('a');
        links.forEach(link => {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        });
        
        // IMPORTANT: Process all images to ensure proper sizing and add click-to-enlarge
        const images = content.querySelectorAll('img');
        images.forEach(img => {
            // Force size constraints
            img.style.maxHeight = '200px';
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            img.style.width = 'auto';
            img.style.objectFit = 'contain';
            img.style.cursor = 'pointer';
            img.style.borderRadius = '8px';
            img.style.margin = '10px 0';
            img.style.display = 'block';
            
            // Add click handler for lightbox
            img.addEventListener('click', function() {
                openImageModal(this.src, this.alt || 'Image');
            });
        });
    } else {
        content.textContent = text;
    }
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        // Remove duplicates by filename
        const uniqueSources = [];
        const seenNames = new Set();
        
        for (const source of sources) {
            // Extract filename from markdown link or plain text
            const match = source.match(/\[(.*?)\]/);
            const fileName = match ? match[1] : source;
            
            if (!seenNames.has(fileName)) {
                seenNames.add(fileName);
                uniqueSources.push(source);
            }
        }
        
        // Process links in sources (show ALL, no limit)
        const sourceLinks = uniqueSources.map(source => {
            const match = source.match(/\[(.*?)\]\((.*?)\)/);
            if (match) {
                return `<a href="${match[2]}" target="_blank">${match[1]}</a>`;
            }
            return source;
        });
        
        // Display all sources without "+ X more"
        let sourcesText = `Sources: ${sourceLinks.join(', ')}`;
        
        sourcesDiv.innerHTML = sourcesText;
        content.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question && !attachedImage) return;
    
    // Add user message with image thumbnail if present
    if (attachedImage) {
        // Create message with thumbnail
        const reader = new FileReader();
        reader.onload = function(e) {
            const messageText = question || 'Analyze this image';
            const messageWithImage = `
                <div class="user-message-text">${messageText}</div>
                <img src="${e.target.result}" class="user-image-thumbnail" alt="Uploaded image" />
            `;
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '\u{1F464}';  // 👤 User emoji
            
            const content = document.createElement('div');
            content.className = 'message-content';
            content.innerHTML = messageWithImage;
            
            messageDiv.appendChild(content);
            messageDiv.appendChild(avatar);
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };
        reader.readAsDataURL(attachedImage);
    } else {
        addMessage(question, true);
    }
    
    questionInput.value = '';
    
    // Disable input while processing
    sendButton.disabled = true;
    questionInput.disabled = true;
    sendButton.innerHTML = '<span class="loading">Thinking</span>';
    
    try {
        // Use FormData to send both text and image
        const formData = new FormData();
        formData.append('question', question || 'Analyze this image');
        formData.append('method', 'mmr');
        formData.append('session_id', sessionId);
        
        if (attachedImage) {
            formData.append('image', attachedImage);
        }
        
        const response = await fetch('/ask', {
            method: 'POST',
            body: formData  // No Content-Type header - browser sets it automatically with boundary
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update session ID from response
        if (data.session_id) {
            sessionId = data.session_id;
            sessionStorage.setItem('chat_session_id', sessionId);
        }
        
        addMessage(data.answer, false, data.sources);
        
        // Clear attached image after sending
        clearImagePreview();
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Désolé, une erreur est survenue. Veuillez réessayer.', false);
    } finally {
        sendButton.disabled = false;
        questionInput.disabled = false;
        sendButton.textContent = 'Send';
        questionInput.focus();
    }
}

// ============= IMAGE HANDLING =============

// Handle keyboard shortcuts in textarea
function handleInputKeydown(event) {
    // Send on Enter (without Shift)
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Open image picker
function openImagePicker() {
    imageInput.click();
}

// Handle manual file selection (image or PDF)
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        // Accept images and PDF only
        const validTypes = ['image/', '.pdf'];
        const isValid = validTypes.some(type => 
            file.type.includes(type) || file.name.toLowerCase().endsWith(type)
        );
        
        if (isValid) {
            attachedImage = file;
            showFilePreview(file);
        } else {
            alert('Unsupported file type. Please upload image or PDF.');
        }
    }
}

// Handle paste event for Ctrl+V image paste
questionInput.addEventListener('paste', (event) => {
    const items = event.clipboardData.items;
    
    for (let item of items) {
        if (item.type.indexOf('image') !== -1) {
            event.preventDefault();
            
            const file = item.getAsFile();
            attachedImage = file;
            showFilePreview(file);  // FIXED: was showImagePreview
            
            console.log('📋 Image pasted from clipboard');
            break;
        }
    }
});

// Show file preview (image or document)
function showFilePreview(file) {
    const isImage = file.type.startsWith('image/');
    
    if (isImage) {
        // Image preview with thumbnail
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreviewArea.innerHTML = `
                <div class="image-preview">
                    <img src="${e.target.result}" alt="Preview" />
                    <button class="remove-image" onclick="clearImagePreview()" title="Remove file">
                        ✕
                    </button>
                    <div class="image-filename">${file.name}</div>
                </div>
            `;
            imagePreviewArea.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        // Document preview with icon
        const fileIcon = getFileIcon(file.name);
        imagePreviewArea.innerHTML = `
            <div class="file-preview">
                <div class="file-icon">${fileIcon}</div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <button class="remove-image" onclick="clearImagePreview()" title="Remove file">
                    ✕
                </button>
            </div>
        `;
        imagePreviewArea.style.display = 'block';
    }
}

// Get icon for file type
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (ext === 'pdf') {
        return '📄';
    }
    return '📎';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Clear image preview
function clearImagePreview() {
    attachedImage = null;
    imagePreviewArea.innerHTML = '';
    imagePreviewArea.style.display = 'none';
    imageInput.value = ''; // Reset file input
}

// ============= INITIALIZATION =============

// Focus input on load
questionInput.focus();

// Add initial greeting
addMessage("Bonjour! Je suis votre assistant IA pour les documents Auchan. Comment puis-je vous aider ?", false);

// Make all links in chat open in new tabs
function makeLinksOpenInNewTab() {
    const links = document.querySelectorAll('.message-content a');
    links.forEach(link => {
        if (!link.hasAttribute('target')) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer'); // Security best practice
        }
    });
}

// Run on initial load and whenever new messages are added
makeLinksOpenInNewTab();

// Monitor for new messages and update links
const chatObserver = new MutationObserver(() => {
    makeLinksOpenInNewTab();
});

chatObserver.observe(chatMessages, {
    childList: true,
    subtree: true
});

// ============= IMAGE LIGHTBOX MODAL =============

function openImageModal(src, alt) {
    // Create modal backdrop
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        cursor: zoom-out;
    `;
    
    // Create image container
    const imgContainer = document.createElement('div');
    imgContainer.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        position: relative;
    `;
    
    // Create full-size image
    const img = document.createElement('img');
    img.src = src;
    img.alt = alt;
    img.style.cssText = `
        max-width: 100%;
        max-height: 90vh;
        width: auto;
        height: auto;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    `;
    
    // Create close button
    const closeBtn = document.createElement('div');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: -40px;
        right: 0;
        color: white;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        transition: background 0.3s;
    `;
    
    closeBtn.addEventListener('mouseenter', () => {
        closeBtn.style.background = 'rgba(255, 255, 255, 0.2)';
    });
    
    closeBtn.addEventListener('mouseleave', () => {
        closeBtn.style.background = 'rgba(255, 255, 255, 0.1)';
    });
    
    // Close on click
    const closeModal = () => {
        modal.remove();
    };
    
    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // ESC key to close
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closeModal();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
    
    // Assemble and show
    imgContainer.appendChild(closeBtn);
    imgContainer.appendChild(img);
    modal.appendChild(imgContainer);
    document.body.appendChild(modal);
}

// ============= DOCUMENT UPLOAD FUNCTIONALITY =============

function openUploadModal() {
    document.getElementById('uploadModal').style.display = 'flex';
    // Auto-fill date for text tab (French format)
    const today = new Date();
    const dateStr = today.toLocaleDateString('fr-FR');
    const textDateEl = document.getElementById('textDate');
    if (textDateEl) {
        textDateEl.value = dateStr;
    }
    // Ensure file tab is active by default
    switchTab('file');
}

// Switch between File and Text tabs
function switchTab(tabName) {
    // Remove active from all tabs
    document.querySelectorAll('.upload-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    
    // Activate selected tab
    const selectedTab = document.querySelector(`.upload-tab[data-tab="${tabName}"]`);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Show selected tab content
    const tabId = tabName === 'file' ? 'fileTab' : 'textTab';
    const tabContent = document.getElementById(tabId);
    if (tabContent) {
        tabContent.style.display = 'block';
    }
}

function resetUploadForm() {
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <input type="file" id="fileInput" accept=".pdf,.docx,.pptx" onchange="handleFileSelect(event)" style="display: none;">
        <div class="upload-placeholder" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📁</div>
            <p>Click to select a file or drag and drop</p>
        </div>
    `;
}

function resetTextForm() {
    document.getElementById('textTitle').value = '';
    document.getElementById('textEmail').value = '';
    document.getElementById('textContent').value = '';
    // Reset date
    const today = new Date();
    document.getElementById('textDate').value = today.toLocaleDateString('fr-FR');
    // Hide status
    const textStatus = document.getElementById('textStatus');
    textStatus.className = 'text-status';
    textStatus.textContent = '';
}

function closeUploadModal() {
    document.getElementById('uploadModal').style.display = 'none';
    resetUploadForm();
    resetTextForm();
}

// Submit text function
async function submitText() {
    const title = document.getElementById('textTitle').value.trim();
    const email = document.getElementById('textEmail').value.trim();
    const date = document.getElementById('textDate').value;
    const text = document.getElementById('textContent').value.trim();
    const textStatus = document.getElementById('textStatus');
    const submitBtn = document.getElementById('textSubmitBtn');
    
    // Validation
    if (!title) {
        textStatus.className = 'text-status error';
        textStatus.textContent = '⚠️ Le titre est requis';
        return;
    }
    if (!email) {
        textStatus.className = 'text-status error';
        textStatus.textContent = '⚠️ L\'email est requis';
        return;
    }
    if (!email.includes('@')) {
        textStatus.className = 'text-status error';
        textStatus.textContent = '⚠️ Email invalide';
        return;
    }
    if (!text || text.length < 10) {
        textStatus.className = 'text-status error';
        textStatus.textContent = '⚠️ Le contenu doit avoir au moins 10 caractères';
        return;
    }
    
    // Show loading
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';
    textStatus.className = 'text-status loading';
    textStatus.textContent = '⏳ Traitement en cours...';
    
    try {
        const response = await fetch('/upload-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, email, date, text })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        
        // Show success
        textStatus.className = 'text-status success';
        textStatus.innerHTML = `
            ✅ <strong>Text ajouté avec succès!</strong><br>
            📄 ${result.filename}<br>
            📊 ${result.chunks_created} chunks créés
        `;
        
        // Reset form for new entry
        document.getElementById('textTitle').value = '';
        document.getElementById('textContent').value = '';
        
        // Change button to allow adding more
        submitBtn.disabled = false;
        submitBtn.textContent = 'Add Another Text';
        
    } catch (error) {
        console.error('Text upload error:', error);
        textStatus.className = 'text-status error';
        textStatus.textContent = `❌ Erreur: ${error.message}`;
        submitBtn.disabled = false;
        submitBtn.textContent = 'Add Text';
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

async function uploadFile(file) {
    // Validate file type
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'];
    if (!allowedTypes.includes(file.type)) {
        alert('Format non supporté. Utilisez PDF, DOCX ou PPTX.');
        return;
    }
    
    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        alert('Fichier trop volumineux. Taille maximale: 50MB');
        return;
    }
    
    // Show upload progress
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div class="upload-progress">
            <div class="upload-spinner">⏳</div>
            <p>Uploading ${file.name}...</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
    `;
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        
        // Show success
        uploadArea.innerHTML = `
            <div class="upload-success">
                <div class="success-icon">✅</div>
                <p><strong>Document ajouté avec succès!</strong></p>
                <p class="upload-detail">${result.filename}</p>
                <p class="upload-detail">${result.chunks_created} chunks created</p>
                <div style="margin-top: 15px;">
                    <button onclick="resetUploadForm()" class="upload-submit-btn" style="width: auto; padding: 10px 20px; margin-right: 10px;">
                        📂 Add Another File
                    </button>
                    <button onclick="closeUploadModal()" class="upload-submit-btn" style="width: auto; padding: 10px 20px; background: #666;">
                        Finish
                    </button>
                </div>
            </div>
        `;
        
        // Removed auto-close setTimeout
        
    } catch (error) {

        console.error('Upload error:', error);
        uploadArea.innerHTML = `
            <div class="upload-error">
                <div class="error-icon">❌</div>
                <p><strong>Erreur d'upload</strong></p>
                <p class="upload-detail">${error.message}</p>
                <button onclick="closeUploadModal()" class="retry-btn">Retry</button>
            </div>
        `;
    }
}

// Drag and drop support
document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('drag-over');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('drag-over');
            }, false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        }, false);
    }
});
