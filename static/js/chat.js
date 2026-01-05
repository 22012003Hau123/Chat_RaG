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
        
        // Limit to first 2 sources
        const maxSources = 2;
        const displaySources = sources.slice(0, maxSources);
        const remainingCount = sources.length - maxSources;
        
        // Process links in sources
        const sourceLinks = displaySources.map(source => {
            const match = source.match(/\[(.*?)\]\((.*?)\)/);
            if (match) {
                return `<a href="${match[2]}" target="_blank">${match[1]}</a>`;
            }
            return source;
        });
        
        let sourcesText = `Sources: ${sourceLinks.join(', ')}`;
        if (remainingCount > 0) {
            sourcesText += ` <span style="opacity: 0.7;">+ ${remainingCount} more</span>`;
        }
        
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
}

function closeUploadModal() {
    document.getElementById('uploadModal').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadArea').innerHTML = `
        <div class="upload-placeholder" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📁</div>
            <p>Click to select a file or drag and drop</p>
        </div>
    `;
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
            </div>
        `;
        
        // Close modal after 2 seconds
        setTimeout(() => {
            closeUploadModal();
        }, 2000);
        
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
