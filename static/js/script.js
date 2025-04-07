document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const avatar = document.getElementById('avatar');
    const statusContainer = document.getElementById('status-container');
    
    // Emotion bars
    const joyBar = document.getElementById('joy-bar');
    const sadnessBar = document.getElementById('sadness-bar');
    const angerBar = document.getElementById('anger-bar');
    const fearBar = document.getElementById('fear-bar');
    const curiosityBar = document.getElementById('curiosity-bar');
    
    // Check server health on load
    checkServerHealth();
    
    // Start checking initialization status
    checkInitializationStatus();
    
    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Reset height if empty
        if (this.value.length === 0) {
            this.style.height = '';
        }
    });
    
    // Send message when Enter key is pressed (without Shift)
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Add keyboard shortcut for sending messages
    document.addEventListener('keydown', function(e) {
        // Command+Enter or Ctrl+Enter to send
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendButton.addEventListener('click', sendMessage);
    
    function checkServerHealth() {
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok') {
                    if (data.gemini_available) {
                        showStatusMessage('Connected to Gemini API', false);
                    } else {
                        showStatusMessage('Warning: Gemini API not available. Using fallback responses.', true);
                    }
                } else {
                    showStatusMessage('Server health check failed', true);
                }
            })
            .catch(error => {
                showStatusMessage('Failed to connect to server', true);
                console.error('Health check error:', error);
            });
    }
    
    function showStatusMessage(message, isError = false) {
        statusContainer.textContent = message;
        statusContainer.className = isError ? 
            'status-container error' : 'status-container success';
        statusContainer.style.display = 'block';
        setTimeout(() => {
            statusContainer.style.display = 'none';
        }, 5000);
    }
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message.length === 0) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        userInput.value = '';
        userInput.style.height = '';
        
        // Show typing indicator
        addTypingIndicator();
        
        // Send to backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add bot response
            addMessage(data.response, 'bot');
            
            // Update avatar
            updateAvatar(data.avatar_shape);
            
            // Update emotion bars
            updateEmotionBars(data.emotions);
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error processing your request.', 'system');
            showStatusMessage('Error connecting to server: ' + error.message, true);
        });
    }
    
    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-indicator';
        typingDiv.innerHTML = '<span></span><span></span><span></span>';
        typingDiv.id = 'typing-indicator';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function updateAvatar(shape) {
        // Remove all shape classes first
        avatar.classList.remove('circle', 'triangle', 'square');
        
        // Add the new shape class
        if (shape === 'Circle' || shape === 'Triangle' || shape === 'Square') {
            avatar.classList.add(shape.toLowerCase());
        }
    }
    
    function updateEmotionBars(emotions) {
        if (!emotions) return;
        
        // Update each emotion bar
        joyBar.style.width = `${emotions.joy * 100}%`;
        sadnessBar.style.width = `${emotions.sadness * 100}%`;
        angerBar.style.width = `${emotions.anger * 100}%`;
        fearBar.style.width = `${emotions.fear * 100}%`;
        curiosityBar.style.width = `${emotions.curiosity * 100}%`;
    }
    
    // Add initialization status check
    function checkInitializationStatus() {
        const statusContainer = document.getElementById('status-container');
        
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.initializing) {
                    // Still initializing
                    showStatusMessage('Galatea AI is still initializing. You can start chatting, but responses may be delayed.', false);
                    setTimeout(checkInitializationStatus, 3000); // Check again in 3 seconds
                } else if (data.is_initialized) {
                    // Initialization complete
                    showStatusMessage('Galatea AI is ready!', false);
                    setTimeout(() => {
                        statusContainer.style.display = 'none';
                    }, 3000);
                } else {
                    // Something wrong with initialization
                    showStatusMessage('Galatea initialization is taking longer than expected. You can still chat, but with limited functionality.', true);
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                showStatusMessage('Error checking Galatea status', true);
            });
    }
});
