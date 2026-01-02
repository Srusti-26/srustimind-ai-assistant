// Voice input fallback for browsers without speech recognition
function initVoiceFallback() {
    // Check if speech recognition is available
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        // Replace voice button with text input prompt
        const voiceBtn = document.getElementById('voiceInputBtn');
        voiceBtn.title = 'Voice input not supported - Click for text prompt';
        voiceBtn.style.opacity = '0.6';
        
        voiceBtn.addEventListener('click', () => {
            const text = prompt('Enter your message:');
            if (text && text.trim()) {
                document.getElementById('questionInput').value = text.trim();
                showNotification('Text entered successfully', 'success');
            }
        });
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initVoiceFallback);