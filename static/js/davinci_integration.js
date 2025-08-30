/**
 * Simple DaVinci Resolve integration for search results.
 */

class DaVinciIntegration {
    constructor() {
        this.init();
    }
    
    async init() {
        await this.checkDaVinciStatus();
        this.addDaVinciButtons();
    }
    
    async checkDaVinciStatus() {
        try {
            const response = await fetch('/davinci/status/');
            const data = await response.json();
            
            this.daVinciAvailable = data.available;
            this.projectOpen = data.project_open;
            
            if (!data.available) {
                console.warn('DaVinci Resolve not available:', data.error);
            }
        } catch (error) {
            console.error('Failed to check DaVinci status:', error);
            this.daVinciAvailable = false;
        }
    }
    
    addDaVinciButtons() {
        // Find all search result items
        const searchResults = document.querySelectorAll('[data-keyframe-id]');
        
        searchResults.forEach(result => {
            const keyframeId = result.dataset.keyframeId;
            if (!keyframeId) return;
            
            // Create DaVinci button
            const button = document.createElement('button');
            button.className = 'davinci-btn';
            button.innerHTML = 'Send to DaVinci';
            button.title = 'Send clip to DaVinci Resolve preview';
            
            if (!this.daVinciAvailable) {
                button.disabled = true;
                button.title = 'DaVinci Resolve not available';
                button.innerHTML = 'DaVinci (offline)';
            } else if (!this.projectOpen) {
                button.title = 'Open a project in DaVinci Resolve first';
                button.innerHTML = 'DaVinci (no project)';
            }
            
            button.onclick = () => this.sendToDaVinci(keyframeId, button);
            
            // Add to result controls area
            const controls = result.querySelector('.result-controls') || 
                           result.querySelector('.result-actions') || 
                           result;
            controls.appendChild(button);
        });
    }
    
    async sendToDaVinci(keyframeId, button) {
        if (!this.daVinciAvailable || !this.projectOpen) {
            this.showMessage('Please make sure DaVinci Resolve is running with a project open.', 'error');
            return;
        }
        
        // Show loading state
        const originalText = button.innerHTML;
        button.innerHTML = 'Sending...';
        button.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append('keyframe_id', keyframeId);
            
            const response = await fetch('/davinci/send/', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showMessage(
                    `Sent to DaVinci! ${data.video_name} at ${data.timecode}`, 
                    'success'
                );
                
                // Track engagement - sending to DaVinci is a strong signal!
                this.trackDaVinciEngagement(keyframeId);
                
            } else {
                this.showMessage(`Failed: ${data.error}`, 'error');
            }
            
        } catch (error) {
            this.showMessage(`Error: ${error.message}`, 'error');
        } finally {
            // Restore button
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }
    
    trackDaVinciEngagement(keyframeId) {
        // Track that user sent clip to DaVinci - very strong engagement signal!
        if (window.searchInteractionId) {
            fetch('/track-engagement/', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    interaction_id: window.searchInteractionId,
                    keyframe_id: keyframeId,
                    engagement_data: {
                        'clip_copied': true,  // DaVinci preview is essentially copying
                        'video_opened_fullscreen': true,  // Opening in professional software
                        'was_used_as_filter': false,
                        'next_action': 'davinci_preview'
                    }
                })
            }).catch(e => console.log('Engagement tracking failed:', e));
        }
    }
    
    showMessage(message, type = 'info') {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#d1ecf1'};
            color: ${type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#0c5460'};
            padding: 12px 20px;
            border-radius: 4px;
            border: 1px solid ${type === 'success' ? '#c3e6cb' : type === 'error' ? '#f5c6cb' : '#bee5eb'};
            z-index: 1000;
            max-width: 400px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        `;
        
        document.body.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
        
        // Click to dismiss
        toast.onclick = () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        };
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.daVinciIntegration = new DaVinciIntegration();
});

// Add basic button styling
const style = document.createElement('style');
style.textContent = `
.davinci-btn {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-weight: bold;
    margin: 2px;
    transition: all 0.2s ease;
}

.davinci-btn:hover:not(:disabled) {
    background: linear-gradient(135deg, #ee5a24, #ff6b6b);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(238, 90, 36, 0.3);
}

.davinci-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
    opacity: 0.6;
}
`;
document.head.appendChild(style);