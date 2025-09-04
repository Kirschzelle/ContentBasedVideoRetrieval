class DaVinciFrameCapture {
    constructor() {
        this.init();
    }

    init() {
        document.querySelectorAll('.davinci-capture-btn').forEach(button => {
            button.addEventListener('click', (e) => this.captureFrame(e));
        });
    }

    async captureFrame(event) {
        const button = event.target;
        const filterType = button.dataset.filterType;
        
        button.disabled = true;
        button.textContent = 'Capturing...';
        
        try {
            const response = await fetch('/davinci/get_current_frame/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`Frame captured at ${result.timecode} for ${filterType} filter`);
                this.showSuccess(button, `Frame captured at ${result.timecode}`);
                
                // For now, just show success - later we'll process and apply as filter
                if (result.redirect_url) {
                    // Could redirect to search results or update UI
                    console.log(`Would redirect to: ${result.redirect_url}`);
                }
            } else {
                this.showError(button, result.error || 'Failed to capture frame');
            }
            
        } catch (error) {
            console.error('Frame capture error:', error);
            this.showError(button, 'Network error occurred');
        } finally {
            setTimeout(() => {
                button.disabled = false;
                button.textContent = 'Get Current Frame from DaVinci';
            }, 2000);
        }
    }

    showSuccess(button, message) {
        const originalText = button.textContent;
        button.textContent = `✓ ${message}`;
        button.style.backgroundColor = '#28a745';
        button.style.color = 'white';
        
        setTimeout(() => {
            button.style.backgroundColor = '';
            button.style.color = '';
        }, 2000);
    }

    showError(button, message) {
        const originalText = button.textContent;
        button.textContent = `✗ ${message}`;
        button.style.backgroundColor = '#dc3545';
        button.style.color = 'white';
        
        setTimeout(() => {
            button.style.backgroundColor = '';
            button.style.color = '';
        }, 2000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DaVinciFrameCapture();
});