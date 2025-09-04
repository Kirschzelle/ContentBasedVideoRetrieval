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
                
                this.displayCapturedFrame(filterType, result.image_url, result.frame_id);
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

    displayCapturedFrame(filterType, imageUrl, frameId) {
        const filterAreas = document.querySelectorAll(`[id*="${filterType}-filter"]`);
        
        if (filterAreas.length === 0) {
            console.warn(`No filter areas found for type: ${filterType}`);
            return;
        }
        
        let targetArea = null;
        for (const area of filterAreas) {
            const img = area.querySelector('.preview-image');
            if (!img || img.style.display === 'none') {
                targetArea = area;
                break;
            }
        }
        
        if (!targetArea && filterAreas.length > 0) {
            targetArea = filterAreas[0];
        }
        
        if (targetArea) {
            const previewImg = targetArea.querySelector('.preview-image');
            const textElement = targetArea.querySelector('p');
            
            if (previewImg && textElement) {
                previewImg.src = imageUrl;
                previewImg.style.display = 'block';
                previewImg.dataset.keyframeId = `davinci_${frameId}`;
                previewImg.dataset.filterType = filterType;
                previewImg.classList.add('draggable-image');
                previewImg.draggable = true;
                
                textElement.style.display = 'none';
                
                console.log(`DaVinci frame displayed in ${targetArea.id} with ID: davinci_${frameId}`);
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DaVinciFrameCapture();
});