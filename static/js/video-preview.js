document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById("video-preview");
    const video = container.querySelector("video");
    const hitArea = document.querySelector('.hit-area');
    const progressFill = document.getElementById("progress-fill");
    const knob = document.getElementById("progress-knob");
    const timeLeftDisplay = document.getElementById("time-left");
    let isDragging = false;

    const START_TIME = parseFloat(container.dataset.startTime) || 0;

    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${mins}:${secs}`;
    }

    function updateUI(currentTime) {
        const remaining = video.duration - currentTime;
        timeLeftDisplay.textContent = formatTime(remaining);
        const percent = (currentTime / video.duration) * 100;
        progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
        knob.style.left = `${percent}%`;
    }

    function seek(e) {
        const rect = hitArea.getBoundingClientRect();
        let clickX = e.clientX - rect.left;
        clickX = Math.min(Math.max(0, clickX), rect.width);
        const percent = clickX / rect.width;
        const newTime = percent * video.duration;
        if (!isNaN(video.duration)) {
            video.currentTime = newTime;
            updateUI(newTime);
        }
    }

    video.addEventListener("loadedmetadata", () => {
        video.currentTime = START_TIME;
    });

    video.addEventListener("seeked", () => {
        updateUI(video.currentTime);
    });

    video.addEventListener("timeupdate", () => {
        if (!isDragging) {
            updateUI(video.currentTime);
        }
    });

    container.addEventListener("mouseenter", () => {
        if (Math.abs(video.currentTime - START_TIME) > 0.1) {
            video.currentTime = START_TIME;
        }
        video.play();
    });

    container.addEventListener("mouseleave", () => {
        video.pause();
        video.currentTime = START_TIME;
    });

    hitArea.addEventListener('mousedown', (e) => {
        isDragging = true;
        video.pause()
        seek(e);
    });

    window.addEventListener('mousemove', (e) => {
        if (isDragging) {
            seek(e);
        }
    });

    window.addEventListener('mouseup', (e) => {
        if (isDragging) {
            isDragging = false;
            video.play();
            seek(e);
        }
    });
});
