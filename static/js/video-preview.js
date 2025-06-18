document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById("video-preview");
    const video = container.querySelector("video");
    const timeLeftDisplay = document.getElementById("time-left");
    const progressFill = document.getElementById("progress-fill");

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
    }

    video.addEventListener("loadedmetadata", () => {
        video.currentTime = START_TIME;
    });

    video.addEventListener("seeked", () => {
        updateUI(video.currentTime);
    });

    video.addEventListener("timeupdate", () => {
        updateUI(video.currentTime);
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
});
