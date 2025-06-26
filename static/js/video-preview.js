document.addEventListener("DOMContentLoaded", () => {
    let isDragging = false;

    const container = document.querySelector(".preview-container.video-preview");
    const video = document.querySelector("video");
    const hitArea = container.querySelector('.hit-area');
    const progressFill = container.querySelector(".progress-fill");
    const knob = container.querySelector(".progress-knob");
    const timeLeftDisplay = container.querySelector(".time-left");

    const startFrame = parseInt(container.dataset.clipStartFrame);
    const endFrame = parseInt(container.dataset.clipEndFrame);
    const fps = parseFloat(container.dataset.fps);
    const startSeconds = startFrame / fps;
    const endSeconds = endFrame / fps;
    const keyframeStartFrame = parseFloat(container.dataset.startFrame);
    const keyframeStartTime = (startFrame + keyframeStartFrame) / fps;

    const startMillisecondsDiv = document.getElementById("startMilliseconds");
    const endMillisecondsDiv = document.getElementById("endMilliseconds");
    updateDetailedViewInformation(Math.round(keyframeStartTime * 1000));

    function updateDetailedViewInformation(startTime) {
        startMillisecondsDiv.textContent = startTime;
        endMillisecondsDiv.textContent = Math.round((endFrame / fps) * 1000);
    }

    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${mins}:${secs}`;
    }

    function updateProgressBar(currentTime) {
        const remaining = endSeconds - currentTime;
        timeLeftDisplay.textContent = formatTime(remaining);
        const percent = ((currentTime - startSeconds) / (endSeconds - startSeconds)) * 100;
        const clampedPercent = Math.max(0, Math.min(100, percent));
        progressFill.style.width = `${clampedPercent}%`;
        knob.style.left = `${clampedPercent}%`;
    }

    function seek(e) {
        const rect = hitArea.getBoundingClientRect();
        let clickX = e.clientX - rect.left;
        clickX = Math.min(Math.max(0, clickX), rect.width);
        const percent = clickX / rect.width;
        const newTime = startSeconds + percent * (endSeconds - startSeconds);
        updateDetailedViewInformation(Math.round(newTime * 1000));
        if (!isNaN(video.duration)) {
            video.currentTime = newTime;
            updateProgressBar(newTime);
        }
    }

    video.addEventListener("loadedmetadata", () => {
        video.volume = 0.08;
        video.currentTime = keyframeStartTime;
    });

    video.addEventListener("timeupdate", () => {
        if (!isDragging) {
            updateProgressBar(video.currentTime);
            if (video.currentTime >= endSeconds) {
                video.pause();
                video.currentTime = startSeconds;
                setTimeout(() => {
                    if (container.matches(':hover')) {
                        video.play().catch(() => { });
                    }
                }, 100);
            }
        }
    });

    container.addEventListener("mouseenter", () => {
        video.pause();
        video.currentTime = keyframeStartTime;;

        const onSeeked = () => {
            video.play().catch(() => { });
            video.removeEventListener("seeked", onSeeked);
        };

        video.addEventListener("seeked", onSeeked);
    });

    container.addEventListener("mouseleave", () => {
        video.pause();
        video.currentTime = keyframeStartTime;;
    });

    hitArea.addEventListener('mousedown', (e) => {
        isDragging = true;
        video.pause();
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
            video.play().catch(() => { });
            seek(e);
        }
    });
});