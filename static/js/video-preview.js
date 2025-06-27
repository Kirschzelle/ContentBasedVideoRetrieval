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
    const startSeconds = 0 / fps;
    const endSeconds = video.duration;
    const keyframeStartFrame = parseFloat(container.dataset.startFrame);
    const keyframeStartTime = (startFrame + keyframeStartFrame) / fps;

    const startMillisecondsDiv = document.getElementById("startMilliseconds");
    const endMillisecondsDiv = document.getElementById("endMilliseconds");
    updateDetailedViewInformation(Math.round(keyframeStartTime * 1000));

    function updateDetailedViewInformation(startTime) {
        startMillisecondsDiv.value = startTime;
        endMillisecondsDiv.textContent = Math.round((endFrame / fps) * 1000);
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
        }
    }

    video.addEventListener("loadedmetadata", () => {
        video.volume = 0.08;
        video.currentTime = keyframeStartTime;
    });

    video.addEventListener("timeupdate", () => {
        if (!isDragging) {
            const currentTimeMs = Math.round(video.currentTime * 1000);
            startMillisecondsDiv.value = currentTimeMs;

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

        const onSeeked = () => {
            video.play().catch(() => { });
            video.removeEventListener("seeked", onSeeked);
        };

        video.addEventListener("seeked", onSeeked);
    });

    container.addEventListener("mouseleave", () => {
        video.pause();
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