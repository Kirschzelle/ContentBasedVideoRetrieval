document.addEventListener("DOMContentLoaded", () => {
    let isDragging = false;

    const container = document.querySelector(".preview-container");
    const video = document.querySelector("video");

    const startFrame = parseInt(container.dataset.clipStartFrame);
    const endFrame = parseInt(container.dataset.clipEndFrame);
    const fps = parseFloat(container.dataset.fps);
    const startSeconds = startFrame / fps;
    const endSeconds = endFrame / fps;
    const keyframeStartFrame = parseFloat(container.dataset.startFrame);
    const keyframeStartTime = (startFrame + keyframeStartFrame) / fps;

    video.addEventListener("loadedmetadata", () => {
        video.volume = 0.08;
        video.currentTime = keyframeStartTime;
    });

    video.addEventListener("timeupdate", () => {
        if (!isDragging) {
            if (video.currentTime >= endSeconds) {
                video.pause();
                video.currentTime = startSeconds;
                setTimeout(() => {
                    video.play().catch(() => { });
                }, 100);
            }
        }
    });
});