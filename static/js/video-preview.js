document.addEventListener("DOMContentLoaded", () => {

    const container = document.querySelector(".preview-container");
    const video = document.querySelector("video");
    const fps = parseFloat(container.dataset.fps);
    const keyframeStartFrame = parseFloat(container.dataset.startFrame);
    const clipStartFrame = parseFloat(container.dataset.clipStartFrame);
    const keyframeStartTime = (clipStartFrame + keyframeStartFrame) / fps;

    video.addEventListener("loadeddata", () => {
        video.volume = 0.08;
        video.currentTime = keyframeStartTime;
    });
});1