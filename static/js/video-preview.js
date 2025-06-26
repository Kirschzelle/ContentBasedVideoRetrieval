document.addEventListener("DOMContentLoaded", () => {
    let activeVideo = null;
    let activeContainer = null;

    const containers = document.querySelectorAll(".preview-container.video-preview");
    containers.forEach(container => {
        let isDragging = false;
        const videoId = container.dataset.videoId;
        const sharedVideo = document.getElementById(videoId);
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
            if (startMillisecondsDiv && endMillisecondsDiv) {
                startMillisecondsDiv.textContent = startTime;
                endMillisecondsDiv.textContent = Math.round((endFrame / fps) * 1000);
            }
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
            if (!isNaN(sharedVideo.duration)) {
                sharedVideo.currentTime = newTime;
                updateProgressBar(newTime);
            }
        }

        sharedVideo.addEventListener("loadedmetadata", () => {
            if (activeContainer === container) {
                sharedVideo.currentTime = keyframeStartTime;
            }
        });

        sharedVideo.addEventListener("timeupdate", () => {
            if (activeContainer === container && !isDragging) {
                updateProgressBar(sharedVideo.currentTime);
                if (sharedVideo.currentTime >= endSeconds) {
                    sharedVideo.pause();
                    sharedVideo.currentTime = startSeconds;
                    setTimeout(() => {
                        if (container.matches(':hover')) {
                            sharedVideo.play().catch(() => { });
                        }
                    }, 100);
                }
            }
        });

        container.addEventListener("mouseenter", () => {
            if (activeVideo && activeVideo !== sharedVideo) {
                activeVideo.pause();
                if (activeContainer) {
                    activeContainer.querySelector(".thumbnail").style.opacity = "1";
                    activeContainer.appendChild(activeVideo);
                }
            }

            activeVideo = sharedVideo;
            activeContainer = container;

            container.querySelector(".thumbnail").style.opacity = "0";
            container.appendChild(sharedVideo);
            sharedVideo.classList.add("preview-video");
            sharedVideo.style.opacity = "1";

            sharedVideo.pause();
            sharedVideo.currentTime = keyframeStartTime;;

            const onSeeked = () => {
                sharedVideo.play().catch(() => { });
                sharedVideo.removeEventListener("seeked", onSeeked);
            };

            sharedVideo.addEventListener("seeked", onSeeked);
        });

        container.addEventListener("mouseleave", () => {
            if (activeContainer === container) {
                sharedVideo.pause();
                sharedVideo.currentTime = keyframeStartTime;;
                document.getElementById("video-pool").appendChild(sharedVideo);
                container.querySelector(".thumbnail").style.opacity = "1";
                activeVideo = null;
                activeContainer = null;
            }
        });

        hitArea.addEventListener('mousedown', (e) => {
            if (activeContainer === container) {
                isDragging = true;
                sharedVideo.pause();
                seek(e);
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (isDragging && activeContainer === container) {
                seek(e);
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (isDragging && activeContainer === container) {
                isDragging = false;
                sharedVideo.play().catch(() => { });
                seek(e);
            }
        });
    });
});