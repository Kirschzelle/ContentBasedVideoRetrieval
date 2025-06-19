document.addEventListener("DOMContentLoaded", () => {
    const containers = document.querySelectorAll(".preview-container.video-preview");

    containers.forEach(container => {
        const video = container.querySelector("video");
        const hitArea = container.querySelector('.hit-area');
        const progressFill = container.querySelector(".progress-fill");
        const knob = container.querySelector(".progress-knob");
        const timeLeftDisplay = container.querySelector(".time-left");
        let isDragging = false;

        const startFrame = parseInt(container.dataset.startFrame)
        const endFrame = parseInt(container.dataset.endFrame)
        const fps = parseFloat(container.dataset.fps)

        const startSeconds = startFrame / fps;
        const endSeconds = endFrame / fps;

        const START_TIME = startSeconds //parseFloat(container.dataset.startTime) || 0; TODO: Change this back once we have keyframe start timer

        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
            return `${mins}:${secs}`;
        }

        function updateUI(currentTime) {
            const remaining = endSeconds - currentTime;
            timeLeftDisplay.textContent = formatTime(remaining);
            const percent = ((currentTime - startSeconds) / (endSeconds - startSeconds)) * 100;
            progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
            knob.style.left = `${Math.max(0, Math.min(100, percent))}%`;
        }

        function seek(e) {
            const rect = hitArea.getBoundingClientRect();
            let clickX = e.clientX - rect.left;
            clickX = Math.min(Math.max(0, clickX), rect.width);
            const percent = clickX / rect.width;
            const newTime = startSeconds + percent * (endSeconds - startSeconds);
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

            if (video.currentTime >= endSeconds) {
                video.pause();
                video.currentTime = START_TIME;

                setTimeout(() => {
                    if (container.matches(':hover')) {
                        video.play();
                    }
                }, 100);
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
                video.play();
                seek(e);
            }
        });
    });
});
