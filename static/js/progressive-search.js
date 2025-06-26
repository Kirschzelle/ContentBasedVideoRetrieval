let activeVideo = null;
let activeContainer = null;

document.addEventListener("DOMContentLoaded", function () {
    const query = new URLSearchParams(window.location.search).get("q");
    const resultContainer = document.getElementById("progressive-results");
    const loadMoreBtn = document.getElementById("load-more");
    const returnedKeyframes = new Set();
    let resultsFound = false;
    let stop = false;

    let fetchInProgress = false;

    async function fetchNextResult() {
        if (!query || stop || fetchInProgress) return;
        fetchInProgress = true;

        const params = new URLSearchParams();
        params.append("q", query);
        [...returnedKeyframes].forEach(id => params.append("returned[]", id));

        const filtersRaw = new URLSearchParams(window.location.search).get("filters");
        if (filtersRaw) {
            filtersRaw.split(",").forEach(pair => params.append("filters[]", pair));
        }

        try {
            const response = await fetch(`/api/search/?${params.toString()}`);
            const data = await response.json();

            if (data.done || !data.keyframe_id) {
                if (!resultsFound) {
                    resultContainer.innerHTML = `<p>No clips found matching "${query}".</p>`;
                }
                stop = true;
                loadMoreBtn.disabled = true;
                loadMoreBtn.textContent = "No more results";
                return;
            }

            resultsFound = true;
            returnedKeyframes.add(data.keyframe_id);

            const div = document.createElement("div");
            div.className = "clip-card preview-container video-preview";
            div.setAttribute("data-video-id", data.media_url);
            div.setAttribute("data-start-frame", data.frame);
            div.setAttribute("data-clip-start-frame", data.clip_start_frame);
            div.setAttribute("data-clip-end-frame", data.clip_end_frame);
            div.setAttribute("data-fps", data.fps);
            div.innerHTML = `
        <a href="/detailed_view/${data.keyframe_id}?q=${encodeURIComponent(query)}">
          <img src="${data.thumbnail}" alt="Keyframe" class="thumbnail draggable-image" draggable="true" />
        </a>
        <div class="video-overlay">
          <div class="progress-bar">
            <div class="hit-area"></div>
            <div class="progress-fill"></div>
            <div class="progress-knob"></div>
          </div>
          <span class="time-left">0:00</span>
        </div>
        `;

            resultContainer.appendChild(div);
            initVideoPreviewsForElement(div, data);

            setTimeout(() => {
                fetchInProgress = false;
                fetchNextResult();
            }, 1);

        } catch (err) {
            console.error("Error fetching result:", err);
            stop = true;
            fetchInProgress = false;
        }
    }

    if (query) {
        loadMoreBtn.style.display = "none";
        fetchNextResult();
    }
});

function initVideoPreviewsForElement(container, data) {
    if (container.dataset.initialized) return
    container.dataset.initialized = "true";

    let isDragging = false;
    let sharedVideo = document.getElementById(data.media_url);
    if (!sharedVideo) {
        sharedVideo = document.createElement("video");
        sharedVideo.id = data.keyframe_id;
        sharedVideo.muted = true;
        sharedVideo.playsInline = true;
        sharedVideo.preload = "auto";
        sharedVideo.src = data.media_url;
        document.getElementById("video-pool").appendChild(sharedVideo);
    }
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
        sharedVideo.currentTime = keyframeStartTime;

        const onSeeked = () => {
            sharedVideo.play().catch(() => { });
            sharedVideo.removeEventListener("seeked", onSeeked);
        };

        sharedVideo.addEventListener("seeked", onSeeked);
    });

    container.addEventListener("mouseleave", () => {
        if (activeContainer === container) {
            sharedVideo.pause();
            sharedVideo.currentTime = keyframeStartTime;
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
}