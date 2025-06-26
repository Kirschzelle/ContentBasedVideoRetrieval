document.getElementById("progressive-results").addEventListener("dragstart", (e) => {
    const img = e.target;
    if (img.classList.contains("draggable-image")) {
        const payload = {
            src: img.src,
            keyframeId: img.dataset.keyframeId,
        };
        e.dataTransfer.setData("text/plain", JSON.stringify(payload));
    }
});

document.querySelectorAll(".drag-drop-area").forEach(area => {
    const imgPreview = area.querySelector("img");

    area.addEventListener("click", () => {
        if (imgPreview) {
            imgPreview.src = "";
            imgPreview.style.display = "none";
        }
    });

    area.addEventListener("dragover", e => {
        e.preventDefault();
    });

    area.addEventListener("drop", e => {
        e.preventDefault();

        try {
            const data = JSON.parse(e.dataTransfer.getData("text/plain"));
            const keyframeId = data.keyframeId;
            const baseUrl = window.location.origin + window.location.pathname;

            const params = new URLSearchParams(window.location.search);

            [...params.keys()].forEach(key => {
                if (key.startsWith('filters[')) {
                    params.delete(key);
                }
            });

            const filterParam = `filters[${keyframeId}:colors]`;

            let queryString = params.toString();
            if (queryString.length > 0) {
                queryString += `&${filterParam}`;
            } else {
                queryString = filterParam;
            }

            window.location.href = `${baseUrl}?${queryString}`;
        } catch (err) {
            console.error("Failed to parse drag data:", err);
        }

    });
});