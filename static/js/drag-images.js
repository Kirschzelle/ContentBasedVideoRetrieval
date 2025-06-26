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
    area.addEventListener("click", () => {
        const imgPreview = area.querySelector("img.preview-image");
        if (imgPreview) {
            imgPreview.src = "";
            imgPreview.style.display = "none";
        }

        const filterType = area.id.replace("-filter", ""); // e.g., "color", "embedding", "object"
        const storageKey = `filterPreview_${filterType}`;
        localStorage.removeItem(storageKey);

        const params = new URLSearchParams(window.location.search);
        let changed = false;
        for (const key of [...params.keys()]) {
            if (key.startsWith("filters[") && key.includes(`:${filterType}`)) {
                params.delete(key);
                changed = true;
            }
        }

        if (changed) {
            const baseUrl = window.location.origin + window.location.pathname;
            const queryString = params.toString();
            window.location.href = queryString.length > 0 ? `${baseUrl}?${queryString}` : baseUrl;
        }
    });

    area.addEventListener("dragover", e => e.preventDefault());

    area.addEventListener("drop", e => {
        e.preventDefault();

        try {
            const data = JSON.parse(e.dataTransfer.getData("text/plain"));
            const keyframeId = data.keyframeId;
            const imgPreview = area.querySelector("img.preview-image");
            const filterMap = {
                color: "colors",
                embedding: "embeddings",
                object: "objects"
            };

            const rawType = area.id.replace("-filter", "");
            const filterType = filterMap[rawType];
            const filterKey = `filters[${keyframeId}:${filterType}]`;
            const storageKey = `filterPreview_${filterType}`;

            if (imgPreview && data.src) {
                imgPreview.src = data.src;
                imgPreview.style.display = "block";
            }

            localStorage.setItem(storageKey, data.src);

            const params = new URLSearchParams(window.location.search);
            [...params.keys()].forEach(key => {
                if (key.startsWith('filters[') && key.includes(`:${filterType}`)) {
                    params.delete(key);
                }
            });

            for (const key of [...params.keys()]) {
                if (key.startsWith("filters[") && key.includes(`:${filterType}`)) {
                    params.delete(key);
                }
            }

            const queryParts = [];

            for (const [key, value] of params.entries()) {
                queryParts.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
            }

            queryParts.push(`${filterKey}=`);

            const baseUrl = window.location.origin + window.location.pathname;
            const queryString = queryParts.join("&");

            window.location.href = `${baseUrl}?${queryString}`;
        } catch (err) {
            console.error("Failed to parse drag data:", err);
        }
    });
});