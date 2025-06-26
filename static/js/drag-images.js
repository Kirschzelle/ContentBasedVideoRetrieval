document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("progressive-results").addEventListener("dragstart", (e) => {
        const img = e.target;
        if (img.classList.contains("draggable-image")) {
            const payload = {
                src: img.src,
                keyframeId: img.dataset.keyframeId,
            };
            console.log("dragstart payload set:", payload);
            e.dataTransfer.setData("text/plain", JSON.stringify(payload));
        }
    });

    const filterMap = {
        color: "colors",
        embedding: "embeddings",
        object: "objects"
    };
    
    document.querySelectorAll(".drag-drop-area").forEach(area => {
        const idMatch = area.id.match(/^([a-z]+)-filter(?:-(\d+))?$/);
        console.log("ID:", area.id, "Match:", idMatch);
        if (!idMatch) return;

        const rawType = idMatch[1];
        const index = idMatch[2] || "1";
        const filterType = filterMap[rawType];
        const storageKey = `filterPreview_${filterType}_${index}`;

        area.addEventListener("click", () => {
            const imgPreview = area.querySelector("img.preview-image");
            if (imgPreview) {
                imgPreview.src = "";
                imgPreview.style.display = "none";
            }

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
            const raw = e.dataTransfer.getData("text/plain");

            try {
                const data = JSON.parse(raw);
                const keyframeId = data.keyframeId;
                const imgPreview = area.querySelector("img.preview-image");
                const filterKey = `filters[${keyframeId}:${filterType}]`;

                if (imgPreview && data.src) {
                    imgPreview.src = data.src;
                    imgPreview.style.display = "block";
                }

                localStorage.setItem(storageKey, data.src);
                area.dataset.keyframeId = keyframeId;

                const params = new URLSearchParams(window.location.search);

                // Clean out any previous filters of the same type
                for (const key of [...params.keys()]) {
                    if (key.startsWith("filters[") && key.includes(`:${filterType}`)) {
                        params.delete(key);
                    }
                }

                params.set(filterKey, "");

                const baseUrl = window.location.origin + window.location.pathname;
                window.location.href = `${baseUrl}?${params.toString()}`;
            } catch (err) {
                console.error("Failed to parse drag data:", err);
            }
        });
    });
});