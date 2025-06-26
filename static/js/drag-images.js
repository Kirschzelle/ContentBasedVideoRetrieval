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
            const params = new URLSearchParams(window.location.search);
            params.set("cfilter", encodeURIComponent(parseInt(data.keyframeId)));
            window.location.search = params.toString();
        } catch (err) {
            console.error("Failed to parse drag data:", err);
        }
    });
});