document.querySelectorAll(".draggable-image").forEach(img => {
    img.addEventListener("dragstart", e => {
        e.dataTransfer.setData("text/plain", e.target.src);
    });
});

document.querySelectorAll(".drag-drop-area").forEach(area => {
    const imgPreview = area.querySelector("img");

    area.addEventListener("click", () => {
        const imgPreview = area.querySelector("img");
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

        const imgSrc = e.dataTransfer.getData("text/plain");
        imgPreview.src = imgSrc;
        imgPreview.style.display = "block";

        const params = new URLSearchParams(window.location.search);
        params.set("cfilter", encodeURIComponent(imgSrc));
        window.location.search = params.toString();

        const tempImg = new Image();
        tempImg.crossOrigin = "anonymous";
        tempImg.src = imgSrc;
    });
});