function getAverageColor(img) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    /*canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);*/
    canvas.width = 16;
    canvas.height = 16;
    ctx.drawImage(img, 0, 0, 16, 16);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    let r = 0, g = 0, b = 0, count = 0;
    for (let i = 0; i < imageData.length; i += 4) {
        r += imageData[i];
        g += imageData[i + 1];
        b += imageData[i + 2];
        count++;
    }

    return count > 0 ? {
        r: r / count,
        g: g / count,
        b: b / count
    } : { r: 0, g: 0, b: 0 };
}

function colorDistance(c1, c2) {
    return Math.sqrt(
        Math.pow(c1.r - c2.r, 2) +
        Math.pow(c1.g - c2.g, 2) +
        Math.pow(c1.b - c2.b, 2)
    );
}

function filterClipsByColor(referenceColor) {
    const threshold = 50;

    document.querySelectorAll(".clip-card").forEach(card => {
        const img = card.querySelector(".thumbnail");

        if (img.complete) {
            const color = getAverageColor(img);
            const dist = colorDistance(referenceColor, color);
            card.style.display = dist <= threshold ? "block" : "none";
        }
    });
}

document.querySelectorAll(".draggable-image").forEach(img => {
    img.addEventListener("dragstart", e => {
        e.dataTransfer.setData("text/plain", e.target.src);
    });
});

document.querySelectorAll(".drag-drop-area").forEach(area => {
    const imgPreview = area.querySelector("img");

    area.addEventListener("click", () => {
        document.querySelectorAll(".clip-card").forEach(card => {
            card.style.display = "block";
        });

        const imgPreview = area.querySelector("img");
        if (imgPreview) {
            imgPreview.src = "";
            imgPreview.style.display = "none";
        }
    });

    area.addEventListener("dragover", e => {
        e.preventDefault();
        area.classList.add("dragging");
    });

    area.addEventListener("dragleave", () => {
        area.classList.remove("dragging");
    });

    area.addEventListener("drop", e => {
        e.preventDefault();
        area.classList.remove("dragging");

        const imgSrc = e.dataTransfer.getData("text/plain");
        imgPreview.src = imgSrc;
        imgPreview.style.display = "block";

        const tempImg = new Image();
        tempImg.crossOrigin = "anonymous";
        tempImg.src = imgSrc;
        tempImg.onload = () => {
            const avgColor = getAverageColor(tempImg);
            filterClipsByColor(avgColor);
        };
    });
});