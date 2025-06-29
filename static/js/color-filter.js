document.addEventListener("DOMContentLoaded", function () {
    const params = new URLSearchParams(window.location.search);
    const filtersToLoad = [
        ["colors", 3],
        ["embeddings", 3],
        ["objects", 1],
    ];

    for (const [filterType, count] of filtersToLoad) {
        for (let i = 1; i <= count; i++) {
            const idSuffix = filterType === "objects" ? "" : `-${i}`;
            const containerId = `${filterType.slice(0, -1)}-filter${idSuffix}`;
            const container = document.getElementById(containerId);
            if (!container) continue;

            const storageKey = `filterPreview_${filterType}_${i}`;
            const previewSrc = localStorage.getItem(storageKey);
            const hasFilter = [...params.keys()].some(k => k.startsWith("filters[") && k.includes(`:${filterType}`));
            const imgPreview = container.querySelector("img.preview-image");

            if (previewSrc && hasFilter && imgPreview) {
                imgPreview.src = previewSrc;
                imgPreview.style.display = "block";
            } else {
                localStorage.removeItem(storageKey);
            }
        }
    }
    const query = params.get("q");
    const entries = Array.from(params.entries());

    const resultContainer = document.getElementById("progressive-results");
    const loadMoreBtn = document.getElementById("load-more");
    const returnedKeyframes = new Set();
    let resultsFound = false;
    let stop = false;
    let fetchInProgress = false;

    const filterKeys = entries
        .filter(([key, _]) => key.startsWith("filters["))
        .map(([key]) => {
            const match = key.match(/^filters\[(.+)\]$/);
            return match ? match[1] : null;
        })
        .filter(Boolean);

    async function fetchNextResult() {
        if (!query || stop || fetchInProgress) return;
        fetchInProgress = true;

        const requestParams = new URLSearchParams();
        requestParams.append("q", query);

        [...returnedKeyframes].forEach(id =>
            requestParams.append("returned[]", id)
        );

        filterKeys.forEach(f =>
            requestParams.append("filters[]", f)
        );

        try {
            const response = await fetch(`/api/search/?${requestParams.toString()}`);
            const data = await response.json();

            if (data.done || !data.results || data.results.length === 0) {
                if (!resultsFound) {
                    resultContainer.innerHTML = `<p>No clips found matching "${query}" with the selected filters.</p>`;
                }
                stop = true;
                loadMoreBtn.disabled = true;
                loadMoreBtn.textContent = "No more results";
                return;
            }

            resultsFound = true;

            data.results.forEach(result => {
                returnedKeyframes.add(result.keyframe_id);

                const div = document.createElement("div");
                div.className = "clip-card preview-container-home";
                div.innerHTML = `
                    <a href="/detailed_view/${result.keyframe_id}?${window.location.search.substring(1)}" draggable="false">
                        <img src="${result.thumbnail}" alt="Keyframe" data-keyframe-id="${result.keyframe_id}" class="thumbnail draggable-image" draggable="true" />
                    </a>
                `;
                resultContainer.appendChild(div);
            });

        } catch (err) {
            console.error("Error fetching result:", err);
            stop = true;
        } finally {
            fetchInProgress = false;
        }
    }

    if (query) {
        loadMoreBtn.style.display = "none";
        fetchNextResult();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("search-form");

    form.addEventListener("submit", (e) => {
        e.preventDefault();

        const query = form.q.value.trim();
        const params = new URLSearchParams(window.location.search);

        if (!query) return;

        params.set("q", query);

        const baseUrl = window.location.origin + window.location.pathname;
        window.location.href = `${baseUrl}?${params.toString()}`;
    });
});