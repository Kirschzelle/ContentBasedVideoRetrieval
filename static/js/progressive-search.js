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
            div.className = "clip-card preview-container-home";
            div.innerHTML = `
            <a href="/detailed_view/${data.keyframe_id}?q=${encodeURIComponent(query)}">
                <img src="${data.thumbnail}" alt="Keyframe" class="thumbnail draggable-image" draggable="true" />
            </a>
            `;

            resultContainer.appendChild(div);

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