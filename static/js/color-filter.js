document.addEventListener("DOMContentLoaded", function () {
    const params = new URLSearchParams(window.location.search);
    const query = new URLSearchParams(window.location.search).get("q");
    const entries = Array.from(params.entries());
    const resultContainer = document.getElementById("progressive-results");
    const loadMoreBtn = document.getElementById("load-more");
    const returnedKeyframes = new Set();
    let resultsFound = false;
    let stop = false;
    const [key, value] = entries[1] || [];
    let fetchInProgress = false;

    async function fetchNextResult() {
        if (!key || stop || fetchInProgress) return;
        fetchInProgress = true;

        const params = new URLSearchParams();
        params.append("q", query);
        [...returnedKeyframes].forEach(id => params.append("returned[]", id));

        if (key && key.startsWith("filters[")) {
            const match = key.match(/^filters\[(.+)\]$/);
            if (match) {
                params.append("filters[]", match[1]);
            }
        }

        try {
            const response = await fetch(`/api/color/?${params.toString()}`);
            const data = await response.json();

            if (data.done || !data.keyframe_id) {
                if (!resultsFound) {
                    resultContainer.innerHTML = `<p>No clips found matching "${query}" with a color filter.</p>`;
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
            <a href="/detailed_view/${data.keyframe_id}?q=${encodeURIComponent(query)}" draggable="false">
                <img src="${data.thumbnail}" alt="Keyframe" data-keyframe-id="${data.keyframe_id}" class="thumbnail draggable-image" draggable="true" />
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

    if (key) {
        loadMoreBtn.style.display = "none";
        fetchNextResult();
    }
});