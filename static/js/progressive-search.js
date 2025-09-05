document.addEventListener("DOMContentLoaded", function () {
    const query = new URLSearchParams(window.location.search).get("q");
    const resultContainer = document.getElementById("progressive-results");
    const loadMoreBtn = document.getElementById("load-more");
    const returnedKeyframes = new Set();
    let resultsFound = false;
    let stop = false;
    let fetchInProgress = false;
    const urlParams = new URLSearchParams(window.location.search);
    let currentSearchMode = urlParams.get("mode") || "balanced";

    const searchModeButtons = document.querySelectorAll(".search-mode-btn");
    
    // Set active button based on current mode
    searchModeButtons.forEach(btn => {
        if (btn.dataset.mode === currentSearchMode) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });
    
    searchModeButtons.forEach(btn => {
        btn.addEventListener("click", function() {
            searchModeButtons.forEach(b => b.classList.remove("active"));
            this.classList.add("active");
            currentSearchMode = this.dataset.mode;
            
            const params = new URLSearchParams(window.location.search);
            params.set("mode", currentSearchMode);
            const newUrl = `${window.location.pathname}?${params.toString()}`;
            window.history.pushState({}, "", newUrl);
            
            resetSearch();
        });
    });

    function resetSearch() {
        resultContainer.innerHTML = "";
        returnedKeyframes.clear();
        resultsFound = false;
        stop = false;
        fetchInProgress = false;
        
        if (query) {
            fetchNextResult();
        }
    }

    async function fetchNextResult() {
        if (!query || stop || fetchInProgress) return;
        fetchInProgress = true;

        const params = new URLSearchParams();
        params.append("q", query);
        params.append("mode", currentSearchMode);
        
        [...returnedKeyframes].forEach(id => params.append("returned[]", id));

        const urlParams = new URLSearchParams(window.location.search);
        const filterKeys = Array.from(urlParams.entries())
            .filter(([key, _]) => key.startsWith("filters["))
            .map(([key]) => {
                const match = key.match(/^filters\[(.+)\]$/);
                return match ? match[1] : null;
            })
            .filter(Boolean);
        
        filterKeys.forEach(f => params.append("filters[]", f));

        try {
            const response = await fetch(`/api/search/?${params.toString()}`);
            const data = await response.json();

            if (!data.results || data.results.length === 0) {
                if (!resultsFound) {
                    resultContainer.innerHTML = `<p>No clips found matching "${query}" in ${currentSearchMode} mode.</p>`;
                }
                stop = true;
                fetchInProgress = false;
                return;
            }

            resultsFound = true;
            data.results.forEach(result => {
                returnedKeyframes.add(result.keyframe_id);

                const div = document.createElement("div");
                div.className = "clip-card preview-container-home";
                const detailParams = new URLSearchParams();
                detailParams.set("q", query);
                detailParams.set("mode", currentSearchMode);
                
                div.innerHTML = `
                    <a href="/detailed_view/${result.keyframe_id}?${detailParams.toString()}" draggable="false">
                        <img src="${result.thumbnail}" alt="Keyframe" data-keyframe-id="${result.keyframe_id}" class="thumbnail draggable-image" draggable="true" />
                    </a>
                `;
                resultContainer.appendChild(div);
            });

            fetchInProgress = false;
            
            if (data.done) {
                stop = true;
            }

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
    
    // Form submission handler
    const form = document.getElementById("search-form");
    if (form) {
        form.addEventListener("submit", (e) => {
            e.preventDefault();

            const query = form.q.value.trim();
            const params = new URLSearchParams(window.location.search);

            if (!query) return;

            params.set("q", query);
            params.set("mode", currentSearchMode);

            const baseUrl = window.location.origin + window.location.pathname;
            window.location.href = `${baseUrl}?${params.toString()}`;
        });
    }
});