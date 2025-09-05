document.addEventListener("DOMContentLoaded", function () {
    const query = new URLSearchParams(window.location.search).get("q");
    const resultContainer = document.getElementById("progressive-results");
    const loadMoreBtn = document.getElementById("load-more");
    const returnedKeyframes = new Set();
    let resultsFound = false;
    let stop = false;
    let fetchInProgress = false;
    let currentSearchMode = "balanced";

    const searchModeButtons = document.querySelectorAll(".search-mode-btn");
    
    searchModeButtons.forEach(btn => {
        btn.addEventListener("click", function() {
            searchModeButtons.forEach(b => b.classList.remove("active"));
            this.classList.add("active");
            currentSearchMode = this.dataset.mode;
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

        const filtersRaw = new URLSearchParams(window.location.search).get("filters");
        if (filtersRaw) {
            filtersRaw.split(",").forEach(pair => params.append("filters[]", pair));
        }

        try {
            const response = await fetch(`/api/search/?${params.toString()}`);
            const data = await response.json();

            if (data.done || !data.results || data.results.length === 0) {
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
                div.innerHTML = `
                    <a href="/detailed_view/${result.keyframe_id}?q=${encodeURIComponent(query)}" draggable="false">
                        <img src="${result.thumbnail}" alt="Keyframe" data-keyframe-id="${result.keyframe_id}" class="thumbnail draggable-image" draggable="true" />
                    </a>
                `;
                resultContainer.appendChild(div);
            });

            fetchInProgress = false;
            
            if (!data.done && !stop) {
                setTimeout(() => {
                    if (!stop && !fetchInProgress) {
                        fetchNextResult();
                    }
                }, 200);
            }

        } catch (err) {
            console.error("Error fetching result:", err);
            stop = true;
            fetchInProgress = false;
        }
    }

    function handleScroll() {
        if (stop || fetchInProgress) return;
        
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        
        if (scrollTop + windowHeight >= documentHeight - 500) {
            fetchNextResult();
        }
    }

    if (query) {
        loadMoreBtn.style.display = "none";
        fetchNextResult();
        
        window.addEventListener('scroll', handleScroll, { passive: true });
    }
});