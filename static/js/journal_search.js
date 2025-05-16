
const keywords = [];
let allResults = [];

function displayResults(data) {
    allResults = data;
    filterAndRenderResults();
}

function filterAndRenderResults() {
    const quartile = document.getElementById("quartile-filter").value;
    const resultsBody = document.getElementById("results-body");

    // Filter by quartile if selected, else show all
    const filteredResults = quartile
        ? allResults.filter(journal => journal["Best Quartile"] === quartile)
        : allResults;

    resultsBody.innerHTML = "";

    if (filteredResults.length === 0) {
        resultsBody.innerHTML = "<tr><td colspan='8'>No journals found.</td></tr>";
        return;
    }

    filteredResults.forEach(journal => {
        const row = document.createElement("tr");
        row.innerHTML = `
                    <td>${journal.Title || ""}</td>
                    <td>${journal.Rank || ""}</td>
                    <td>${journal.OA || ""}</td>
                    <td>${journal["Best Quartile"] || ""}</td>
                    <td>${journal.Country || ""}</td>
                    <td>${journal.CiteScore || ""}</td>
                    <td>${journal["H-index"] || ""}</td>
                    <td>${journal["Best Subject Area"] || ""}</td>
                `;
        resultsBody.appendChild(row);
    });
}

document.getElementById("keyword-input").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        const val = this.value.trim();
        if (val === "") return;

        // Prevent duplicate keywords
        if (!keywords.includes(val)) {
            keywords.push(val);
            displayKeywords();
        }
        this.value = "";
    }
});

function displayKeywords() {
    const display = document.getElementById("keywords-display");
    display.innerHTML = "";
    keywords.forEach((keyword, index) => {
        const span = document.createElement("span");
        span.textContent = keyword + " ×";
        span.className = "keyword-item";
        span.title = "Click to remove";
        span.onclick = () => removeKeyword(index);
        display.appendChild(span);
    });
}

function removeKeyword(index) {
    keywords.splice(index, 1);
    displayKeywords();
}

document.getElementById("search-btn").addEventListener("click", function () {
    const quartile = document.getElementById("quartile-filter").value;
    const search_limit = document.getElementById("search_limit").value;
    if (keywords.length === 0) {
        alert("Please enter at least one keyword.");
        return;
    }
    if (!quartile) {
        alert("Please select a quartile.");
        return;
    }

    showSpinner();

    fetch("/search-journal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keywords: keywords, Quartile: quartile, search_limit: search_limit })
    })
        .then(response => {
            if (!response.ok) throw new Error("Network response was not ok");
            return response.json();
        })
        .then(data => { displayResults(data); hideSpinner(); })
        .catch(err => { alert("Error fetching results: " + err.message); hideSpinner(); });
});

document.getElementById("reset-btn").addEventListener("click", function () {
    keywords.length = 0;
    displayKeywords();
    document.getElementById("quartile-filter").value = ""; // Reset quartile filter
    document.getElementById("results-body").innerHTML = ""; // Clear results table
});

function showSpinner() {
    document.getElementById("loading-spinner").style.display = "flex";
}

function hideSpinner() {
    document.getElementById("loading-spinner").style.display = "none";
}