
const search_btn = document.getElementById("search_btn").addEventListener("click", (() => { searchPapers() }));

const reset_btn = document.getElementById("reset_btn").addEventListener("click",(()=>(resetAll())));

function resetAll(){
    document.getElementById("searchType").value = "title";
    document.getElementById("searchQuery").value = "";
    document.getElementById("search_limit").value = "";
    document.querySelector("#resultsTable tbody").innerHTML = "";
}

function searchPapers() {
    let searchType = document.getElementById("searchType").value;
    let searchQuery = document.getElementById("searchQuery").value;
    let search_limit = document.getElementById("search_limit").value;

    showSpinner();

    fetch('/search-papers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ searchType, searchQuery, search_limit })
    })
        .then(response => response.json())
        .then(data => {
            let tableBody = document.querySelector("#resultsTable tbody");
            tableBody.innerHTML = "";
            hideSpinner();
            if (data.length === 0) {
                tableBody.innerHTML = "<tr><td colspan='5'>No results found.</td></tr>";
                return;
            }

            data.forEach(row => {
                let tr = document.createElement("tr");
                tr.innerHTML = `
                        <td>${row.id}</td>
                        <td>${row.title}</td>
                        <td>${row.abstract}</td>
                        <td>${row.authors}</td>
                        <td><a href="https://doi.org/${row.doi}" target="_blank">${row.doi}</a></td>
                    `;
                tableBody.appendChild(tr);
            });

        })
        .catch(error => { hideSpinner(); alert("Error fetching results:", error); });
}


function showSpinner() {
    document.getElementById("loading-spinner").style.display = "flex";
}

function hideSpinner() {
    document.getElementById("loading-spinner").style.display = "none";
}