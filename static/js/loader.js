const messages = [
  "Analyzing the document...",
  "Extracting key insights...",
  "Summarizing important points...",
  "Please wait, processing in progress...",
  "Optimizing the document for readability...",
];

let currentMessageIndex = 0;
let messageInterval; // Store interval reference

function rotateMessages() {
  const messageBlock = document.getElementById("loading_message");
  messageBlock.classList.remove("message_active");

  // Ensure message change after fade-out animation
  setTimeout(() => {
    messageBlock.innerHTML = messages[currentMessageIndex];
    messageBlock.classList.add("message_active");
    currentMessageIndex = (currentMessageIndex + 1) % messages.length;
  }, 200);
}

function activeLoader() {
  const loader = document.getElementById("loader_div");
  loader.classList.add("loader_active");

  // Start message rotation only if not already running
  if (!messageInterval) {
    rotateMessages(); // Start immediately
    messageInterval = setInterval(rotateMessages, 5000);
  }
}

function deactivateLoader() {
  const loader = document.getElementById("loader_div");
  const messageBlock = document.getElementById("loading_message");

  loader.classList.remove("loader_active");
  messageBlock.classList.remove("message_active");

  // Clear message and stop rotation
  messageBlock.innerHTML = "";
  clearInterval(messageInterval);
  messageInterval = null; // Reset interval reference
}

// Ensure loader resets when navigating back
window.addEventListener("pageshow", function (event) {
  if (event.persisted) {
    deactivateLoader();
  }
});


document.getElementById("form_submit").addEventListener("submit",(()=>{startLoader()}));

function startLoader(){
    activeLoader();
}