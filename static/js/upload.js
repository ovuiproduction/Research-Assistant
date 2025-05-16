function updateFileName() {
    const input = document.getElementById("researchPaper");
    const display = document.getElementById("fileNameDisplay");
    if (input.files.length > 0) {
      display.textContent = `Selected file: ${input.files[0].name}`;
      display.classList.add("file-attached");
    } else {
      display.textContent = "";
      display.classList.remove("file-attached");
    }
  }

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
  
  
  function updateWordCount() {
    const maxWordLimit = 120; 
    const textArea = document.getElementById('inputText');
    const wordCountDisplay = document.getElementById('wordCount');
    const limitMessage = document.getElementById('limitMessage');
    
   
    const text = textArea.value;
    const words = text.trim().split(/\s+/).filter(word => word.length > 0);
    const wordCount = words.length;

    
    wordCountDisplay.textContent = `${wordCount}/${maxWordLimit} words`;

    
    // if (wordCount > maxWordLimit) {
    //     limitMessage.style.visibility = 'visible'; 
    // } else {
    //     limitMessage.style.visibility = 'hidden'; 
    // }
}


function validateWordCount() {
  const textArea = document.getElementById("inputText");
  const text = textArea.value.trim();
  const words = text.split(/\s+/).filter(word => word.length > 0);

  if (words.length > 120) {
    // alert("Word limit exceeded. Please reduce the word count to 120 or fewer.");
    return true;
  }
  activeLoader()
  return true;
}