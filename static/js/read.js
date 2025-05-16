// // // Upload PDF
// // const uploadForm = document.getElementById("uploadForm");
// // const uploadStatus = document.getElementById("uploadStatus");
// // const questionArea = document.getElementById("questionArea");

// // uploadForm.addEventListener("submit", async (e) => {
// //     e.preventDefault();
// //     const formData = new FormData(uploadForm);

// //     uploadStatus.textContent = "Uploading PDF...";
// //     const response = await fetch("/upload", { method: "POST", body: formData });

// //     if (response.ok) {
// //         uploadStatus.textContent = "✅ PDF uploaded and processed!";
// //         questionArea.style.display = "block";
// //     } else {
// //         uploadStatus.textContent = "❌ Error uploading PDF.";
// //     }
// // });

// // // Ask Question
// // async function askQuestion() {
// //     const queryInput = document.getElementById("queryInput");
// //     const answerOutput = document.getElementById("answerOutput");

// //     const userQuery = queryInput.value.trim();
// //     if (!userQuery) return alert("Please enter a question!");

    
// // }


// let currentPDF = null;

// document.getElementById('uploadForm').addEventListener('submit', async function(e) {
//     e.preventDefault();
//     const fileInput = document.getElementById('pdfFile');
//     const uploadBtn = document.getElementById('uploadBtn');
//     const uploadStatus = document.getElementById('uploadStatus');
    
//     if (fileInput.files.length === 0) {
//         showAlert('Please select a PDF file first.');
//         return;
//     }
    
//     const file = fileInput.files[0];
//     if (file.type !== 'application/pdf') {
//         showAlert('Please upload a valid PDF file.');
//         return;
//     }
    
//     try {
//         // Upload PDF to server
//         uploadBtn.disabled = true;
//         uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Uploading...</span>';
//         uploadStatus.textContent = `Uploading ${file.name}...`;
        
//         const formData = new FormData();
//         formData.append('pdf', file);
        
//         const uploadResponse = await fetch('/upload', {
//             method: 'POST',
//             body: formData
//         });
        
//         if (!uploadResponse.ok) {
//             throw new Error('Upload failed');
//         }
        
//         currentPDF = file;
//         uploadStatus.textContent = `Successfully uploaded: ${file.name}`;
//         uploadBtn.innerHTML = '<i class="fas fa-check"></i><span>Uploaded</span>';
//         document.getElementById('input-area').classList.remove('hidden');
        
//         addMessage('user', `Uploaded PDF: ${file.name}`);
//         addMessage('bot', 'Great! Now you can ask questions about this research paper.');
        
//     } catch (error) {
//         console.error('Upload error:', error);
//         uploadStatus.textContent = 'Error uploading file. Please try again.';
//         uploadBtn.innerHTML = '<i class="fas fa-upload"></i><span>Try Again</span>';
//         uploadBtn.disabled = false;
//     }
// });

// async function askQuestion() {
//     const questionInput = document.getElementById('queryInput');
//     const question = questionInput.value.trim();
//     const askBtn = document.getElementById('askBtn');
    
//     if (!currentPDF) {
//         showAlert('Please upload a PDF file first.');
//         return;
//     }
    
//     if (!question) {
//         showAlert('Please enter a question.');
//         return;
//     }
    
//     // Add user question to chat
//     addMessage('user', question);
    
//     // Show typing indicator
//     const typingId = showTypingIndicator();
    
//     // Disable input during processing
//     questionInput.disabled = true;
//     askBtn.disabled = true;
//     askBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
    
//     try {
//         // Make API call to your backend
//         const response = await fetch("/ask", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ query: question }),
//         });

//         // Remove typing indicator
//         removeTypingIndicator(typingId);
        
//         if (response.ok) {
//             const data = await response.json();
//             addMessage('bot', data.answer);
//         } else {
//             throw new Error('API request failed');
//         }
//     } catch (error) {
//         console.error('Error:', error);
//         addMessage('bot', "❌ Error getting the answer. Please try again.");
//     } finally {
//         // Re-enable input
//         questionInput.disabled = false;
//         askBtn.disabled = false;
//         askBtn.innerHTML = '<i class="fas fa-paper-plane"></i><span>Ask</span>';
//         questionInput.value = '';
//         questionInput.focus();
//     }
// }

// // Helper functions (keep these the same as before)
// function addMessage(sender, text) {
//     const chatContainer = document.getElementById('chat-container');
//     const messageDiv = document.createElement('div');
//     messageDiv.className = `message ${sender}-message`;
    
//     const senderName = sender === 'user' ? 'You' : 'PDF Assistant';
//     const senderIcon = sender === 'user' ? 'fa-user' : 'fa-robot';
    
//     messageDiv.innerHTML = `
//         <div class="message-header">
//             <i class="fas ${senderIcon}"></i>
//             <span>${senderName}</span>
//         </div>
//         <p>${text}</p>
//     `;
    
//     chatContainer.appendChild(messageDiv);
//     chatContainer.scrollTop = chatContainer.scrollHeight;
// }

// function showTypingIndicator() {
//     const chatContainer = document.getElementById('chat-container');
//     const typingDiv = document.createElement('div');
//     typingDiv.className = 'message bot-message typing-indicator-container';
//     typingDiv.id = 'typing-' + Date.now();
    
//     typingDiv.innerHTML = `
//         <div class="message-header">
//             <i class="fas fa-robot"></i>
//             <span>PDF Assistant</span>
//         </div>
//         <div class="typing-indicator">
//             <div class="typing-dot"></div>
//             <div class="typing-dot"></div>
//             <div class="typing-dot"></div>
//         </div>
//     `;
    
//     chatContainer.appendChild(typingDiv);
//     chatContainer.scrollTop = chatContainer.scrollHeight;
//     return typingDiv.id;
// }

// function removeTypingIndicator(id) {
//     const typingElement = document.getElementById(id);
//     if (typingElement) {
//         typingElement.remove();
//     }
// }

// function showAlert(message) {
//     // Consider using a more elegant notification system in production
//     alert(message);
// }

// // Allow pressing Enter to submit question
// document.getElementById('queryInput').addEventListener('keypress', function(e) {
//     if (e.key === 'Enter') {
//         askQuestion();
//     }
// });


let currentPDF = null;

document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('pdfFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    
    if (fileInput.files.length === 0) {
        showAlert('Please select a PDF file first.');
        return;
    }
    
    const file = fileInput.files[0];
    if (file.type !== 'application/pdf') {
        showAlert('Please upload a valid PDF file.');
        return;
    }
    
    try {
        // Upload PDF to server
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Uploading...</span>';
        uploadStatus.textContent = `Uploading ${file.name}...`;
        
        const formData = new FormData();
        formData.append('pdf', file);
        
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            throw new Error('Upload failed');
        }
        
        currentPDF = file;
        uploadStatus.textContent = '';
        uploadBtn.innerHTML = '<i class="fas fa-check"></i><span>Uploaded</span>';
        document.getElementById('input-area').classList.remove('hidden');
        
        // Show file info in sidebar
        document.getElementById('fileInfoContainer').classList.remove('hidden');
        document.getElementById('fileName').textContent = file.name;
        
        // Add to chat
        addMessage('user', `Uploaded PDF: ${file.name}`);
        addMessage('bot', 'Great! Now you can ask questions about this research paper.');
        
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = 'Error uploading file. Please try again.';
        uploadBtn.innerHTML = '<i class="fas fa-upload"></i><span>Try Again</span>';
        uploadBtn.disabled = false;
    }
});

async function askQuestion() {
    const questionInput = document.getElementById('queryInput');
    const question = questionInput.value.trim();
    const askBtn = document.getElementById('askBtn');
    
    if (!currentPDF) {
        showAlert('Please upload a PDF file first.');
        return;
    }
    
    if (!question) {
        showAlert('Please enter a question.');
        return;
    }
    
    // Add user question to chat
    addMessage('user', question);
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    // Disable input during processing
    questionInput.disabled = true;
    askBtn.disabled = true;
    askBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
    
    try {
        // Make API call to your backend
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: question }),
        });

        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            const data = await response.json();
            addMessage('bot', data.answer);
        } else {
            throw new Error('API request failed');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('bot', "❌ Error getting the answer. Please try again.");
    } finally {
        // Re-enable input
        questionInput.disabled = false;
        askBtn.disabled = false;
        askBtn.innerHTML = '<i class="fas fa-paper-plane"></i><span>Ask</span>';
        questionInput.value = '';
        questionInput.focus();
    }
}

function newChat() {
    // Clear the current chat
    document.getElementById('chat-container').innerHTML = `
        <div class="message bot-message">
            <div class="message-header">
                <i class="fas fa-robot"></i>
                <span>PDF Assistant</span>
            </div>
            <p>Hello! Please upload a PDF research paper to get started. Once uploaded, you can ask questions about its content.</p>
        </div>
    `;
    
    // Reset the input area
    document.getElementById('input-area').classList.add('hidden');
    document.getElementById('queryInput').value = '';
    
    // Reset upload section
    document.getElementById('pdfFile').value = '';
    document.getElementById('uploadBtn').innerHTML = '<i class="fas fa-upload"></i><span>Upload PDF</span>';
    document.getElementById('uploadBtn').disabled = false;
    document.getElementById('uploadStatus').textContent = '';
    document.getElementById('fileInfoContainer').classList.add('hidden');
    
    currentPDF = null;
}

function addMessage(sender, text) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const senderName = sender === 'user' ? 'You' : 'PDF Assistant';
    const senderIcon = sender === 'user' ? 'fa-user' : 'fa-robot';
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <i class="fas ${senderIcon}"></i>
            <span>${senderName}</span>
        </div>
        <p>${text}</p>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showTypingIndicator() {
    const chatContainer = document.getElementById('chat-container');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator-container';
    typingDiv.id = 'typing-' + Date.now();
    
    typingDiv.innerHTML = `
        <div class="message-header">
            <i class="fas fa-robot"></i>
            <span>PDF Assistant</span>
        </div>
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return typingDiv.id;
}

function removeTypingIndicator(id) {
    const typingElement = document.getElementById(id);
    if (typingElement) {
        typingElement.remove();
    }
}

function showAlert(message) {
    alert(message);
}

// Allow pressing Enter to submit question
document.getElementById('queryInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion();
    }
});