document.getElementById('download_pdf').addEventListener('click', function() {
    // Get content for the PDF
    const author = document.getElementById('authorName').textContent;
    const title = document.getElementById('paperTitle').textContent;
    const modelType = document.getElementById('model_type').textContent;
    let summary = document.getElementById('summaryText').textContent;
    summary = summary.trim();

    // Create a new jsPDF instance
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // Set margins and page size
    const margin = 10;
    const pageWidth = doc.internal.pageSize.width;
    const maxWidth = pageWidth - 2 * margin;  // Max width for the text

    // Add Author, Title, and Model Type to PDF
    doc.setFontSize(14);
    doc.text(`Author: ${author}`, margin, 20);
    doc.text(`Title: ${title}`, margin, 30);
    doc.text(`Model Used: ${modelType}`, margin, 40);

    // Add Summary Title
    doc.setFontSize(12);
    doc.text("Summary:", margin, 50);

    // Add Summary Text with word wrapping
    doc.setFontSize(10);
    doc.text(summary, margin, 60, { maxWidth: maxWidth, align: 'left' });

    // Save the PDF
    doc.save('summary.pdf');
});

document.addEventListener("DOMContentLoaded", function () {
    const summaryTextElement = document.getElementById("summaryText");
    const originalTextElement = document.getElementById("originalText");

    if (summaryTextElement) {
        const summary = summaryTextElement.innerText;
        updateWordCount(summary,'wordCount');
        updateWordCount(summary,'wordCountSummary');
    }
    if (summaryTextElement) {
        const originalText = originalTextElement.innerText;
        updateWordCount(originalText,'wordCountOriginal');
    }
});

function updateWordCount(summary,id) {
    console.log("Update word count");
    const wordCountDisplay = document.getElementById(id);
    const words = summary.trim().split(/\s+/).filter(word => word.length > 0);
    const wordCount = words.length;
    console.log(wordCount);
    wordCountDisplay.textContent = `${wordCount} words`;
}

document.getElementById("copy_btn").addEventListener("click", function () {
    const summaryText = document.getElementById("summaryText").textContent;
    navigator.clipboard.writeText(summaryText).then(() => {
      const copyBtn = document.getElementById("copy_btn");
      copyBtn.textContent = "✔ Copied!";
      setTimeout(() => {
        copyBtn.textContent = "📋";
      }, 2000);
    });
  });