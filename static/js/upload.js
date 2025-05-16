// function updateFileName() {
//   const input = document.getElementById("researchPaper");
//   const display = document.getElementById("fileNameDisplay");
//   if (input.files.length > 0) {
//     display.textContent = `Selected file: ${input.files[0].name}`;
//     display.classList.add("file-attached");
//   } else {
//     display.textContent = "";
//     display.classList.remove("file-attached");
//   }
// }



function updateWordCount() {
  const maxWordLimit = 120;
  const textArea = document.getElementById('inputText');
  const wordCountDisplay = document.getElementById('wordCount');
  const limitMessage = document.getElementById('limitMessage');


  const text = textArea.value;
  const words = text.trim().split(/\s+/).filter(word => word.length > 0);
  const wordCount = words.length;


  wordCountDisplay.textContent = `${wordCount} words`;

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