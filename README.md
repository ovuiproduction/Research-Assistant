# 🤖 AI-ResearchMate: Humanizing and Assisting Research with LLMs ®

An AI-powered co-research assistant that simplifies academic research using intelligent search, interactive Q&A, and humanized text transformation to help bypass AI content detection mechanisms.

---

## 🚀 Objective

**AI-ResearchMate** is designed to:
- Assist researchers in finding high-quality academic content.
- Answer research questions using a RAG (Retrieval-Augmented Generation) pipeline.
- Humanize AI-generated content to avoid AI detection.
- Identify and highlight AI-written sections in research drafts and rewrite them in a more human tone.

---

## 🧠 Features

- 🔍 **Smart Semantic Search**: Retrieve top relevant papers using embeddings from ArXiv and Journal Ranking datasets.
- 💬 **Interactive Q&A**: Ask questions about research content and get answers powered by Large Language Models.
- 📝 **AI Content Humanization**: Use BART-based pipeline to rewrite AI-generated text into human-like writing.
- 🚨 **AI Content Detection**: Detect and highlight AI-written content to improve authenticity and originality.

---

## 🧰 Tech Stack

- **Language**: Python
- **LLM Models**: BART (`facebook/bart-large` via HuggingFace)
- **Vector Search**: FAISS
- **Database**: MongoDB
- **Datasets**: 
  - ArXiv Paper Dataset
  - Journal Ranking Dataset

---

## 🛠️ Key Contributions

- 🔧 Developed a RAG-based **research & journal retrieval system** using document embeddings stored in FAISS.
- ✍️ Engineered a **BART-powered content humanization pipeline** that rewrites detected AI content to human tone.
- 🧪 Created an **AI content detector** to flag AI-written sections and replace them with more natural writing.

---

## 📦 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/AI-ResearchMate.git
   cd AI-ResearchMate
```
