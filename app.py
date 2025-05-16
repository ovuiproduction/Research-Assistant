from flask import Flask, render_template, request, jsonify , redirect, url_for, session
import pdfplumber
import re
import os
from dotenv import load_dotenv
from flask_cors import CORS
import google.generativeai as genai
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

from textsimilarity import getScore
from codebase.humanize import humanize_text

app = Flask(__name__, template_folder='templates', static_folder='static')


load_dotenv()
app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = API_KEY)
geminimodel = genai.GenerativeModel("gemini-1.5-flash")

app.secret_key = os.urandom(24)

# Initialize Embedding Model and FAISS
model_rag = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
index = faiss.IndexFlatIP(embedding_dim) 
documents = []  # Stores text chunks

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


abstract_index = faiss.read_index("static/vectordatabase/arxiv_abstract_index.faiss")
title_index = faiss.read_index("static/vectordatabase/arxiv_title_index.faiss")
author_index = faiss.read_index("static/vectordatabase/arxiv_authors_index.faiss")
arxiv_10000_metadata = pd.read_csv("static/vectordatabase/arxiv_10000_metadata.csv")


journal_index = faiss.read_index("static/vectordatabase/journal_domain_index.faiss")
journal_metadata = pd.read_csv("static/vectordatabase/journal_metadata.csv")



###### Functions 


#### Paper search functions

def search_papers_by_abstract(query, top_k):
    query_embedding = model_rag.encode([query]).astype('float32')
    D, I = abstract_index.search(query_embedding, top_k)
    results = arxiv_10000_metadata.iloc[I[0]]
    return results[['id', 'title', 'abstract', 'authors','doi']]

def search_papers_by_title(query, top_k):
    query_embedding = model_rag.encode([query]).astype('float32')
    D, I = title_index.search(query_embedding, top_k)
    results = arxiv_10000_metadata.iloc[I[0]]
    return results[['id', 'title', 'abstract', 'authors','doi']]

def search_papers_by_author(query, top_k):
    query_embedding = model_rag.encode([query]).astype('float32')
    D, I = author_index.search(query_embedding, top_k)
    results = arxiv_10000_metadata.iloc[I[0]]
    return results[['id', 'title', 'abstract', 'authors','doi']]



#### Journal search function

def search_journals(query, top_k, distance_threshold=0.8):
    query_embedding = model_rag.encode([query]).astype('float32')
    D, I = journal_index.search(query_embedding, top_k)

    filtered_rows = []

    for i, dist in zip(I[0], D[0]):
        if i == -1:
            continue
        if dist <= distance_threshold:
            filtered_rows.append(journal_metadata.iloc[i])

    # Case 1: Filtered results found within distance threshold
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
    else:
        # Case 2: No results within threshold — return top_k fallback
        filtered_df = journal_metadata.iloc[I[0]].copy()

    # Select required columns
    return filtered_df[['Rank', 'OA', 'Title', 'Best Quartile', 'Country', 'CiteScore', 'H-index', 'Best Subject Area']]



###### functions for rag

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Chunking Text
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Store Chunks in FAISS
def store_chunks_in_faiss(chunks):
    global index, documents
    embeddings = model_rag.encode(chunks, convert_to_numpy=True)
    index.reset()
    # Normalize embeddings for inner product similarity
    faiss.normalize_L2(embeddings)

    # Add to FAISS index
    index.add(embeddings)
    documents = chunks

# Search in FAISS Index
def search_faiss(query, top_k=5):
    query_embedding = model_rag.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity

    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [documents[i] for i in indices[0]]
    return retrieved_chunks

# Query Gemini Flash 1.5
def query_gemini(context, user_query):
    prompt = f"Context: {context}\n\nUser Query: {user_query}\n\nAnswer:"
    response = geminimodel.generate_content(prompt)
    return response.text



###### Routes

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')



##### AI Content Removal Route

@app.route('/upload',methods=['GET'])
def upload():
    return render_template('upload_text.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    print("Request Arrived")
    # author = request.form['authorName']
    # paperTitle = request.form['paperTitle']
    # model_type = request.form['modelType']
    author = "Aditya"
    paperTitle = "Research Paper"
    text = request.form['inputText']
    model_type = "fine_tuned_1000"
    
    qualityIndex = int(request.form['qualityIndex'])
    max_limit = int(request.form['max_limit'])
    min_limit = int(request.form['min_limit'])
    
    model_version="Model-Pro-Max"
  
    print("\nOriginal Text\n")
    print(text)
    
    generated_summary = humanize_text(text,qualityIndex,max_limit,min_limit,model_type)
    if generated_summary == "Error":
        print("\nGenerated Summary error")
        return jsonify({"error":"Error in humanizing given text"}) ,400

    print(generated_summary)
    
    session.pop('summary_data', None)
    
    session['summary_data'] = {
        'original_text': text,
        'summary': generated_summary,
        'author': author,
        'paperTitle': paperTitle,
        'model_type': model_version,
        'qualityIndex': qualityIndex,
        'max_limit': max_limit,
        'min_limit': min_limit
    }
     
    return redirect(url_for('result'))

@app.route('/result', methods=['GET'])
def result():
    summary_data = session.get('summary_data', None) 
    if not summary_data:
        return redirect(url_for('home'))
    return render_template('fetch.html', **summary_data)

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    data = request.get_json()
    original_text = data.get('original_text', '')
    summary_text = data.get('summary_text', '')

    score = getScore(original_text, summary_text)
    return jsonify({"score": round(score * 100, 2)})



### Read With AI

@app.route("/read-with-ai")
def readAI():
    return render_template("read.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"error": "No selected PDF"}), 400

    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text and create FAISS index
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    store_chunks_in_faiss(chunks)

    return jsonify({"message": "PDF uploaded and indexed successfully!"})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Retrieve chunks and query LLM
    retrieved_chunks = search_faiss(user_query)
    answer = query_gemini("\n".join(retrieved_chunks), user_query)
    answer = re.sub(r"\*", "", answer)
    return jsonify({"query": user_query, "answer": answer})



### Search Relavant research papers


@app.route('/search-papers',methods=['GET'])
def searchPaper():
    return render_template('search.html')

@app.route('/search-papers', methods=['POST'])
def search_papers():
    data = request.json
    search_type = data.get("searchType")
    query = data.get("searchQuery")
    search_limit = int(data.get("search_limit"))

    no_of_papers = search_limit

    if search_type == "title":
        results = search_papers_by_title(query,no_of_papers)
    elif search_type == "abstract":
        results = search_papers_by_abstract(query,no_of_papers)
    elif search_type == "author":
        results = search_papers_by_author(query,no_of_papers)
    else:
        return jsonify({"error": "Invalid search type"}), 400
    results = results.fillna("")
    return jsonify(results.to_dict(orient="records"))



### Search Journal


@app.route('/search-journal', methods=['GET'])
def journalSearch():
    return render_template('journal_search.html')


@app.route('/search-journal', methods=['POST'])
def searchjournal():
    data = request.get_json()
    keywords = data.get('keywords', [])
    quartile = data.get('Quartile', None)
    search_limit = int(data.get("search_limit"))

    if not keywords:
        subjects = ""
    elif len(keywords) == 1:
        subjects = keywords[0]
    else:
        subjects = ', '.join(keywords[:-1]) + ' and ' + keywords[-1]

    # Build query with quartile if provided
    if quartile:
        query = f"Journal with {quartile} quartile in {subjects}"
    else:
        query = subjects  # fallback to just keywords if no quartile

    # Search journals
    results = search_journals(query,search_limit)
    results = results.fillna("")

    # Convert results to JSON
    results_json = results.to_dict(orient='records')
    return jsonify(results_json)



if __name__ == '__main__':
    app.run(debug=True,use_reloader=False,port=3000)
