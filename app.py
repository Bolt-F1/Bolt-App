from flask import Flask, render_template, request, redirect, session
import sqlite3
from datetime import datetime, timedelta
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import os

import subprocess
import sys
import ssl

# ------------------- 1. Install NLTK if missing -------------------
try:
    import nltk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

# ------------------- 2. Fix SSL issues on macOS -------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ------------------- 3. Download 'punkt' -------------------
nltk.download('punkt', quiet=False)

# ------------------- 4. Test sent_tokenize -------------------
from nltk.tokenize import sent_tokenize


project_dir = "/Users/ayaansinghal/Downloads/Bolt App"

# Walk through all files in the project directory
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".py"):  # only Python files
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # fallback encoding
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            if "punkt" in content:
                content_new = content.replace("punkt", "punkt")
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content_new)
                except UnicodeEncodeError:
                    with open(file_path, "w", encoding="latin-1") as f:
                        f.write(content_new)
                print(f"Fixed {file_path}")



app = Flask(__name__)

app.secret_key = "supersecretkey"


USERS = ["Aanya", "Ayaan", "User3", "User4", "User5"]

USER_COLORS = {
    "Aanya": "tomato",
    "Ayaan": "orange",
    "User3": "sienna",
    "User4": "peru",
    "User5": "salmon"
}


conn = sqlite3.connect("chat.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender TEXT,
    body TEXT,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS todolist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    body TEXT,
    creater TEXT,
    deadline TEXT,
    completed_at TIMESTAMP NULL
)
""")


# Create meta table to track last cleared week
c.execute("""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
)
""")
conn.commit()
conn.close()

def clear_if_new_week():
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    
    # Get last cleared week
    c.execute("SELECT value FROM meta WHERE key='last_cleared_week'")
    row = c.fetchone()
    current_week = datetime.now().isocalendar()[1]  # ISO week number
    
    if not row or int(row[0]) != current_week:
        # Clear messages table
        c.execute("DELETE FROM messages")
        # Update last cleared week
        c.execute("REPLACE INTO meta (key, value) VALUES ('last_cleared_week', ?)", (str(current_week),))
        conn.commit()
    
    conn.close()

#
clear_if_new_week()




# Initialize models (use GPU if available)
device = 0  # set -1 for CPU
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
summarizer = pipeline("summarization", model="facebook/bart-base", device=device)
qa_model = pipeline("text2text-generation", model="bigscience/bloomz-560m", device=device)

# ------------------- Text Extraction -------------------
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# ------------------- Chunking -------------------
def chunk_text(text, chunk_size=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sent in sentences:
        if len(current_chunk) + len(sent.split()) <= chunk_size:
            current_chunk.append(sent)
        else:
            chunks.append(" ".join(current_chunk))
            # Overlap handling
            current_chunk = current_chunk[-overlap:] + [sent] if overlap > 0 else [sent]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ------------------- FAISS Index -------------------
def create_faiss_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def search_query(query, chunks, index, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

# ------------------- Generate Answer -------------------
def generate_answer(query, retrieved_chunks):
    # Summarize chunks in batch
    chunk_summaries = [
        summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        for chunk in retrieved_chunks
    ]
    
    summary = "\n".join(chunk_summaries)

    # Build prompt
    prompt = (
        f"Summary:\n{summary}\n\n"
        f"Question: {query}\n\n"
        f"Answer in bullet points. Include numbers from the summary with short explanations. "
        f"If no numbers, give a ~100-word bullet-point answer."
    )

    response = qa_model(
        prompt, 
        max_new_tokens=250, 
        num_beams=1, 
        top_p=0.95, 
        do_sample=True
    )
    
    return response[0]["generated_text"]

# ------------------- Example Usage -------------------
pdf_text = extract_text_from_pdf("f1_in_schools_technical_regulations_2024-2025_development_class.pdf")
chunks = chunk_text(pdf_text)
index, embeddings = create_faiss_index(chunks)





@app.route("/", methods=["GET", "POST"])
def login():

    print("Request method:", request.method)
    print("Form data:", request.form)

    if request.method == "POST":
        username = request.form.get("enter_username")
        password = request.form.get("enter_password")


        if username in USERS and password == "Bolt@StemRacing0":
            session["username"] = username 
            return redirect(f"/chat?username={username}")
        else:
            return render_template("select_user.html", users=USERS)
    
    return render_template("select_user.html", users=USERS)




@app.route("/chat", methods=["GET", "POST"])
def chat():

    username = session.get("username")  

    if not username or username not in USERS:
        return render_template("select_user.html", users=USERS)  # Ask user to pick

    if request.method == "POST":
        message = request.form.get("message")
        if message:
            conn = sqlite3.connect("chat.db")
            c = conn.cursor()
            c.execute("INSERT INTO messages (sender, body) VALUES (?, ?)", (username, message))
            conn.commit()
            conn.close()
        return redirect(f"/chat?username={username}")  # Refresh page

    
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    c.execute("SELECT sender, body, time FROM messages ORDER BY id ASC")
    messages = c.fetchall()
    conn.close()

    return render_template("chat.html", messages=messages, username=username, user_colors=USER_COLORS)








@app.route("/todo", methods=["GET", "POST"])
def todo():
    username = session.get("username")  

    # Check if user wants to see completed tasks
    show_completed = request.args.get("show_completed") == "1"

    if request.method == "POST":
        form_id = request.form.get("form_id")

        if form_id == 'create_task':
            todo_title = request.form.get("todo-title")
            todo_body = request.form.get("todo-body")
            todo_deadline = request.form.get("todo-deadline")
            if todo_title and todo_body and todo_deadline:
                conn = sqlite3.connect("chat.db")
                c = conn.cursor()
                c.execute(
                    "INSERT INTO todolist (title, body, creater, deadline) VALUES (?, ?, ?, ?)",
                    (todo_title, todo_body, username, todo_deadline)
                )
                conn.commit()
                conn.close()
            return redirect(f"/todo?show_completed={int(show_completed)}")

        elif form_id == "complete_task":
            task_id = request.form.get("task_id")
            conn = sqlite3.connect("chat.db")
            c = conn.cursor()
            c.execute("UPDATE todolist SET completed_at = CURRENT_TIMESTAMP WHERE id = ?", (task_id,))
            conn.commit()
            conn.close()
            return redirect(f"/todo?show_completed={int(show_completed)}")

    # Fetch tasks
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    if show_completed:
        c.execute("SELECT id, title, body, creater, deadline, completed_at FROM todolist ORDER BY deadline ASC")
    else:
        c.execute("SELECT id, title, body, creater, deadline, completed_at FROM todolist WHERE completed_at IS NULL ORDER BY deadline ASC")
    todolist_items = c.fetchall()
    conn.close()

    return render_template(
        "todo.html",
        todolist_items=todolist_items,
        username=username,
        show_completed=show_completed
    )



@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    
    answer = None
   
    if request.method == "POST":
        query = request.form.get("chatbot_query")
        
        retrieved = search_query(query, chunks, index)
        answer = generate_answer(query, retrieved)
           
        
    return render_template("chatbot.html", answer=answer)
        
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)





