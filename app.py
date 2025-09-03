from flask import Flask, render_template, request, redirect, session
import sqlite3
from datetime import datetime, timedelta, date
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import os
import nltk

from nltk.tokenize import sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import io
import base64
import joblib
from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepProj import BRepProj_Projection




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

c.execute("""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    key_day TEXT,
    value_day TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS chatbot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    answer TEXT
)
""")


conn.commit()
conn.close()


# ------------------------------------------------------------------------------------------------------


def clear_if_new_week():
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    
    
    c.execute("SELECT value FROM meta WHERE key='last_cleared_week'")
    row = c.fetchone()
    current_week = datetime.now().isocalendar()[1]  
    
    if not row or int(row[0]) != current_week:
     
        c.execute("DELETE FROM messages")

        
        c.execute("REPLACE INTO meta (key, value) VALUES ('last_cleared_week', ?)", (str(current_week),))
        conn.commit()
    
    conn.close()

#
clear_if_new_week()


def clear_if_new_day():
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    
    c.execute("SELECT value_day FROM meta WHERE key_day='last_cleared_day'")
    row = c.fetchone()
    today_str = date.today().isoformat()  
    
    if not row or row[0] != today_str:
  
        c.execute("DELETE FROM chatbot")
        
        c.execute("REPLACE INTO meta (key_day, value_day) VALUES ('last_cleared_day', ?)", (today_str,))
    
        conn.commit()

    conn.close()

clear_if_new_day()


# ------------------------------------------------------------------------------------------------------



device = 0  # set -1 for CPU
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
summarizer = pipeline("summarization", model="facebook/bart-base", device=device)
qa_model = pipeline("text2text-generation", model="bigscience/bloomz-560m", device=device)


def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


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


def generate_answer(query, retrieved_chunks):
    
    chunk_summaries = [
        summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        for chunk in retrieved_chunks
    ]
    
    summary = "\n".join(chunk_summaries)

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


pdf_text = extract_text_from_pdf("f1_in_schools_technical_regulations_2024-2025_development_class.pdf")
chunks = chunk_text(pdf_text)
index, embeddings = create_faiss_index(chunks)


# ------------------------------------------------------------------------------------------------------




def run_track_time_sim(drag_co, lift_co, mass, cross_section):
    # Constants
    rolling_co = 0
    friction = 0.05
    gravity = 9.8
    air_density = 1.225
    dt = 0.001

    initial_speed = 30
    initial_energy = 2000

    distance = 0
    speed = initial_speed
    energy = initial_energy
    time = 0
    iter = 0.001

    distances = []
    speeds = []
    energies = []

    while distance < 20 and energy > 0:
        dis_travelled = speed * dt
        distance += dis_travelled

        downforce = 0.5 * air_density * (-lift_co) * cross_section * (speed ** 2)

        drag_energy_loss = 0.5 * air_density * drag_co * cross_section * (speed ** 2) * dis_travelled
        rolling_energy_loss = rolling_co * ((mass * gravity) + downforce) * dis_travelled
        friction_energy_loss = friction * (dis_travelled / 20)

        total_energy_loss = drag_energy_loss + rolling_energy_loss + friction_energy_loss
        energy -= total_energy_loss

        time += iter

        if energy > 0:
            speed = np.sqrt((2 * energy) / mass)
        else:
            speed = 0

        distances.append(distance)
        speeds.append(speed)
        energies.append(energy)

    return distances, speeds, energies, time
 


# ------------------------------------------------------------------------------------------------------




UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/drag_rf_model.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- STEP loader ---
def load_step_file(file_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != 1:
        raise ValueError(f"Error reading STEP file: {file_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape

def calc_cross_section(shape, direction=(0, 0, 1)):
   
    airflow_dir = gp_Dir(*direction)
    plane = gp_Pln(gp_Pnt(0, 0, 0), airflow_dir)

    
    proj = BRepProj_Projection(shape, plane, airflow_dir)
    proj_shape = proj.Shape()

 
    props = GProp_GProps()
    brepgprop_SurfaceProperties(proj_shape, props)
    area = props.Mass()  

    return area

# --- Feature extraction ---
def extract_features(shape):
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    volume = props.Mass()

    props_surface = GProp_GProps()
    brepgprop_SurfaceProperties(shape, props_surface)
    area = props_surface.Mass()

    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    aspect_xy = dx / dy if dy != 0 else 0
    aspect_xz = dx / dz if dz != 0 else 0

    cross_section = calc_cross_section(shape)
    features = np.array([volume, area, dx, dy, dz, aspect_xy, aspect_xz, cross_section])

    return features, cross_section





def train_model(step_files, drag_values, lift_values):
    X = []
    cross_sections = []

    for f in step_files:
        shape = load_step_file(f)
        features, cross_section = extract_features(shape)
        X.append(features)
        cross_sections.append(cross_section)
    X = np.array(X)
    y = np.column_stack((drag_values, lift_values))

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    joblib.dump(rf, MODEL_PATH)
    return rf, cross_sections

# --- Prediction function ---
def predict_coeffs(step_file_path):
    if not os.path.exists(MODEL_PATH):
        raise ValueError("Model not trained yet.")
    rf = joblib.load(MODEL_PATH)
    shape = load_step_file(step_file_path)
    features, cross_section = extract_features(shape)
    features = features.reshape(1, -1)
    drag, lift = rf.predict(features)[0]
    return float(drag), float(lift), cross_section


# ------------------------------------------------------------------------------------------------------



def ml_importance_predicter():
    # ========================
    # Example dataset
    # ========================
    # Replace this with your real data
    # Columns = factors, Target = lap_time (or energy loss)
    data = pd.DataFrame({
        "drag": [0.4],
        "downforce": [0.65],
        "weight": [0.065],
        "friction": [0.05],
        "lap_time": [1.85, 1.92, 1.88, 2.05, 1.99]  # measured result
    })

    # Features (inputs) and target (output)
    X = data[["drag", "downforce", "weight", "friction"]]
    y = data["lap_time"]

    # ========================
    # Train Random Forest
    # ========================
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Feature importances (how much each factor contributes)
    importances = model.feature_importances_
    factors = X.columns

    # ========================
    # Plot with Matplotlib
    # ========================
    plt.figure(figsize=(8, 5))
    plt.bar(factors, importances, color="skyblue")
    plt.xlabel("Factors Contributing to Lap Time / Energy Loss")
    plt.ylabel("Importance (relative influence)")
    plt.title("What to Prioritize in Car Design")
    plt.show()




# ------------------------------------------------------------------------------------------------------




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



# ------------------------------------------------------------------------------------------------------




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



# ------------------------------------------------------------------------------------------------------




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


# ------------------------------------------------------------------------------------------------------



@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    
    answer = None
   
    if request.method == "POST":
        query = request.form.get("chatbot_query")
        
        retrieved = search_query(query, chunks, index)
        answer = generate_answer(query, retrieved)

        conn = sqlite3.connect("chat.db")
        c = conn.cursor()
        c.execute("INSERT INTO chatbot (query, answer) VALUES (?, ?)", (query, answer))
        c.execute("SELECT id, query, answer FROM chatbot ORDER BY id")
        chatbot_convo = c.fetchall()
        conn.commit()
        conn.close()
           
        
    return render_template("chatbot.html", answer=answer, chatbot_convo=chatbot_convo)



# ------------------------------------------------------------------------------------------------------



@app.route("/sim", methods=["GET", "POST"])
def sim():

    if request.method == "POST":

        form_id = request.form.get("form_id")

        if form_id == "track_time_calc":

            drag_co = float(request.form.get("drag_co"))
            lift_co = float(request.form.get("lift_co"))
            mass = float(request.form.get("mass"))
            cross_section = float(request.form.get("cross_section"))
        
        
            
    
            distances, speeds, energies, time = run_track_time_sim(drag_co, lift_co, mass, cross_section)

    # Create Matplotlib figure in memory
            buf = io.BytesIO()
            plt.figure(figsize=(8,5))
            plt.plot(distances, speeds, label="Speed (m/s)", color="gold")
            plt.xlabel("Distance (m)")
            plt.ylabel("Speed (m/s)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()

    # Encode image as base64 to embed directly in HTML
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return render_template("sim.html", time=time, graph=img_base64)
        

        else:
            action = request.form.get("action")
            files = request.files.getlist("step_files")

            if action == "train":
                drag_values_str = request.form.get("drag_values")
                if not drag_values_str:
                    return render_template("sim.html", message="Drag values are required for training")
                drag_values = [float(x.strip()) for x in drag_values_str.split(",")]
                if len(drag_values) != len(files):
                    return render_template("sim.html", message="Number of drag values must match number of STEP files")
                

                lift_values_str = request.form.get("lift_values")
                if not lift_values_str:
                    return render_template("sim.html", message="Lift values are required for training")
                lift_values = [float(x.strip()) for x in lift_values_str.split(",")]
                if len(lift_values) != len(files):
                    return render_template("sim.html", message="Number of drag values must match number of STEP files")

                saved_paths = []
                for f in files:
                    path = os.path.join(UPLOAD_FOLDER, f.filename)
                    f.save(path)
                    saved_paths.append(path)

                rf_model, cross_sections = train_model(saved_paths, drag_values, lift_values)
                cross_sections = [round(cs, 3) for cs in cross_sections]
                    
                
                return render_template("sim.html", message="Model trained successfully!", cross_sections=cross_sections)

            elif action == "predict":
                if len(files) != 1:
                    return render_template("sim.html", message="Please upload exactly one STEP file for prediction")
                f = files[0]
                path = os.path.join(UPLOAD_FOLDER, f.filename)
                f.save(path)
                drag, lift, frontal_area = predict_coeffs(path)
                drag_mlsim_coeff = round(drag, 3)
                lift_mlsim_coeff = round(lift, 3)
                frontal_area = round(frontal_area, 3)

                return render_template("sim.html", drag=drag_mlsim_coeff, lift=lift_mlsim_coeff, frontal_area=frontal_area)
            else:
                return render_template("sim.html", message="Unknown action")




# ------------------------------------------------------------------------------------------------------
    


 



    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)





