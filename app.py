from flask import Flask, render_template, request, redirect, session
from datetime import datetime, date
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import io
import base64
import joblib
import trimesh
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import requests

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ------------------------ USERS ------------------------
USERS = ["Aanya", "Ayaan", "User3", "User4", "User5"]
USER_COLORS = {
    "Aanya": "tomato",
    "Ayaan": "orange",
    "User3": "sienna",
    "User4": "peru",
    "User5": "salmon"
}

# ------------------------ POSTGRES SETUP ------------------------
DB_URL = os.getenv("DATABASE_URL")

def get_pg_conn():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def init_db():
    conn = get_pg_conn()
    c = conn.cursor()
    
    # Messages table
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        sender TEXT,
        body TEXT,
        time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Todo list
    c.execute("""
    CREATE TABLE IF NOT EXISTS todolist (
        id SERIAL PRIMARY KEY,
        title TEXT,
        body TEXT,
        creater TEXT,
        deadline DATE,
        completed_at TIMESTAMP NULL
    )
    """)
    
    # Meta table
    c.execute("""
    DROP TABLE IF EXISTS meta;          
    
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT

    """)
    
    # Doc summary
    c.execute("""
    CREATE TABLE IF NOT EXISTS doc_summary (
        id SERIAL PRIMARY KEY,
        summary TEXT
    )
    """)
    
    # Chatbot table
    c.execute("""
    CREATE TABLE IF NOT EXISTS chatbot (
        id SERIAL PRIMARY KEY,
        query TEXT,
        answer TEXT
    )
    """)
    
    # ML features table
    c.execute("""
    CREATE TABLE IF NOT EXISTS ml_features (
        id SERIAL PRIMARY KEY,
        volume DOUBLE PRECISION,
        area DOUBLE PRECISION,
        dx DOUBLE PRECISION,
        dy DOUBLE PRECISION,
        dz DOUBLE PRECISION,
        aspect_xy DOUBLE PRECISION,
        aspect_xz DOUBLE PRECISION,
        avg_cross_section DOUBLE PRECISION,
        convex_vol DOUBLE PRECISION,
        diag DOUBLE PRECISION,
        slenderness DOUBLE PRECISION,
        num_vertices INTEGER,
        num_faces INTEGER,
        drag DOUBLE PRECISION,
        lift DOUBLE PRECISION
    )
    """)
    
    conn.commit()
    conn.close()

init_db()

# ------------------------ WEEK / DAY CLEAR ------------------------
def clear_if_new_week():
    conn = get_pg_conn()
    c = conn.cursor()
    current_week = datetime.now().isocalendar()[1]
    c.execute(
        "INSERT INTO meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        ('last_cleared_week', str(current_week))
    )
    conn.commit()
    conn.close()

def clear_if_new_day():
    conn = get_pg_conn()
    c = conn.cursor()
    today_str = date.today().isoformat()
    c.execute(
        "INSERT INTO meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        ('last_cleared_day', today_str)
    )
    conn.commit()
    conn.close()


clear_if_new_week()
clear_if_new_day()

# ------------------------ NLP / CHATBOT ------------------------
API_URL = "https://api-inference.huggingface.co/models/Ay8/google/flan-t5-large"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

my_summary = "This is the summary string your answers will be based on."

def ask_question_with_prompt(question):
    prompt = f"""Summary:{my_summary} Question:{question} Answer concisely in bullet points: Use numbers and their explanations"""
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return "Error: Unexpected response format from API."
    except requests.exceptions.Timeout:
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed ({str(e)})"
    except (KeyError, IndexError, TypeError):
        return "Error: Failed to parse API response."

# ------------------------ TRACK TIME SIM ------------------------
def run_track_time_sim(drag_co, lift_co, mass, cross_section):
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
    distances, speeds, energies = [], [], []

    while distance < 20 and energy > 0:
        dis_travelled = speed * dt
        distance += dis_travelled
        downforce = 0.5 * air_density * (-lift_co) * cross_section * (speed ** 2)
        drag_energy_loss = 0.5 * air_density * drag_co * cross_section * (speed ** 2) * dis_travelled
        rolling_energy_loss = rolling_co * ((mass * gravity) + downforce) * dis_travelled
        friction_energy_loss = friction * (dis_travelled / 20)
        total_energy_loss = drag_energy_loss + rolling_energy_loss + friction_energy_loss
        energy -= total_energy_loss
        time += dt
        speed = np.sqrt((2 * energy) / mass) if energy > 0 else 0
        distances.append(distance)
        speeds.append(speed)
        energies.append(energy)

    return distances, speeds, energies, time

# ------------------------ CONFIG ------------------------
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/drag_rf_model.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------------ OBJ LOADER & FEATURES ------------------------
def load_obj_file(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("File could not be loaded as a mesh")
    return mesh

def extract_features(mesh):
    volume = mesh.volume
    area = mesh.area
    bbox = mesh.bounds
    dx, dy, dz = bbox[1] - bbox[0]
    aspect_xy = dx / dy if dy != 0 else 0
    aspect_xz = dx / dz if dz != 0 else 0
    slenderness = max(dx, dy, dz) / min(dx, dy, dz) if min(dx, dy, dz) != 0 else 0
    convex_vol = mesh.convex_hull.volume
    diag = np.linalg.norm(bbox[1] - bbox[0])
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.faces)
    z_levels = np.linspace(bbox[0][2], bbox[1][2], num=3)
    cross_sections = [mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1]).area if mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1]) else 0 for z in z_levels]
    avg_cross_section = np.mean(cross_sections)
    frontal_cross_section = dx * dy
    features = np.array([volume, area, dx, dy, dz, aspect_xy, aspect_xz, avg_cross_section, convex_vol, diag, slenderness, num_vertices, num_faces])
    return features, frontal_cross_section

# ------------------------ POSTGRES ML DATA ------------------------
def save_training_data(features, drag, lift):
    conn = get_pg_conn()
    c = conn.cursor()
    query = """
        INSERT INTO ml_features
        (volume, area, dx, dy, dz, aspect_xy, aspect_xz, avg_cross_section, convex_vol, diag, slenderness, num_vertices, num_faces, drag, lift)
        VALUES %s
    """
    execute_values(c, query, [(*features.tolist(), drag, lift)])
    conn.commit()
    conn.close()

def train_model_from_dataset():
    conn = get_pg_conn()
    df = pd.read_sql("SELECT * FROM ml_features", conn)
    conn.close()
    if df.empty:
        raise ValueError("No training data available yet.")
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2:].values
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_PATH)
    return rf

def predict_coeffs(obj_file_path):
    rf = train_model_from_dataset()
    mesh = load_obj_file(obj_file_path)
    features, frontal_cross_section = extract_features(mesh)
    features = features.reshape(1, -1)
    drag, lift = rf.predict(features)[0]
    return float(drag), float(lift), frontal_cross_section

# ------------------------ ROUTES ------------------------
@app.route("/", methods=["GET", "POST"])
def login():
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
        return render_template("select_user.html", users=USERS)
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            conn = get_pg_conn()
            c = conn.cursor()
            c.execute("INSERT INTO messages (sender, body) VALUES (%s, %s)", (username, message))
            conn.commit()
            conn.close()
        return redirect(f"/chat?username={username}")
    conn = get_pg_conn()
    c = conn.cursor()
    c.execute("SELECT sender, body, time FROM messages ORDER BY id ASC")
    messages = c.fetchall()
    conn.close()
    return render_template("chat.html", messages=messages, username=username, user_colors=USER_COLORS)

@app.route("/todo", methods=["GET", "POST"])
def todo():
    username = session.get("username")
    show_completed = request.args.get("show_completed") == "1"
    if request.method == "POST":
        form_id = request.form.get("form_id")
        conn = get_pg_conn()
        c = conn.cursor()
        if form_id == 'create_task':
            todo_title = request.form.get("todo-title")
            todo_body = request.form.get("todo-body")
            todo_deadline = request.form.get("todo-deadline")
            if todo_title and todo_body and todo_deadline:
                c.execute("INSERT INTO todolist (title, body, creater, deadline) VALUES (%s, %s, %s, %s)", (todo_title, todo_body, username, todo_deadline))
                conn.commit()
            conn.close()
            return redirect(f"/todo?show_completed={int(show_completed)}")
        elif form_id == "complete_task":
            task_id = request.form.get("task_id")
            c.execute("UPDATE todolist SET completed_at = CURRENT_TIMESTAMP WHERE id = %s", (task_id,))
            conn.commit()
            conn.close()
            return redirect(f"/todo?show_completed={int(show_completed)}")
    conn = get_pg_conn()
    c = conn.cursor()
    if show_completed:
        c.execute("SELECT id, title, body, creater, deadline, completed_at FROM todolist ORDER BY deadline ASC")
    else:
        c.execute("SELECT id, title, body, creater, deadline, completed_at FROM todolist WHERE completed_at IS NULL ORDER BY deadline ASC")
    todolist_items = c.fetchall()
    conn.close()
    return render_template("todo.html", todolist_items=todolist_items, username=username, show_completed=show_completed)

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    answer = None
    chatbot_convo = []
    if request.method == "POST":
        query = request.form.get("chatbot_query")
        if query:
            answer = ask_question_with_prompt(query)
        conn = get_pg_conn()
        c = conn.cursor()
        c.execute("SELECT id, query, answer FROM chatbot ORDER BY id")
        chatbot_convo = c.fetchall()
        conn.close()
    return render_template("chatbot.html", answer=answer, chatbot_convo=chatbot_convo)

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
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return render_template("sim.html", time=time, graph=img_base64)
        else:
            action = request.form.get("action")
            files = request.files.getlist("obj_files")
            message = ""
            if action == "train":
                drag_values_str = request.form.get("drag_values")
                lift_values_str = request.form.get("lift_values")
                cross_sections = []
                if drag_values_str and lift_values_str:
                    drag_values = [float(x.strip()) for x in drag_values_str.split(",")]
                    lift_values = [float(x.strip()) for x in lift_values_str.split(",")]
                    if len(drag_values) != len(files) or len(lift_values) != len(files):
                        return render_template("sim.html", message="Number of values must match number of OBJ files")
                    for f, drag, lift in zip(files, drag_values, lift_values):
                        path = os.path.join(UPLOAD_FOLDER, f.filename)
                        f.save(path)
                        shape = load_obj_file(path)
                        features, cross_section = extract_features(shape)
                        save_training_data(features, drag, lift)
                        cross_sections.append(round(cross_section, 3))
                    return render_template("sim.html", message="Data saved successfully! Model will update on next prediction.", cross_sections=cross_sections)
                else:
                    for f in files:
                        path = os.path.join(UPLOAD_FOLDER, f.filename)
                        f.save(path)
                        shape = load_obj_file(path)
                        _, cross_section = extract_features(shape)
                        frontal_area = round(cross_section, 3)
                    return render_template("sim.html", message="Cross-section(s) extracted (no training data saved).", frontal_area=frontal_area)
            elif action == "predict":
                if len(files) != 1:
                    return render_template("sim.html", message="Please upload exactly one OBJ file for prediction")
                f = files[0]
                path = os.path.join(UPLOAD_FOLDER, f.filename)
                f.save(path)
                drag, lift, frontal_area = predict_coeffs(path)
                return render_template("sim.html", drag=round(drag,3), lift=round(lift,3), frontal_area=round(frontal_area,3))
            else:
                return render_template("sim.html", message="Unknown action")
    return render_template("sim.html")

@app.route("/vr", methods=["GET", "POST"])
def vr():
    return render_template("Bolt_VR.html")

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 5000 for local dev
    app.run(host="0.0.0.0", port=port, debug=True)




