from flask import Flask, render_template, request, redirect, session, jsonify
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
from PDF_summary import Rules_and_Regs_sum
import math
from huggingface_hub import InferenceClient
import dropbox
from mailjet_rest import Client
import google.generativeai as genai
import fitz


app = Flask(__name__)
app.secret_key = "supersecretkey"

# ------------------------ USERS ------------------------
USERS = ["Ahaan", "Ayaan", "Ayush", "Vishak", "Nathan", "Tharun"]
USER_COLORS = {
    "Ahaan": "tomato",
    "Ayaan": "orange",
    "Ayush": "sienna",
    "Vishak": "peru",
    "Tharun": "yellow",
    "Nathan": "salmon"
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
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
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

    # ML features
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

    c.execute("""
    CREATE TABLE IF NOT EXISTS ar_sim_results (
        id SERIAL PRIMARY KEY,
        car_name TEXT,
        drag_co DOUBLE PRECISION,
        lift_co DOUBLE PRECISION,
        filepath TEXT
    )
    """)


    c.execute("""
    CREATE TABLE IF NOT EXISTS timeline_progress (
        id SERIAL PRIMARY KEY,
        progress_date DATE DEFAULT CURRENT_DATE
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

    # Check last cleared week
    c.execute("SELECT value FROM meta WHERE key = %s", ('last_cleared_week',))
    row = c.fetchone()

    if not row or int(row['value']) != current_week:
        # Delete old messages
        c.execute("DELETE FROM messages WHERE time < NOW() - INTERVAL '30 days'")

        # Update meta table
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

    # Check last cleared day
    c.execute("SELECT value FROM meta WHERE key = %s", ('last_cleared_day',))
    row = c.fetchone()

    if not row or row['value'] != today_str:
        # Delete old chatbot data
        c.execute("DELETE FROM chatbot")

        # Update meta table
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

import os
import fitz  # PyMuPDF
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemma-3-27b-it")

def get_chunks_from_pdfs(pdf_list):
    """Splits PDFs into 'chunks' (roughly 1 page each)"""
    chunks = []
    for path in pdf_list:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    # Store text along with where it came from
                    chunks.append({"text": text, "source": f"{path} - Page {i+1}"})
    return chunks

# 1. Load the knowledge base into chunks
pdf_paths = ["Competition_Regs.pdf", "Technical_Regs.pdf", "Regionals_engineering.pdf", "Regionals_enterprise.pdf"]
ALL_CHUNKS = get_chunks_from_pdfs(pdf_paths)

def find_relevant_chunks(query, chunks, top_n=3):
    """Simple search to find chunks containing query keywords"""
    query_words = query.lower().split()
    scored_chunks = []
    
    for chunk in chunks:
        score = sum(1 for word in query_words if word in chunk['text'].lower())
        scored_chunks.append((score, chunk))
    
    # Sort by score and take the best ones
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_chunks[:top_n]]


def ask_chatbot(question, chat_history=[]):
    relevant_data = find_relevant_chunks(question, ALL_CHUNKS)
    context_text = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in relevant_data])

    chat = model.start_chat(history=chat_history)

    prompt = f"""
    CONTEXT FROM REGS:
    {context_text}

    USER QUESTION:
    {question}

    (Remember to use a natural, conversational tone without markdown or bullet points. Your Name is lumin, and you are part of team bolt, here to help the userstand the competition and the team)
    """

    try:
        response = chat.send_message(prompt)
        return response.text, chat.history
    except Exception as e:
        return f"Error: {str(e)}", chat_history




# ------------------------ TRACK TIME SIM ------------------------
import math

def simulate_track_time(drag_20ms, lift_20ms, car_mass_g,
                        total_energy=330,  # J
                        track_length=20,    # m
                        dt=0.001,
                        burn_time=0.6,      # seconds to deliver energy
                        show_diagnostics=False):

    mass = car_mass_g / 1000.0  # convert g → kg
    position = 0.0
    velocity = 0.0
    time = 0.0
    energy_used = 0.0

    positions = []
    speeds = []

    thrust_trace = []
    drag_trace = []
    rolling_trace = []
    lift_trace = []

    while position < track_length:

        positions.append(position)
        speeds.append(velocity)

        # Aerodynamic forces
        drag = drag_20ms * (velocity / 20.0)**2
        lift = lift_20ms * (velocity / 20.0)**2

        # Rolling resistance
        normal_force = mass * 9.81 - lift
        rolling_force = 0.015 * normal_force
        axle_friction = 0.4

        resist_force = drag + rolling_force + axle_friction

        # Energy-based thrust
        if energy_used < total_energy:
            remaining_energy = total_energy - energy_used
            power = remaining_energy / burn_time
            thrust = power / max(velocity, 0.1)  # F = P/v
        else:
            thrust = 0

        # Energy accounting
        energy_used += thrust * velocity * dt

        # Dynamics
        net_force = thrust - resist_force
        acceleration = net_force / mass
        velocity += acceleration * dt
        if velocity < 0:
            velocity = 0
        position += velocity * dt
        time += dt

        # Diagnostics
        thrust_trace.append(thrust)
        drag_trace.append(drag)
        rolling_trace.append(rolling_force)
        lift_trace.append(lift)

        # Safety break
        if time > 60:
            break

    diagnostics = {
        "thrust": thrust_trace,
        "drag": drag_trace,
        "rolling": rolling_trace,
        "lift": lift_trace
    }

    if show_diagnostics:
        return positions, speeds, time, diagnostics
    else:
        return positions, speeds, time

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
        if username in USERS and password == "Bolt@StemRacing0" and not username == 'Ayaan':
            session["username"] = username 
            return redirect(f"/chat")
        elif username == 'Ayaan' and password == "Bolt@StemRacing8":
            session["username"] = username 
            return redirect(f"/chat")
        else:
            return render_template("select_user.html", users=USERS)
    return render_template("select_user.html", users=USERS)





@app.route("/chat", methods=["GET", "POST"])
def chat():
    username = session.get("username")
    if not username or username not in USERS:
        return render_template("select_user.html", users=USERS)

    conn = get_pg_conn()
    c = conn.cursor()

    if request.method == "POST":
        # Handle new message from form or AJAX
        if request.is_json:
            data = request.get_json()
            message = data.get("message")
        else:
            message = request.form.get("message")

        if message:
            c.execute(
                "INSERT INTO messages (sender, body) VALUES (%s, %s)",
                (username, message)
            )
            conn.commit()

    # Fetch all messages
    c.execute("SELECT sender, body, time FROM messages ORDER BY id ASC")
    messages = c.fetchall()
    conn.close()

    # If AJAX request for JSON
    if request.args.get("json"):
        json_messages = []
        for sender, body, time in messages:
            json_messages.append({
                "sender": sender,
                "body": body,
                "time": time.strftime("%H:%M"),  # format time
                "color": USER_COLORS.get(sender, "#000000")
            })
        return jsonify({"messages": json_messages})

    # Normal page render
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
                c.execute(
                    """
                    INSERT INTO todolist (title, body, creater, deadline)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (todo_title, todo_body, username, todo_deadline)
                )
                conn.commit()
            conn.close()
            return redirect(f"/todo?show_completed={int(show_completed)}")

        elif form_id == "complete_task":
            task_id = request.form.get("task_id")
            c.execute(
                "UPDATE todolist SET completed_at = CURRENT_TIMESTAMP WHERE id = %s",
                (task_id,)
            )
            conn.commit()
            conn.close()
            return redirect(f"/todo?show_completed={int(show_completed)}")

    conn = get_pg_conn()
    c = conn.cursor(cursor_factory=RealDictCursor)

    if show_completed:
        c.execute(
            """
            SELECT id, title, body, creater, deadline, completed_at
            FROM todolist
            ORDER BY deadline ASC
            """
        )
    else:
        c.execute(
            """
            SELECT id, title, body, creater, deadline, completed_at
            FROM todolist
            WHERE completed_at IS NULL
            ORDER BY deadline ASC
            """
        )

    todolist_items = c.fetchall()  # already list of dicts
    conn.close()

    return render_template(
        "todo.html",
        todolist_items=todolist_items,
        username=username,
        show_completed=show_completed
    )

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    conn = get_pg_conn()
    c = conn.cursor()

    if request.method == "POST":
        
        data = request.get_json()
        query = data.get("chatbot_query")
        
        if query:
      
            conn = get_pg_conn()
            c = conn.cursor()
            c.execute("SELECT answer FROM chatbot ORDER BY id DESC LIMIT 1")
            row = c.fetchone()

            if row:
                formatted_history = [{"role": "assistant", "content": row[0]}]
            else:
                formatted_history = []

            answer, _ = ask_chatbot(query, chat_history=formatted_history)
            c.execute(
                "INSERT INTO chatbot (query, answer) VALUES (%s, %s)",
                (query, answer)
            )
            conn.commit()

            return jsonify({"answer": answer})


    c.execute("SELECT id, query, answer FROM chatbot ORDER BY id")
    chatbot_convo = [{"query": q, "answer": a} for _, q, a in c.fetchall()]
    conn.close()
    return render_template("chatbot.html", chatbot_convo=chatbot_convo)



@app.route("/sim", methods=["GET", "POST"])
def sim():
    import io, base64
    import matplotlib.pyplot as plt

    img_speeds = None
    img_forces = None
    time_result = None
    message = None
    drag = None
    lift = None
    frontal_area = None
    cross_sections = None

    if request.method == "POST":
        form_id = request.form.get("form_id")

        if form_id == "track_time_calc":
            try:
                # --- get user inputs ---
                drag_force = float(request.form.get("drag_force"))     # N at 20 m/s
                lift_force = float(request.form.get("lift_force"))     # N at 20 m/s
                mass_g = float(request.form.get("mass"))              # grams

                # --- run simulation ---
                distances, speeds, time_result, diag = simulate_track_time(
                    drag_20ms=drag_force,
                    lift_20ms=lift_force,
                    car_mass_g=mass_g,
                    show_diagnostics=True
                )

                # --- speed vs distance plot ---
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
                img_speeds = base64.b64encode(buf.getvalue()).decode("utf-8")

                # --- forces vs distance plot ---
                buf2 = io.BytesIO()
                plt.figure(figsize=(8,5))
                plt.plot(distances, diag["drag"], label="Drag Force (N)", color="red")
                plt.plot(distances, diag["lift"], label="Lift Force (N)", color="blue")
                plt.xlabel("Distance (m)")
                plt.ylabel("Force (N)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(buf2, format="png")
                buf2.seek(0)
                plt.close()
                img_forces = base64.b64encode(buf2.getvalue()).decode("utf-8")

            except Exception as e:
                message = f"Simulation error: {str(e)}"

        else:
            action = request.form.get("action")
            files = request.files.getlist("obj_files")

            try:
                if action == "train":
                    drag_values_str = request.form.get("drag_values")
                    lift_values_str = request.form.get("lift_values")
                    cross_sections = []

                    if drag_values_str and lift_values_str:
                        drag_values = [float(x.strip()) for x in drag_values_str.split(",")]
                        lift_values = [float(x.strip()) for x in lift_values_str.split(",")]

                        if len(drag_values) != len(files) or len(lift_values) != len(files):
                            message = "Number of values must match number of OBJ files"
                        else:
                            for f, drag_val, lift_val in zip(files, drag_values, lift_values):
                                path = os.path.join(UPLOAD_FOLDER, f.filename)
                                f.save(path)
                                shape = load_obj_file(path)
                                features, cross_section = extract_features(shape)
                                save_training_data(features, drag_val, lift_val)
                                cross_sections.append(round(cross_section, 3))
                            message = "Data saved successfully! Model will update on next prediction."
                    else:
                        
                        cross_sections = []
                        for f in files:
                            path = os.path.join(UPLOAD_FOLDER, f.filename)
                            f.save(path)
                            shape = load_obj_file(path)
                            _, cross_section = extract_features(shape)
                            cross_sections.append(round(cross_section,3))
                        message = "Cross-section(s) extracted (no training data saved)."

                elif action == "predict":
                    if len(files) != 1:
                        message = "Please upload exactly one OBJ file for prediction"
                    else:
                        f = files[0]
                        path = os.path.join(UPLOAD_FOLDER, f.filename)
                        f.save(path)
                        drag, lift, frontal_area = predict_coeffs(path)
                else:
                    message = "Unknown action"
            except Exception as e:
                message = f"Error processing files: {str(e)}"

    
    return render_template(
        "sim.html",
        time=time_result,
        graph_speed=img_speeds,
        graph_forces=img_forces,
        drag=round(drag,3) if drag is not None else None,
        lift=round(lift,3) if lift is not None else None,
        frontal_area=round(frontal_area,3) if frontal_area is not None else None,
        message=message,
        cross_sections=cross_sections
    )


ACCESS_TOKEN = os.environ.get("DROPBOX_SIM_TOKEN")  
DROPBOX_FOLDER = "/ANSYS_GLB_Files"
dbx = dropbox.Dropbox(ACCESS_TOKEN)

last_dropbox_paths = {}

os.makedirs("uploads", exist_ok=True)

@app.route("/ansys/upload", methods=["POST"])
def upload():
    # --- GLB file upload ---
    if "file" in request.files:
        glb_file = request.files["file"]
        filename = glb_file.filename
        local_path = os.path.join("uploads", filename)
        glb_file.save(local_path)

        dropbox_path = f"{DROPBOX_FOLDER}/{filename}"
        with open(local_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)

        # Save path by car name (use filename without extension)
        car_name = os.path.splitext(filename)[0]
        last_dropbox_paths[car_name] = dropbox_path

        print(f"Uploaded {filename} to Dropbox at {dropbox_path}")
        return "GLB uploaded", 200

    # --- JSON drag/lift data upload ---
    if request.is_json:
        data = request.get_json()
        car_name = data.get("car_name")
        drag = data.get("drag")
        lift = data.get("lift")

        # Get Dropbox path if GLB was uploaded
        dropbox_path = last_dropbox_paths.get(car_name, None)

        # Insert into Postgres
        conn = get_pg_conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO ar_sim_results (car_name, drag_co, lift_co, filepath) VALUES (%s, %s, %s, %s)",
            (car_name, drag, lift, dropbox_path)
        )
        conn.commit()
        conn.close()

        print(f"Stored results for {car_name}: Drag={drag}, Lift={lift}, File={dropbox_path}")
        return "Data stored", 200

    return "Invalid request", 400




@app.route("/ansys/ar", methods=["GET", "POST"])
def AR_sim():

    conn = get_pg_conn()
    c = conn.cursor()

    if request.method == "POST":
        table_ids = request.form.getlist('options')
        if len(table_ids) > 3:
            return render_template("ar_sim.html", error='Please select 3 values or less')
        
        file_paths = []
        for i in table_ids:
            c.execute("SELECT filepath FROM ar_sim_results WHERE id = %s;", (i,))
            result = c.fetchone()
            if result:
                file_paths.append(result[0])
        
        
        urls = []
        for fpath in file_paths:
            try:
                shared_link_metadata = dbx.sharing_create_shared_link_with_settings(fpath)
            except dropbox.exceptions.ApiError:
                links = dbx.sharing_list_shared_links(fpath).links
                if links:
                    shared_link_metadata = links[0]
                else:
                    raise

            
            url = shared_link_metadata.url.replace("?dl=0", "?dl=1")
            urls.append(url)

        
        return render_template("AR_sim_viewer.html", urls=urls)

    c.execute("SELECT id, car_name, drag_co, lift_co, filepath FROM ar_sim_results ORDER BY id")
    ar_sim_selections = c.fetchall()
    c.close()
    conn.close()


    return render_template("AR_sim_settings.html", ar_sim_selections=ar_sim_selections)





MAILJET_API_KEY = os.getenv("MAILJET_API_KEY")
MAILJET_API_SECRET = os.getenv("MAILJET_API_SECRET")
MAILJET_SENDER = os.getenv("MAILJET_SENDER")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(BASE_DIR, "templates", "mass_email_template.html")

with open(template_path, "r", encoding="utf-8") as f:
    EMAIL_TEMPLATE = f.read()

SENT_FILE = "sent_emails.txt"

def load_sent_emails():
    if not os.path.exists(SENT_FILE):
        return set()
    with open(SENT_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_sent_emails(emails):
    with open(SENT_FILE, "a") as f:
        for email in emails:
            f.write(email + "\n")

def send_emails(subject, contacts, attachment):
    mailjet = Client(auth=(MAILJET_API_KEY, MAILJET_API_SECRET), version='v3.1')

    attachment_data = None
    attachment_name = None
    if attachment and attachment.filename:
        attachment_data = base64.b64encode(attachment.read()).decode()
        attachment_name = attachment.filename

    for name, email in contacts:
        html_body = EMAIL_TEMPLATE.replace("{{name}}", name)

        data = {
            'Messages': [
                {
                    "From": {"Email": MAILJET_SENDER, "Name": "Your App"},
                    "To": [{"Email": email, "Name": name}],
                    "Subject": subject,
                    "HTMLPart": html_body,
                    "Attachments": [
                        {
                            "ContentType": "application/pdf",
                            "Filename": attachment_name,
                            "Base64Content": attachment_data
                        }
                    ] if attachment_data else []
                }
            ]
        }

        result = mailjet.send.create(data=data)
        print(f"Sent to {email}: {result.status_code}")

@app.route("/mass-email", methods=["GET", "POST"])
def mass_email():
    try:
        if request.method == "POST":
            subject = request.form["subject"]
            raw_contacts = request.form["contacts"]
            attachment = request.files.get("attachment")

            previous_emails = load_sent_emails()
            contacts = []

            for line in raw_contacts.splitlines():
                if "," in line:
                    name, email = line.split(",", 1)
                    name = name.strip()
                    email = email.strip()
                    if email not in previous_emails:
                        contacts.append((name, email))
                        previous_emails.add(email)

            if not contacts:
                return "<h3>No new emails to send (all duplicates skipped)</h3>"

            send_emails(subject, contacts, attachment)
            save_sent_emails([email for _, email in contacts])

            return f"<h3>Emails sent successfully ({len(contacts)} recipients)</h3>"

        return render_template("mass_email.html")


    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h3>Internal Server Error: {e}</h3>", 500





    

@app.route("/reactiontime", methods=["GET", "POST"])
def react():
    return render_template("React_Test.html")

@app.route("/arviewer", methods=["GET", "POST"])
def public_ar():
    return render_template("AR.html")

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)




