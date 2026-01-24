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
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
load_dotenv()

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


client = InferenceClient(token=os.environ["HUGGINGFACE_API_TOKEN"])


def split_summary_into_sections(summary, max_words=500):
    words = summary.split()
    sections = []
    for i in range(0, len(words), max_words):
        section = " ".join(words[i:i + max_words])
        sections.append(section)
    return sections


def select_relevant_sections(question, sections, top_n=2):
    question_lower = question.lower()
    scores = []
    for sec in sections:
        score = sum(word in sec.lower() for word in question_lower.split())
        scores.append(score)
    
    top_sections = [sec for score, sec in sorted(zip(scores, sections), reverse=True)[:top_n]]
    return "\n\n".join(top_sections)


def ask_question_with_summary(question):
    
    summary_sections = split_summary_into_sections(Rules_and_Regs_sum, max_words=500)

    
    relevant_text = select_relevant_sections(question, summary_sections)

    
    prompt = f"""Summary:
{relevant_text}

Question:
{question}

Answer concisely in bullet points with numbers and explanations.
Answer clearly in short paragraphs, not in code or markdown.
Use plain text with simple bullet points only if it improves readability.
Keep the answer conversational.
"""

    try:
        
        result = client.chat_completion(
            model="meta-llama/Llama-3.2-1B-Instruct",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that bases answers ONLY on the provided summary."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )

        return result.choices[0].message["content"]

    except Exception as e:
        return f"Error: API request failed ({str(e)})"

# ------------------------ TRACK TIME SIM ------------------------

import math

def simulate_track_time(dragaero, lift_coefficient, frontal_area, car_mass, show_diagnostics=False):
    # ----- track & car params -----
    track_length = 20.0        # meters
    cartridge_mass = 0.008     # kg CO2
    slope_angle_deg = 0.0
    axle_friction_force = 0.005  # N, small axle friction
    nozzle_diameter = 0.002
    dt = 0.001
    air_density = 1.225
    g = 9.81

    # ----- gas constants -----
    R_co2 = 188.9          # specific gas constant, J/(kg·K)
    gamma = 1.3
    T_initial = 293.15     # K
    P_initial = 5.8e6      # Pa
    Cd_nozzle = 0.9
    system_efficiency = 0.18
    delivered_energy = 0.0

   

    nozzle_area = math.pi * (nozzle_diameter / 2.0)**2
    V_cartridge = 0.01
    C_rr = 0.105

    # ----- initial state -----
    position = 0.0
    velocity = 0.0
    time = 0.0
    remaining_co2 = cartridge_mass
    total_mass = car_mass + remaining_co2
    liquid_mass = 0.9 * cartridge_mass
    theta = math.radians(slope_angle_deg)
    TARGET_ENERGY = 333.0  # J
 
    energy_scale = None
    raw_energy = 0

    speeds = []
    positions = []

    thrust_trace = []
    drag_trace = []
    rolling_trace = []
    slope_trace = []
    mdot_trace = []
    pressure_trace = []

    # choked flow factor
    choked_factor = math.sqrt(gamma / (R_co2 * T_initial)) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))
    exit_velocity = math.sqrt((2 * gamma / (gamma + 1)) * R_co2 * T_initial)

    while position < track_length and (velocity > 1e-6 or remaining_co2 > 1e-6):
        positions.append(position)
        speeds.append(velocity)

        # --- pressure & mass flow ---
        if liquid_mass > 0:
            current_pressure = P_initial * (remaining_co2 / cartridge_mass)
            current_pressure = max(current_pressure, 101325.0)
            # mass flow from liquid vaporization
            mdot = Cd_nozzle * nozzle_area * current_pressure * choked_factor
            liquid_mass -= mdot * dt
            if liquid_mass < 0:
                liquid_mass = 0.0
        elif remaining_co2 > 0:
            # ideal gas mass flow
            current_pressure = remaining_co2 * R_co2 * T_initial / V_cartridge
            if current_pressure < 101325:
                current_pressure = 101325
            mdot = Cd_nozzle * nozzle_area * current_pressure * choked_factor
        else:
            current_pressure = 101325.0
            mdot = 0.0

        # --- thrust ---
        raw_thrust = mdot * exit_velocity
        raw_energy += raw_thrust * velocity * dt

        # determine scale factor once
        if energy_scale is None and raw_energy > 0:
            energy_scale = TARGET_ENERGY / raw_energy

        thrust_force = energy_scale * raw_thrust if energy_scale else raw_thrust
        delivered_energy += thrust_force * velocity * dt



        # --- update remaining mass ---
        remaining_co2 -= mdot * dt
        if remaining_co2 < 0:
            remaining_co2 = 0.0
        total_mass = car_mass + remaining_co2

        # --- resistances ---
        drag_force = 0.5 * air_density * velocity**2 * dragaero * frontal_area
        lift_force = 0.5 * air_density * velocity**2 * lift_coefficient * frontal_area
        normal_force = total_mass * g * math.cos(theta) - lift_force
        rolling_force = C_rr * normal_force
        slope_force = total_mass * g * math.sin(theta)
        resist_force = drag_force + rolling_force + slope_force + axle_friction_force

        # --- dynamics ---
        net_force = thrust_force - resist_force
        acceleration = net_force / total_mass
        position += velocity * dt
        velocity += acceleration * dt
        # numerical safety
        if velocity < 1e-12:
            velocity = 0.0

        time += dt

        # --- diagnostics ---
        thrust_trace.append(thrust_force)
        drag_trace.append(drag_force)
        rolling_trace.append(rolling_force)
        slope_trace.append(slope_force)
        mdot_trace.append(mdot)
        pressure_trace.append(current_pressure)

        if time > 60.0:
            break

    diagnostics = {
        "thrust": thrust_trace,
        "drag": drag_trace,
        "rolling": rolling_trace,
        "slope": slope_trace,
        "mdot": mdot_trace,
        "pressure": pressure_trace
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
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            conn = get_pg_conn()
            c = conn.cursor()
            c.execute("INSERT INTO messages (sender, body) VALUES (%s, %s)", (username, message))
            conn.commit()
            conn.close()
        
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
            
            answer = ask_question_with_summary(query)

            
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

    img_speeds = None
    img_forces = None
    
    time = None
    
    drag = None
    lift = None
    frontal_area = None
    message = None
    cross_sections = None

    if request.method == "POST":
        form_id = request.form.get("form_id")

        if form_id == "track_time_calc":
            try:
                drag_co = float(request.form.get("drag_co"))
                lift_co = float(request.form.get("lift_co"))
                mass = float(request.form.get("mass"))
                cross_section = float(request.form.get("cross_section"))

                distances, speeds, time, diag = simulate_track_time(drag_co, lift_co, cross_section, mass, show_diagnostics=True)

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

                buf2 = io.BytesIO()
                plt.figure(figsize=(8,5))
                plt.plot(distances, diag["drag"], label="Drag Force (N)", color="red")
                plt.plot(distances, diag["rolling"], label="Rolling Resistance (N)", color="blue")
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
        time=time,
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





GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder containing app.py
template_path = os.path.join(BASE_DIR, "templates", "mass_email_template.html")

with open(template_path, "r", encoding="utf-8") as f:
    EMAIL_TEMPLATE = f.read()

# File to track previously contacted emails
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
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)

    attachment_data = None
    attachment_name = None
    if attachment and attachment.filename:
        attachment_data = attachment.read()
        attachment_name = attachment.filename

    for name, email in contacts:
        html_body = EMAIL_TEMPLATE.replace("{{name}}", name)

        msg = MIMEMultipart()
        msg["From"] = GMAIL_EMAIL
        msg["To"] = email
        msg["Subject"] = subject

        msg.attach(MIMEText(html_body, "html"))

        if attachment_data:
            part = MIMEApplication(attachment_data, Name=attachment_name)
            part["Content-Disposition"] = f'attachment; filename="{attachment_name}"'
            msg.attach(part)

        server.sendmail(GMAIL_EMAIL, email, msg.as_string())

    server.quit()

@app.route("/mass-email", methods=["GET", "POST"])
def email():
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





    

@app.route("/reactiontime", methods=["GET", "POST"])
def react():
    return render_template("React_Test.html")

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)




