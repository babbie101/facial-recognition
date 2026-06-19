import base64
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from flask import Flask, jsonify, redirect, render_template_string, request, url_for


app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "Database"
PHOTOS_DIR = BASE_DIR / "Photos"
DB_PATH = DATABASE_DIR / "attendance.db"
MATCH_THRESHOLD = 0.6


def init_db():
    DATABASE_DIR.mkdir(exist_ok=True)
    PHOTOS_DIR.mkdir(exist_ok=True)

    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            encoding BLOB,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            last_visit TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def current_time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decode_camera_image(image_data):
    if not image_data or "," not in image_data:
        raise ValueError("No valid camera image was sent.")

    header, encoded_image = image_data.split(",", 1)

    if "image/jpeg" not in header and "image/png" not in header:
        raise ValueError("The camera image must be JPEG or PNG.")

    image_bytes = base64.b64decode(encoded_image)
    np_image = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("The camera image could not be opened.")

    return image_bytes, frame


def get_face_encoding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        raise ValueError("No face was found. Try again with your face clearly in view.")

    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not encodings:
        raise ValueError("The face was found, but it could not be encoded.")

    return encodings[0]


def get_people():
    conn = get_connection()
    people = conn.execute(
        "SELECT id, name, age, last_visit FROM attendance ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return people


def get_person(person_id):
    conn = get_connection()
    person = conn.execute(
        "SELECT id, name, age, last_visit FROM attendance WHERE id = ?",
        (person_id,),
    ).fetchone()
    conn.close()
    return person


def find_matching_person(face_encoding):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, age, last_visit, encoding FROM attendance WHERE encoding IS NOT NULL"
    ).fetchall()
    conn.close()

    best_match = None
    best_distance = None

    for row in rows:
        stored_encoding = np.frombuffer(row["encoding"], dtype=np.float64)
        distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_match = row

    if best_match is not None and best_distance is not None and best_distance < MATCH_THRESHOLD:
        return best_match, float(best_distance)

    return None, None


def save_new_person(name, age, image_bytes, face_encoding):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO attendance (image, encoding, name, age, last_visit)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            sqlite3.Binary(image_bytes),
            sqlite3.Binary(face_encoding.tobytes()),
            name,
            age,
            "Never",
        ),
    )
    conn.commit()
    conn.close()


def update_last_visit(person_id):
    visit_time = current_time_string()
    conn = get_connection()
    conn.execute(
        "UPDATE attendance SET last_visit = ? WHERE id = ?",
        (visit_time, person_id),
    )
    conn.commit()
    conn.close()
    return visit_time


PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Attendance System</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f4f7fb;
            --panel: #ffffff;
            --line: #d8e0ea;
            --text: #162033;
            --muted: #5d6b7c;
            --blue: #2563eb;
            --blue-dark: #1d4ed8;
            --red: #dc2626;
            --green: #047857;
            --amber: #b45309;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: Arial, sans-serif;
        }

        main {
            width: min(1120px, calc(100% - 32px));
            margin: 28px auto 48px;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 20px;
        }

        h1 {
            margin: 0;
            font-size: 30px;
        }

        h2 {
            margin: 0 0 14px;
            font-size: 20px;
        }

        .menu-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 18px;
        }

        .menu-option {
            display: grid;
            gap: 8px;
            min-height: 128px;
            padding: 16px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: white;
            color: var(--text);
            text-align: left;
            cursor: pointer;
        }

        .menu-option:hover {
            border-color: var(--blue);
        }

        .menu-number {
            width: 34px;
            height: 34px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #e8eef6;
            color: var(--blue-dark);
            font-weight: bold;
        }

        .menu-title {
            font-size: 18px;
            font-weight: bold;
        }

        .menu-text {
            color: var(--muted);
            line-height: 1.4;
        }

        .screen {
            display: none;
        }

        .screen.active {
            display: block;
        }

        .screen-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 14px;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
            gap: 18px;
            align-items: start;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
        }

        video, canvas {
            width: 100%;
            aspect-ratio: 4 / 3;
            background: #111827;
            border-radius: 6px;
            object-fit: cover;
        }

        canvas {
            display: none;
        }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 12px;
        }

        button, .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 38px;
            border: 0;
            border-radius: 6px;
            padding: 9px 13px;
            background: var(--blue);
            color: white;
            font: inherit;
            text-decoration: none;
            cursor: pointer;
        }

        button:hover, .button:hover {
            background: var(--blue-dark);
        }

        button.secondary, .button.secondary {
            background: #475569;
        }

        button.danger, .button.danger {
            background: var(--red);
        }

        button:disabled {
            background: #9aa8b7;
            cursor: not-allowed;
        }

        .status {
            min-height: 26px;
            margin-top: 12px;
            color: var(--muted);
        }

        .status.success {
            color: var(--green);
        }

        .status.warning {
            color: var(--amber);
        }

        .status.error {
            color: var(--red);
        }

        form.stack {
            display: grid;
            gap: 12px;
        }

        label {
            display: grid;
            gap: 5px;
            color: var(--muted);
        }

        input {
            min-height: 38px;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 8px 10px;
            font: inherit;
            color: var(--text);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            overflow: hidden;
            border-radius: 8px;
        }

        th, td {
            border-bottom: 1px solid var(--line);
            padding: 11px;
            text-align: left;
            vertical-align: middle;
        }

        th {
            background: #e8eef6;
            color: #26364d;
        }

        td form {
            display: inline;
        }

        .record-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .muted {
            color: var(--muted);
        }

        @media (max-width: 820px) {
            header {
                align-items: flex-start;
                flex-direction: column;
            }

            .menu-grid {
                grid-template-columns: 1fr;
            }

            .screen-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .layout {
                grid-template-columns: 1fr;
            }

            table {
                display: block;
                overflow-x: auto;
                white-space: nowrap;
            }
        }
    </style>
</head>
<body>
    <main>
        <header>
            <div>
                <h1>Attendance System</h1>
                <div class="muted">Choose an option like the old Python menu.</div>
            </div>
            <a class="button secondary" href="{{ url_for('home') }}">Refresh</a>
        </header>

        <section id="menuScreen" class="screen active">
            <div class="menu-grid">
                <button class="menu-option" type="button" data-screen="checkScreen">
                    <span class="menu-number">1</span>
                    <span class="menu-title">Check Attendance</span>
                    <span class="menu-text">Open the camera, take a photo, and mark a known person as visited.</span>
                </button>

                <button class="menu-option" type="button" data-screen="updateScreen">
                    <span class="menu-number">2</span>
                    <span class="menu-title">Update Database</span>
                    <span class="menu-text">Add a new person with a photo, edit details, or remove a record.</span>
                </button>

                <button class="menu-option" type="button" data-screen="databaseScreen">
                    <span class="menu-number">3</span>
                    <span class="menu-title">Look At Database</span>
                    <span class="menu-text">View everyone currently saved in the attendance database.</span>
                </button>
            </div>
        </section>

        <section id="checkScreen" class="screen panel">
            <div class="screen-header">
                <h2>1. Check Attendance</h2>
                <button class="secondary" type="button" data-screen="menuScreen">Back To Menu</button>
            </div>

            <video id="camera" autoplay playsinline></video>
            <canvas id="snapshot"></canvas>

            <div class="actions">
                <button id="startButton" type="button">Start Camera</button>
                <button id="takeButton" type="button" disabled>Take Photo</button>
                <button id="checkButton" type="button" disabled>Check Attendance</button>
                <button id="stopButton" class="secondary" type="button" disabled>Stop Camera</button>
            </div>

            <div id="cameraStatus" class="status">Camera is off.</div>
        </section>

        <section id="updateScreen" class="screen">
            <div class="screen-header">
                <h2>2. Update Database</h2>
                <button class="secondary" type="button" data-screen="menuScreen">Back To Menu</button>
            </div>

            <div class="layout">
                <div class="panel">
                    <h2>Add User</h2>
                    <p class="muted">Use the camera screen first, take a photo, then return here to save the new person.</p>
                    <form id="registerForm" class="stack">
                        <label>
                            Name
                            <input id="nameInput" name="name" required>
                        </label>
                        <label>
                            Age
                            <input id="ageInput" name="age" type="number" min="1" required>
                        </label>
                        <button id="registerButton" type="submit" disabled>Register With Photo</button>
                    </form>
                    <div id="registerStatus" class="status">Take a photo before registering.</div>
                </div>

                <div class="panel">
                    <h2>Edit Or Remove User</h2>
                    {% if people %}
                        <table>
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Age</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for person in people %}
                                    <tr>
                                        <td>{{ person.name }}</td>
                                        <td>{{ person.age }}</td>
                                        <td>
                                            <div class="record-actions">
                                                <a class="button secondary" href="{{ url_for('edit_person', person_id=person.id) }}">Edit</a>
                                                <form method="post" action="{{ url_for('delete_person', person_id=person.id) }}">
                                                    <button class="danger" type="submit">Remove</button>
                                                </form>
                                            </div>
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    {% else %}
                        <p class="muted">No people have been registered yet.</p>
                    {% endif %}
                </div>
            </div>
        </section>

        <section id="databaseScreen" class="screen panel">
            <div class="screen-header">
                <h2>3. Look At Database</h2>
                <button class="secondary" type="button" data-screen="menuScreen">Back To Menu</button>
            </div>

            {% if people %}
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Age</th>
                            <th>Last Visit</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for person in people %}
                            <tr>
                                <td>{{ person.id }}</td>
                                <td>{{ person.name }}</td>
                                <td>{{ person.age }}</td>
                                <td>{{ person.last_visit or "Never" }}</td>
                                <td>
                                    <div class="record-actions">
                                        <a class="button secondary" href="{{ url_for('edit_person', person_id=person.id) }}">Edit</a>
                                        <form method="post" action="{{ url_for('delete_person', person_id=person.id) }}">
                                            <button class="danger" type="submit">Delete</button>
                                        </form>
                                    </div>
                                </td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p class="muted">No people have been registered yet.</p>
            {% endif %}
        </section>
    </main>

    <script>
        const video = document.getElementById("camera");
        const canvas = document.getElementById("snapshot");
        const startButton = document.getElementById("startButton");
        const takeButton = document.getElementById("takeButton");
        const checkButton = document.getElementById("checkButton");
        const stopButton = document.getElementById("stopButton");
        const registerButton = document.getElementById("registerButton");
        const registerForm = document.getElementById("registerForm");
        const nameInput = document.getElementById("nameInput");
        const ageInput = document.getElementById("ageInput");
        const cameraStatus = document.getElementById("cameraStatus");
        const registerStatus = document.getElementById("registerStatus");
        const screenButtons = document.querySelectorAll("[data-screen]");
        const screens = document.querySelectorAll(".screen");

        let stream = null;
        let latestImage = null;

        function showScreen(screenId) {
            screens.forEach((screen) => {
                screen.classList.toggle("active", screen.id === screenId);
            });

            if (screenId === "updateScreen" && latestImage) {
                setMessage(registerStatus, "Photo ready for registration.", "success");
            }
        }

        function setMessage(element, message, type = "") {
            element.textContent = message;
            element.className = `status ${type}`.trim();
        }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });

                video.srcObject = stream;
                startButton.disabled = true;
                takeButton.disabled = false;
                checkButton.disabled = false;
                stopButton.disabled = false;
                setMessage(cameraStatus, "Camera is ready.", "success");
            } catch (error) {
                setMessage(cameraStatus, "Could not open the camera. Allow camera permission and try again.", "error");
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
            }

            stream = null;
            video.srcObject = null;
            startButton.disabled = false;
            takeButton.disabled = true;
            checkButton.disabled = true;
            stopButton.disabled = true;
            setMessage(cameraStatus, "Camera is off.");
        }

        function takePhoto() {
            if (!video.videoWidth || !video.videoHeight) {
                setMessage(cameraStatus, "Camera is not ready yet.", "warning");
                return null;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            latestImage = canvas.toDataURL("image/jpeg", 0.9);
            registerButton.disabled = false;
            setMessage(cameraStatus, "Photo captured. You can check attendance or register it.", "success");
            setMessage(registerStatus, "Photo ready for registration.", "success");
            return latestImage;
        }

        async function sendJson(url, body) {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(body)
            });

            return response.json();
        }

        async function checkAttendance() {
            const image = takePhoto();

            if (!image) {
                return;
            }

            setMessage(cameraStatus, "Checking attendance...");
            const result = await sendJson("{{ url_for('check_attendance') }}", { image });

            if (result.success && result.matched) {
                setMessage(
                    cameraStatus,
                    `Welcome back, ${result.person.name}. Attendance updated at ${result.last_visit}.`,
                    "success"
                );
                window.setTimeout(() => window.location.reload(), 900);
            } else if (result.success) {
                setMessage(cameraStatus, "No matching person found. Fill in the form to register them.", "warning");
            } else {
                setMessage(cameraStatus, result.error || "Attendance check failed.", "error");
            }
        }

        async function registerPerson(event) {
            event.preventDefault();

            if (!latestImage) {
                setMessage(registerStatus, "Take a photo first.", "warning");
                return;
            }

            setMessage(registerStatus, "Registering person...");
            const result = await sendJson("{{ url_for('register_person') }}", {
                image: latestImage,
                name: nameInput.value,
                age: ageInput.value
            });

            if (result.success) {
                setMessage(registerStatus, `${result.name} was registered.`, "success");
                window.setTimeout(() => window.location.reload(), 900);
            } else {
                setMessage(registerStatus, result.error || "Registration failed.", "error");
            }
        }

        startButton.addEventListener("click", startCamera);
        takeButton.addEventListener("click", takePhoto);
        checkButton.addEventListener("click", checkAttendance);
        stopButton.addEventListener("click", stopCamera);
        registerForm.addEventListener("submit", registerPerson);
        screenButtons.forEach((button) => {
            button.addEventListener("click", () => showScreen(button.dataset.screen));
        });
    </script>
</body>
</html>
"""


EDIT_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Edit Person</title>
    <style>
        body {
            margin: 0;
            background: #f4f7fb;
            color: #162033;
            font-family: Arial, sans-serif;
        }

        main {
            width: min(520px, calc(100% - 32px));
            margin: 40px auto;
            background: white;
            border: 1px solid #d8e0ea;
            border-radius: 8px;
            padding: 18px;
        }

        h1 {
            margin: 0 0 18px;
            font-size: 26px;
        }

        form {
            display: grid;
            gap: 12px;
        }

        label {
            display: grid;
            gap: 5px;
            color: #5d6b7c;
        }

        input {
            min-height: 38px;
            border: 1px solid #d8e0ea;
            border-radius: 6px;
            padding: 8px 10px;
            font: inherit;
        }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 6px;
        }

        button, a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 38px;
            border: 0;
            border-radius: 6px;
            padding: 9px 13px;
            background: #2563eb;
            color: white;
            font: inherit;
            text-decoration: none;
            cursor: pointer;
        }

        a {
            background: #475569;
        }
    </style>
</head>
<body>
    <main>
        <h1>Edit Person</h1>
        <form method="post">
            <label>
                Name
                <input name="name" value="{{ person.name }}" required>
            </label>
            <label>
                Age
                <input name="age" type="number" min="1" value="{{ person.age }}" required>
            </label>
            <div class="actions">
                <button type="submit">Save Changes</button>
                <a href="{{ url_for('home') }}">Cancel</a>
            </div>
        </form>
    </main>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE, people=get_people())


@app.route("/api/check-attendance", methods=["POST"])
def check_attendance():
    try:
        data = request.get_json()
        image_bytes, frame = decode_camera_image(data.get("image") if data else None)
        face_encoding = get_face_encoding(frame)
        person, distance = find_matching_person(face_encoding)

        if person is None:
            return jsonify({"success": True, "matched": False})

        last_visit = update_last_visit(person["id"])
        return jsonify(
            {
                "success": True,
                "matched": True,
                "distance": distance,
                "last_visit": last_visit,
                "person": {
                    "id": person["id"],
                    "name": person["name"],
                    "age": person["age"],
                },
            }
        )
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/api/register", methods=["POST"])
def register_person():
    try:
        data = request.get_json()

        if not data:
            raise ValueError("No registration data was sent.")

        name = data.get("name", "").strip()
        age_text = data.get("age", "").strip()

        if not name:
            raise ValueError("Name is required.")

        if not age_text.isdigit() or int(age_text) <= 0:
            raise ValueError("Age must be a positive number.")

        image_bytes, frame = decode_camera_image(data.get("image"))
        face_encoding = get_face_encoding(frame)
        existing_person, _distance = find_matching_person(face_encoding)

        if existing_person is not None:
            raise ValueError(f"This face already looks like {existing_person['name']}.")

        save_new_person(name, int(age_text), image_bytes, face_encoding)
        return jsonify({"success": True, "name": name})
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/edit/<int:person_id>", methods=["GET", "POST"])
def edit_person(person_id):
    person = get_person(person_id)

    if person is None:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form["name"].strip()
        age_text = request.form["age"].strip()

        if name and age_text.isdigit() and int(age_text) > 0:
            conn = get_connection()
            conn.execute(
                "UPDATE attendance SET name = ?, age = ? WHERE id = ?",
                (name, int(age_text), person_id),
            )
            conn.commit()
            conn.close()

        return redirect(url_for("home"))

    return render_template_string(EDIT_PAGE, person=person)


@app.route("/delete/<int:person_id>", methods=["POST"])
def delete_person(person_id):
    conn = get_connection()
    conn.execute("DELETE FROM attendance WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("home"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
