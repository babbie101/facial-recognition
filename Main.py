import sqlite3
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

print("Database file location:", os.path.abspath("Database/attendance.db"))
photos_folder = "Photos"
cap = cv2.VideoCapture(0)

photo_path = os.path.join(photos_folder, "captured_image.jpg")
img = cv2.imread(photo_path)

def init_db(db_path="Database/attendance.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            encoding BLOB,
            name TEXT,
            age INTEGER,
            last_visit TEXT
        )
    """) 
    conn.close()
    print(f"Database initialized at {db_path}")

def get_connection():
    conn = sqlite3.connect("Database/attendance.db")
    return conn
conn = get_connection()

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        encodelist.append(encodings[0])
    return encodelist

def get_info():
    conn = get_connection()
    img_path = os.path.join(photos_folder, "captured_image.jpg")
    img = cv2.imread(img_path)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    if img is None:
        print("Error: Could not read the captured image.")
        return

    now = datetime.now()
    face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(face_rgb)
    if len(face_loc) == 0:
        print("No face detected. Please try again.")
        return
    face_encoding = face_recognition.face_encodings(face_rgb, face_loc)[0]

    cursor = conn.cursor()
    cursor.execute("SELECT id, encoding FROM attendance")
    records = cursor.fetchall()

    match_found = False
    match_id = None
    for record in records:
        stored_id, stored_encoding_bytes = record
        stored_encoding = np.frombuffer(stored_encoding_bytes, dtype=np.float64)
        distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
        if distance < 0.6:
            match_found = True
            match_id = stored_id
            break

    dateString = now.strftime("%Y-%m-%d %H:%M:%S")

    if match_found:
        cursor.execute("UPDATE attendance SET last_visit=? WHERE id=?", (dateString, match_id))
        conn.commit()
        print("Face match found. Attendance updated.")
    else:
        while True:
            name = input("Enter your name: ").strip()
            if name:
                break
            else:
                print("Name cannot be empty. Please enter a valid name.")
        while True:
            try:
                age = int(input("Enter your age: "))
                if age <= 0:
                    print("Please enter a valid positive integer for age.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer for age.")

        SQL = """
        INSERT INTO attendance (image, encoding, name, age, last_visit) 
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(SQL, (sqlite3.Binary(img.tobytes()), sqlite3.Binary(face_encoding.tobytes()), name, age, dateString))
        conn.commit()
        print("Information saved to database")

    cursor.close()
    conn.close()

    os.remove(img_path)

def take_photo():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM attendance")
    rows = cursor.fetchall()
    now = datetime.now()
    dateString = now.strftime("%Y-%m-%d %H:%M:%S")

    known_names = []
    known_encodings = []

    for name, enc_blob in rows:
        enc = np.frombuffer(enc_blob, dtype=np.float64) 
        known_names.append(name)
        known_encodings.append(enc)

    cap =  cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    matched_name = "Unknown"
    while True:
        ret, frame = cap.read()
        imgS = cv2.resize(frame, (0, 0),None,0.25,0.25) #1/4 scale for speed
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        

        facesCurrentFrame = face_recognition.face_locations(imgS)
        encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)
        
        if encodesCurrentFrame:
            current_encoding = encodesCurrentFrame[0]

            matches = face_recognition.compare_faces(known_encodings, current_encoding)
            face_distances = face_recognition.face_distance(known_encodings, current_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                matched_name = known_names[best_match_index]

        if not ret:
            print("Failed to capture image")
            break
        for (top, right, bottom, left) in facesCurrentFrame:
            y1, x2, y2, x1 = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#for rectangle around face
        cv2.imshow("Capture Photo - Press the space key to save or press ESC to cancel", frame)
        key = cv2.waitKey(1)
        if key == 32: #space
            img_name = os.path.join(photos_folder,"captured_image.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Photo saved as {img_name}")
            print("Photo capture complete")
            print(f"Matched person: {matched_name}")
            get_info()
            cv2.destroyAllWindows()
            return img_name
        elif key == 27:  # ESC key to exit
            print("Photo capture cancelled")
            cv2.destroyAllWindows()
            return None

def check_photo():
    img_path = take_photo(save_filename = "check_image.jpg")
    if img_path is None:
        print("No photo captured for checking.")
        return

    
    img = cv2.imread(img_path)
    face_locations = face_recognition.face_locations(img)
    face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(face_rgb)
    if len(face_loc) == 0:
        print("No face detected in the captured image.")   
    face_encoding = face_recognition.face_encodings(face_rgb, face_loc)[0]


    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, encoding, name, age, last_visit FROM attendance")
    records = cursor.fetchall()

    match_found = False
    match_id = None
    match_name = ""
    match_age = ""
    for stored_id, stored_encoding_bytes, name, age, last_visit in records:
        stored_encoding = np.frombuffer(stored_encoding_bytes, dtype=np.float64)
        distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]

    
    face_encdoings = face_recognition.face_encodings(img, face_locations)[0]
    print("Face encodings from captured image:", face_encdoings)

def remove_person_flow():#finish next
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, last_visit FROM attendance")
    print("Remove person flow selected.")
    records = cursor.fetchall()
    print("Current people in the database:")
    for record in records:
        print(f"ID: {record[0]}, Name: {record[1]}, Age: {record[2]}, Last Visit: {record[3]}")
    while True:
        try:
            remove_id = int(input("Enter the ID of the person to remove (or 0 to cancel): "))
            if remove_id == 0:
                print("Removal cancelled.")
                break
            cursor.execute("SELECT * FROM attendance WHERE id=?", (remove_id,))
            person = cursor.fetchone()
            if person:
                cursor.execute("DELETE FROM attendance WHERE id=?", (remove_id,))
                conn.commit()
                print(f"Person with ID {remove_id} has been removed from the database.")
                break
            else:
                print("No person found with that ID. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid integer ID.")
    cursor.close()
    conn.close()

def register_person_flow():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding , last_visit FROM attendance")
    rows = cursor.fetchall()
    now = datetime.now()
    dateString = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Register person flow selected.")
    matched_name = "Unknown"

    known_names = []
    known_encodings = []
    known_last_visits = []

    for name, enc_blob, last_visit in rows:
        enc = np.frombuffer(enc_blob, dtype=np.float64) 
        known_names.append(name)
        known_encodings.append(enc)
        known_last_visits.append(last_visit)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        imgS = cv2.resize(frame, (0, 0),None,0.25,0.25) #1/4 scale for speed
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        

        facesCurrentFrame = face_recognition.face_locations(imgS)
        encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)
        
        if encodesCurrentFrame:
            current_encoding = encodesCurrentFrame[0]

            matches = face_recognition.compare_faces(known_encodings, current_encoding)
            face_distances = face_recognition.face_distance(known_encodings, current_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                matched_name = known_names[best_match_index]
                matched_last_visit = known_last_visits[best_match_index]
                
            else:
                print("New person detected. Please register.")
                add_person_flow()
                break

        if not ret:
            print("Failed to capture image")
            break
        for (top, right, bottom, left) in facesCurrentFrame:
            y1, x2, y2, x1 = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#for rectangle around face
        cv2.imshow("Register Person - Press ESC to exit", frame)
        key = cv2.waitKey(1)
        if key == 32: #space
            img_name = os.path.join(photos_folder,"captured_image.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Photo saved as {img_name}")
            print("Photo capture complete")
            print(f"Welcome back, {matched_name}! Last visit: {matched_last_visit}")
            cursor.execute(
                    "UPDATE attendance SET last_visit=? WHERE name=?", 
                    (dateString, matched_name)   
                )
            conn.commit()
            
            cv2.destroyAllWindows()
            conn.close()
            return img_name
        elif key == 27:  # ESC key to exit
            print("Photo capture cancelled")
            cv2.destroyAllWindows()
            conn.close()
            return None

    pass

def add_person_flow():
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now()

    print("Add person flow selected.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        small_bgr  = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        small_rgb  = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(small_rgb)
        encodesCurrentFrame = face_recognition.face_encodings(small_rgb, facesCurrentFrame)

        for (top, right, bottom, left) in facesCurrentFrame:
            y1, x2, y2, x1 = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Add Person - Press Space bar to capture photo ESC to exit", frame)
        key = cv2.waitKey(1)
        if key == 32: #space
            img_name = os.path.join(photos_folder,"new_person.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Photo saved as {img_name}")
            print("Photo capture complete")
            # Here you can add code to save the new person's info to the database 
            cursor.execute("SELECT id, encoding FROM attendance")
            records = cursor.fetchall()

            while True:
                name = input("Enter the name of the new person: ").strip()
                if name:
                    break
                else:
                    print("Name cannot be empty. Please enter a valid name.")
            while True:
                try:
                    age = int(input("Enter the age of the new person: "))
                    if age <= 0:
                        print("Please enter a valid positive integer for age.")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid integer for age.")

            SQL = """
            INSERT INTO attendance (image, encoding, name, age, last_visit)
            VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(SQL, (sqlite3.Binary(frame.tobytes()),
                                  sqlite3.Binary(encodesCurrentFrame[0].tobytes()), name,
                                    age,
                                     "Never"
                )
            )
            conn.commit()
            print("New person added to the database.")

            cv2.destroyAllWindows()
            return img_name
        elif key == 27:  # ESC key to exit
            print("Photo capture cancelled")
            cv2.destroyAllWindows()
            return None
    pass


def edit_info_flow():
    print("Edit info flow selected.")
    conn = get_connection()
    cursor = conn.cursor()

    # Show current records
    cursor.execute("SELECT id, name, age, last_visit FROM attendance")
    records = cursor.fetchall()
    print("Current people in the database:")
    for record in records:
        print(f"ID: {record[0]}, Name: {record[1]}, Age: {record[2]}, Last Visit: {record[3]}")

    # Pick who to edit
    while True:
        try:
            edit_id = int(input("Enter the ID of the person to edit (or 0 to cancel): "))
            if edit_id == 0:
                print("Edit cancelled.")
                break
            cursor.execute("SELECT * FROM attendance WHERE id=?", (edit_id,))
            person = cursor.fetchone()
            if person:
                # Ask what to change
                new_name = input(f"Enter new name (leave blank to keep '{person[3]}'): ").strip()
                new_age = input(f"Enter new age (leave blank to keep '{person[4]}'): ").strip()

                if new_name:
                    cursor.execute("UPDATE attendance SET name=? WHERE id=?", (new_name, edit_id))
                if new_age:
                    cursor.execute("UPDATE attendance SET age=? WHERE id=?", (int(new_age), edit_id))

                conn.commit()
                print("Person's info updated successfully.")
                break
            else:
                print("No person found with that ID. Please try again.")
        except ValueError:
            print("Invalid input. Please try again.")

    cursor.close()
    conn.close()

def view_database_flow():
    print("View database flow selected.\n\n")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, last_visit FROM attendance")
    records = cursor.fetchall()
    print("Current people in the database:")
    for record in records:
        print(f"ID: {record[0]}, Name: {record[1]}, Age: {record[2]}, Last Visit: {record[3]}")
        home_screen()
    cursor.close()
    conn.close()

def home_screen():
    while True:
        print("\n--- Attendance System ---")
        print("1. Check Attendance")
        print("2. Update database")  # add and remove user
        print("3. Look at database")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            register_person_flow()
            pass

        elif choice == '2':
            print("Update database selected.")
            print("Do you want to add or remove a user?")
            print("1. Add user")
            print("2. Remove user")
            print("3.edit info")
            print("cancel. Go back to main menu(press any other key and enter)")
            sub_choice = input("Enter your choice (1-2): ").strip()

            if sub_choice == '1':
                register_person_flow()
                pass
            elif sub_choice == '2':
                remove_person_flow()
                pass
            elif sub_choice == '3':
                edit_info_flow()
                pass
            else:
                home_screen()

        elif choice == '3':
            view_database_flow()
            break

        else:
            print("Invalid choice. Please try again.")

home_screen()
conn = get_connection()
cursor = conn.cursor()