# Face Recognition Attendance Web App

A Flask web app for taking attendance with face recognition. Users can open the website, use their browser camera to take a photo, check attendance, register new people, edit saved details, remove people, and view the attendance database.

This project started as a Python terminal attendance system and is being converted into a website.

## Features

- Browser camera capture
- Face recognition attendance checking
- Register new users with name, age, and face image
- View all saved people in the database
- Edit user details
- Remove users
- SQLite database storage
- Flask web interface

## Tech Used

- Python
- Flask
- SQLite
- OpenCV
- face_recognition
- NumPy
- HTML
- CSS
- JavaScript

## Project Structure

```text
AttendanceProject/
├── app.py
├── Main.py
├── Database/
├── Photos/
├── Static/
├── Templates/
└── README.md
```

## How To Run

From the project folder, run:

```powershell
py app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000/
```

Allow camera permission when the browser asks.

## Important GitHub Notes

Before uploading this project to GitHub, avoid committing private data:

- Do not upload real face photos.
- Do not upload a real attendance database.
- Do not upload your `.venv` folder.
- Use a `.gitignore` file to ignore private/generated files.

Recommended files/folders to ignore:

```text
.venv/
__pycache__/
Database/*.db
Photos/*.jpg
Photos/*.png
```

## Future Improvements

- Split the code into separate files for cleaner structure.
- Move HTML into the `Templates` folder.
- Move CSS into the `Static` folder.
- Add login/security for admin actions.
- Add better error messages.
- Add screenshots for the portfolio page.
- Add export options for attendance records.

## Portfolio Description

A Flask-based face recognition attendance system that uses browser camera capture, Python face recognition, and SQLite to register users, check attendance, and manage saved attendance records through a web interface.
