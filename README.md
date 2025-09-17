CyberGuard Toolbox
==================

Beginner-friendly local web app that teaches and helps users apply basic cybersecurity practices.

Features
--------
- Secure Password Generator
- QR Code Link Scanner
- EXIF Metadata Viewer
- Simple Port Scanner (placeholder; coming soon)

Tech Stack
----------
- Python (Flask)
- Bootstrap 5 (via CDN)

Setup (Windows PowerShell)
--------------------------

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
$env:FLASK_SECRET_KEY = "dev-secret"
python app.py
# Open http://127.0.0.1:5000
```

Notes
-----
- QR scanning uses OpenCV QRCodeDetector and runs locally.
- EXIF viewing uses Pillow; some image formats may not contain EXIF.
- Port scanner is intentionally deferred to avoid encouraging aggressive scanning; a safe, limited version will be added later.

Security
--------
- This app is for educational, local use. Do not deploy publicly without hardening.
- File uploads are processed in-memory and not stored on disk.


