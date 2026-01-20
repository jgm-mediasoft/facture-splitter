import os
import subprocess
import webbrowser
import time
import sys

APP_FILE = "app_prediction_yolo_v26_1_com.py"
PORT = "8501"

cmd = [
    sys.executable,
    "-m", "streamlit",
    "run", APP_FILE,
    "--server.port", PORT,
    "--server.headless", "true"
]

subprocess.Popen(cmd)
time.sleep(3)
webbrowser.open(f"http://localhost:{PORT}")
