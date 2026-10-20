"""Chạy app dạng desktop: mở FastAPI server ở thread riêng,
hiển thị UI trong 1 cửa sổ native (pywebview) thay vì mở browser.

Usage: python desktop.py
"""
import os
import threading
import time
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import webview

from app.main import app

HOST = "127.0.0.1"
PORT = 8000


def run_server():
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def wait_until_ready(timeout=30):
    url = f"http://{HOST}:{PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    if not wait_until_ready():
        raise RuntimeError("Server không khởi động được trong 30s")

    webview.create_window(
        "Melanoma Classifier Demo",
        f"http://{HOST}:{PORT}",
        width=1000,
        height=800,
    )
    webview.start()


if __name__ == "__main__":
    main()
