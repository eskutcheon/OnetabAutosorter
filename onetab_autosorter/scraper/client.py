import os
import subprocess
import requests
import time
#import signal
from functools import wraps
import atexit
from typing import Callable, List, Dict




SERVER_URL = "http://localhost:8080/scrape"
JAR_PATH = os.path.abspath("micronaut_scraper/build/libs/micronaut-scraper-all.jar")
print("JAR_PATH:", JAR_PATH)


def fetch_summary(url: str) -> str:
    try:
        resp = requests.post(SERVER_URL, json={"url": url}, timeout=3)
        if resp.ok:
            return resp.json().get("summary", "")
    except Exception as e:
        print(f"Failed to fetch from scraper service: {e}")
    return ""


# to match expected fetcher signature
def fetch_summary_batch(urls: List[str]) -> Dict[str, str]:
    return fetch_batch(urls)


def fetch_batch(urls: List[str]) -> Dict[str, str]:
    # TODO: add exponential backoff for retries to the Java files
    try:
        resp = requests.post(f"{SERVER_URL}/batch", json={"urls": urls}, timeout=25)
        if resp.ok:
            return {entry["url"]: entry.get("summary", "") for entry in resp.json()}
    except Exception as e:
        print(f"Batch fetch failed: {e}")
    return {}





class ScraperServiceManager:
    def __init__(self, jar_path=JAR_PATH, port=8080):
        self.jar_path = jar_path
        self.port = port
        self.process = None

    def is_running(self) -> bool:
        try:
            resp = requests.get(SERVER_URL, timeout=1)
            return resp.status_code < 500
        except Exception:
            return False

    def start(self):
        if self.is_running():
            #print("✅ Scraper service already running.")
            print("Scraper service already running.")
            return
        # print("🚀 Starting Java microservice...")
        print("Starting Java microservice...")
        self.process = subprocess.Popen(
            ["java", "-jar", self.jar_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        for _ in range(15):
            if self.is_running():
                # print("✅ Scraper is up.")
                print("Scraper is up.")
                return
            time.sleep(0.5)
        raise RuntimeError("Failed to start scraper service.")
        # raise RuntimeError("❌ Failed to start scraper service.")

    def stop(self):
        if self.process:
            # print("🛑 Shutting down scraper service...")
            print("Shutting down scraper service...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing scraper process...")
                #print("⚠ Force killing scraper process...")
                self.process.kill()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


    def fetch_within_context(self, fetcher_fn: Callable):
        """ allows interfacing with the Java-based scraper service within a context manager in the same way as the Python implementations """
        self.start()  # start the service
        atexit.register(self.stop)  # ensure the service is stopped when the program exits
        #~ could register the self.stop() call with atexit.register, but it may stay open longer than necessary
        @wraps(fetcher_fn)
        def wrapper(*args, **kwargs):
            return fetcher_fn(*args, **kwargs)
        return wrapper