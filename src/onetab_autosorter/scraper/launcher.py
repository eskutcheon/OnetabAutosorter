import os
import subprocess
import requests
import time
import signal

JAR_PATH = os.path.abspath("micronaut_scraper/build/libs/micronaut-scraper-all.jar")
print("JAR_PATH:", JAR_PATH)
SERVER_URL = "http://localhost:8080/scrape"


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
