import requests
from typing import List, Dict


SCRAPER_URL = "http://localhost:8080/scrape"


def fetch_summary(url: str) -> str:
    try:
        resp = requests.post(SCRAPER_URL, json={"url": url}, timeout=3)
        if resp.ok:
            return resp.json().get("summary", "")
    except Exception as e:
        print(f"Failed to fetch from scraper service: {e}")
    return ""


# to match expected fetcher signature
def fetch_summary_batch(urls: List[str]) -> Dict[str, str]:
    return fetch_batch(urls)


def fetch_batch(urls: List[str]) -> Dict[str, str]:
    # TODO: add exponential backoff for retries
    try:
        resp = requests.post(f"{SCRAPER_URL}/batch", json={"urls": urls}, timeout=25)
        if resp.ok:
            return {entry["url"]: entry.get("summary", "") for entry in resp.json()}
    except Exception as e:
        print(f"Batch fetch failed: {e}")
    return {}
