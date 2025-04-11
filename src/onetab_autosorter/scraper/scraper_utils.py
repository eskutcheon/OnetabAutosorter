"""
    Utility functions for fetching and parsing HTML content from URLs with Python.
    Java microservice scraper functions located in scraper/client.py
"""

import html
import requests
import urllib3.exceptions
import warnings
import time
from tqdm.auto import tqdm
from itertools import islice
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, Tag
from typing import List, Dict
#from onetab_autosorter.utils.utils import preprocess_html_text



#? not sure if I'm going to be using this at all
def strip_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """ Remove script, style, noscript tags and other non-visible elements from the BeautifulSoup object """
    #! literally just does the same thing as soup.get_text(), but might be worthwhile later
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = str(soup.prettify(formatter="minimal")).splitlines()
    text = ""
    for line in lines:
        if not line.strip().startswith("<"):
            text += line.strip() + "\n"
    return text


#& may keep this here or move it to the general TextCleaningFilter class later
def fetch_full_text(soup: BeautifulSoup, max_tokens: int = 500) -> str:
    """ Extract raw visible text from the full HTML page, ignoring scripts/styles """
    # use islice over the iterator of significant strings to limit the number of tokens returned
    text = list(islice(soup.stripped_strings, 0, max_tokens))
    #text = soup.get_text(separator="\n", strip=True).split()
    #text_len = len(text)
    return " ".join(text) #[:min(text_len, max_tokens)])  # split into tokens and limit to max_tokens

################################ requests-related functions ################################

def default_html_fetcher(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=5)
    # Try to use correct encoding
    encoding = resp.apparent_encoding or 'utf-8'
    content_type = resp.headers.get("Content-Type", "").lower()
    # allow XML parsing
    parser_type = "xml" if "xml" in content_type else "html.parser"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        # try-catch handles any garbage encoding that `requests` mis-detects
        try:
            decoded = resp.content.decode(encoding, errors="replace")
        except Exception as e:
            decoded = resp.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(decoded, parser_type)
    # other metadata tags used for previews
    #& if moving to using the filtering classes, this should probably just pass back a tuple of (soup, metadata)
        # then text can be extracted upstream in the main processing pipeline
    return fetch_full_text(soup)


def write_failed_fetch_log(url: str, error: str):
    # TODO: extract this to a more global error logging mechanism eventually
    FAILED_FETCH_LOG = "output/failed_fetches.log"
    with open(FAILED_FETCH_LOG, "a", encoding="utf-8", errors="ignore") as fptr:
        # Log the failed URL to a file for later review
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # encoding the errors to handle any non-UTF-8 characters gracefully - have to do a lot of extra work because Windows sucks and seems to default back to ANSI (cp1252)
        safe_url = html.escape(url) # url.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        safe_error = html.escape(str(error)) # str(error).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        log_entry = f"[{timestamp}] {safe_url}\n\tERROR: \"{safe_error}\"\n"
        try:
            fptr.write(log_entry)
        except Exception as log_error: # last-ditch fallback using ASCII
            fallback = log_entry.encode("ascii", errors="replace").decode("ascii", errors="replace")
            fptr.write(fallback)




def safe_fetch(url: str, attempt=1, max_retries = 3, retry_delay = 2) -> str:
    try:
        text = default_html_fetcher(url)
        #? NOTE: apparently default_html_fetcher_batch is the only function calling this now so I'm just preparing by at least making a flag for it
        return text
        # TODO: add minor preprocessing here to include it in the parallel execution
    except Exception as e:
        if isinstance(e, (requests.exceptions.RequestException, urllib3.exceptions.ReadTimeoutError)):
            if attempt <= max_retries:
                # using exponential backoff for retry delay
                time.sleep(retry_delay * attempt)
                return safe_fetch(url, attempt + 1)
            print(f"\t[Timeout] Failed to fetch after {attempt} attempts: {url}")
        write_failed_fetch_log(url, str(e))
    return ""

# TODO: (General) add support for dynamic sites with Selenium webdriver or similar tools


def default_html_fetcher_batch(urls: List[str], max_workers = 10) -> Dict[str, str]:
    if isinstance(urls, str):
        #print("WARNING: default_html_fetcher_batch received a single URL string instead of a list; using single URL.")
        return {urls: safe_fetch(urls, preprocess=True)}
    from concurrent.futures import as_completed, Future, ThreadPoolExecutor #ProcessPoolExecutor
    #? NOTE: rewrote this in a more explicit way to use a progress bar that updates properly
    #? Since requests releases the GIL lock during I/O, the ThreadPoolExecutor should be faster in this case.
    with tqdm(total = len(urls), colour="cyan", desc="Concurrently fetching webpage contents") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict[Future, str] = {executor.submit(safe_fetch, url): url for url in urls}
            results = {} # instantiate here so that it's visible to all processes
            for future in as_completed(futures):
                url = futures[future]
                results[url] = future.result()
                pbar.update()
    #? NOTE: the url -> text dictionary doesn't preserve order of the input urls, but that could be added easily if needed
    return results