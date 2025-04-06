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
from typing import Protocol, List, Dict
from thefuzz import fuzz
from onetab_autosorter.utils.clean_utils import preprocess_html_text


# just something I was going to try
class SupplementFetcher(Protocol):
    def __call__(self, url: str) -> str: ...



def is_similar_text(text1: str, text2: str, threshold: int = 60) -> bool:
    """ test whether (the start of) two text strings are roughly similar by their Levenshtein similarity ratio """
    # limit text to the first 50 characters for comparison to avoid long texts skewing the similarity (and for speed)
    if not text1 or not text2:
        return False
    MAX_CHARS = 100
    text1_partial = text1[min(len(text1), MAX_CHARS):].strip()
    text2_partial = text2[min(len(text2), MAX_CHARS):].strip()
    # might also try partial_ratio or token_sort_ratio later
    return fuzz.ratio(text1_partial, text2_partial) >= threshold


def latex_remover(tag: Tag) -> bool:
    # Example: remove tags that might store LaTeX code
    # or if 'class' in tag.attrs and 'latex' in tag['class'] ...
    if tag.name == "span" and "latex" in (tag.get("class") or []):
        return True
    return False


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


def fetch_full_text(soup: BeautifulSoup, max_tokens: int = 1000) -> str:
    """ Extract raw visible text from the full HTML page, ignoring scripts/styles """
    # first remove script and style tags
    # for tag in soup(["script", "style", "noscript"]):
    #     tag.decompose()
    # remove_tags = ["script", "style", "noscript"]
    # remove_selectors=["footer", "aside", "nav", ".footnote", "sup.reference", "span.citation", "..."]
    # for tagname in remove_tags:
    #     for t in soup.find_all(tagname):
    #         t.decompose()
    # # 2) Remove elements via CSS selectors if any
    # for sel in remove_selectors:
    #     for t in soup.select(sel):
    #         t.decompose()
    # # iterate all tags in BFS or DFS style
    # for t in soup.find_all():
    #     if latex_remover(t):
    #         t.decompose()
    #text = list(islice(soup.stripped_strings, 0, max_tokens))
    # for element in soup.stripped_strings:
    #     text += element + " "
    text = soup.get_text(separator="\n", strip=True).split()
    text_len = len(text)
    return " ".join(text[:min(text_len, max_tokens)])  # split into tokens and limit to max_tokens

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
        #soup = BeautifulSoup(resp.content.decode(encoding, errors="replace")[:4096], parser_type)
        # try-catch handles any garbage encoding that `requests` mis-detects
        try:
            decoded = resp.content.decode(encoding, errors="replace")
        except Exception as e:
            decoded = resp.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(decoded, parser_type)
    # other metadata tags used for previews
    raw_text = fetch_full_text(soup)
    return raw_text #preprocess_html_text(raw_text)


def write_failed_fetch_log(url: str, error: str):
    # TODO: extract this to a more global error logging mechanism eventually
    FAILED_FETCH_LOG = "output/failed_fetches.log"
    with open(FAILED_FETCH_LOG, "a", encoding="utf-8", errors="ignore") as fptr:
        # Log the failed URL to a file for later review
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # encoding the errors to handle any non-UTF-8 characters gracefully
        # we have to do a lot of extra work because Windows sucks and seems to default back to ANSI (cp1252)
        #safe_error = str(error).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        #safe_url = url.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        safe_url = html.escape(url)
        safe_error = html.escape(str(error))
        log_entry = f"[{timestamp}] {safe_url}\n\tERROR: \"{safe_error}\"\n"
        try:
            fptr.write(log_entry)
        except Exception as log_error: # last-ditch fallback using ASCII
            fallback = log_entry.encode("ascii", errors="replace").decode("ascii", errors="replace")
            fptr.write(fallback)


# might add blacklisting later, but for now I'm hoping this refactor catches it all
# BLACKLIST_DOMAINS = ["google.com", "twitter.com"]

# def is_blacklisted(url: str) -> bool:
#     return any(domain in url for domain in BLACKLIST_DOMAINS)


def safe_fetch(url: str, attempt=1, max_retries = 3, retry_delay = 2, preprocess = True) -> str:
    try:
        text = default_html_fetcher(url)
        #? NOTE: apparently default_html_fetcher_batch is the only function calling this now so I'm just preparing by at least making a flag for it
        if preprocess:
            #!!! FIXME: seemingly returns lists without preprocess - need to iron out the cleaning utils
            text = preprocess_html_text(text)
        return text
        # TODO: add preprocessing here to include it in the parallel execution
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


def default_html_fetcher_batch(urls: List[str], max_workers = 4) -> Dict[str, str]:
    if isinstance(urls, str):
        #print("WARNING: default_html_fetcher_batch received a single URL string instead of a list; using single URL.")
        return {urls: preprocess_html_text(safe_fetch(urls))}
    from concurrent.futures import as_completed, Future, ProcessPoolExecutor #ThreadPoolExecutor
    # TODO: might need to try and see if the ProcessPoolExecutor works properly instead - could be a GIL issue since a lot is done within the functions
    #? NOTE: rewrote this in a more explicit way to use a progress bar that updates properly
    # TODO: need to add rate-limiting to domains, especially if we're fetching a lot of URLs concurrently and with domains grouped initially
    with tqdm(total = len(urls), colour="cyan", desc="Concurrently fetching webpage contents") as pbar:
        #with ThreadPoolExecutor(max_workers=10) as executor:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict[Future, str] = {executor.submit(safe_fetch, url): url for url in urls}
            results = {}
            for future in as_completed(futures):
                url = futures[future]
                results[url] = future.result()
                pbar.update()
    #? NOTE: the url -> text dictionary doesn't preserve order of the input urls, but that could be added easily if needed
    return results
