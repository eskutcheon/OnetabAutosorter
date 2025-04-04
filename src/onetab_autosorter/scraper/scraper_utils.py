"""
    Utility functions for fetching and parsing HTML content from URLs with Python.
    Java microservice scraper functions located in scraper/client.py
"""

import html
import requests
import urllib3.exceptions
import warnings
import time
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from typing import Protocol, List, Dict
from thefuzz import fuzz
from onetab_autosorter.text_cleaning import preprocess_html_text


# just something I was going to try
class SupplementFetcher(Protocol):
    def __call__(self, url: str) -> str: ...



def is_similar_text(text1: str, text2: str, threshold: int = 60) -> bool:
    """ test whether (the start of) two text strings are roughly similar by their Levenshtein similarity ratio """
    # limit text to the first 50 characters for comparison to avoid long texts skewing the similarity (and for speed)
    if not text1 or not text2:
        return False
    MAX_CHARS = 50
    text1_partial = text1[min(len(text1), MAX_CHARS):].strip()
    text2_partial = text2[min(len(text2), MAX_CHARS):].strip()
    # might also try partial_ratio or token_sort_ratio later
    return fuzz.ratio(text1_partial, text2_partial) >= threshold



def fetch_full_text(soup: BeautifulSoup, max_tokens: int = 10000) -> str:
    """ Extract raw visible text from the full HTML page, ignoring scripts/styles """
    # first remove script and style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text.split()[:max_tokens]  # split into tokens and limit to max_tokens



# # TODO: I'm probably about to remove this entirely and start extracting all text from the HTML, then stripping
#     # the boilerplate with a smallish NLP model or heuristic. I'm getting too much garbage otherwise
# def fetch_from_tags(soup: BeautifulSoup) -> str:
#     """ attempt to extract the most relevant text from the page as supplemental text """
#     tag_chunks = []
#     # TODO: consider referencing a dictionary of domains to skip descriptions, etc. (primarily duckduckgo's "privacy simplified" messages)
#     meta_names = [
#         {"name": "description"},
#         {"property": "og:description"},
#         #{"name": "twitter:description"}
#     ]
#     # append the title tag if it exists and has text
#     title_tag = soup.find("title")
#     if title_tag and (title_text := title_tag.get_text(strip=True)):
#         # TODO: make a more structured way to check these tags since I want to compare the title with descriptions
#         if tag_chunks and not is_similar_text(tag_chunks[-1], title_text):
#             #print("Title (unique) found: ", title_text, end="\n\t> ")
#             tag_chunks.append(title_text)
#         elif not tag_chunks:  # if it's the first one found, add it regardless
#             #print("Title (first) found: ", title_text, end="\n\t> ")
#             tag_chunks.append(title_text.strip())
#     for attrs in meta_names:
#         tag = soup.find("meta", attrs=attrs)
#         if tag and (tag_contents := tag.get("content")):
#             if tag_chunks and not is_similar_text(tag_chunks[-1], tag_contents.strip()):
#                 #print(attrs, " (unique) : ", tag_contents, end="\n\t> ")
#                 # avoid duplicates if the same description is found
#                 tag_chunks.append(tag_contents.strip())
#             elif not tag_chunks:  # if it's the first one found, add it regardless
#                 #print(attrs, " (first) : ", tag_contents, end="\n\t> ")
#                 tag_chunks.append(tag_contents.strip())
#     # try to append the first and last paragraph text if they exist
#     paragraphs = soup.find_all("p")
#     if paragraphs:
#         # append the first paragraph
#         p_start = paragraphs[0].get_text(strip=True)
#         #print("First paragraph found: ", p_start, end="\n\t> ")
#         tag_chunks.append(p_start)
#         # append the last paragraph if there are more than one paragraph tags
#         if len(paragraphs) > 1:
#             p_end = paragraphs[-1].get_text(strip=True)
#             #print("Last paragraph found: ", p_end, end="\n\t> ")
#             tag_chunks.append(p_end)
#     # print("\nFetched supplemental text from URL: ", url, end="\n\t>")
#     # TODO: maybe consider trying to grab the last paragraph as well since it would summarize the page?
#     return "\n".join(tag_chunks).strip()



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
    # print("\nFetched supplemental text from URL: ", url, end="\n\t>")
    #return fetch_from_tags(soup)
    raw_text = fetch_full_text(soup)
    return preprocess_html_text(raw_text)


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


def safe_fetch(url: str, attempt=1, max_retries = 3, retry_delay = 2) -> str:
    try:
        return default_html_fetcher(url)
    except Exception as e:
        if isinstance(e, (requests.exceptions.RequestException, urllib3.exceptions.ReadTimeoutError)):
            if attempt <= max_retries:
                # using exponential backoff for retry delay
                time.sleep(retry_delay * attempt)
                return safe_fetch(url, attempt + 1)
            print(f"[Timeout] Failed to fetch after {attempt} attempts: {url}")
        write_failed_fetch_log(url, str(e))
    return ""


def default_html_fetcher_batch(urls: List[str]) -> Dict[str, str]:
    if isinstance(urls, str):
        #print("WARNING: default_html_fetcher_batch received a single URL string instead of a list; using single URL.")
        #! FIXME: not how I want to keep this function but I need to rewrite the way supplementary text is added in general
        #return {urls: default_html_fetcher(urls)}
        #return safe_fetch(urls)
        return {urls: preprocess_html_text(safe_fetch(urls))}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        #results = list(executor.map(default_html_fetcher, urls))
        results = list(executor.map(safe_fetch, urls))
    # raw_map = dict(zip(urls, results))
    # return {url: preprocess_html_text(text) for url, text in raw_map.items()}
    cleaned = [preprocess_html_text(r) for r in results]
    return dict(zip(urls, cleaned))
    #return dict(zip(urls, results))
