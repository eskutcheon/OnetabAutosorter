"""
    Utility functions for fetching and parsing HTML content from URLs with Python.
    Java microservice scraper functions located in scraper/client.py
"""

import html
import requests
import re
import urllib3.exceptions
import warnings
import unicodedata
import time
from tqdm.auto import tqdm
from itertools import islice
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, Tag
from typing import List, Dict, Optional
# import trafilatura
# from newspaper import Article # newspaper4k
from io import BytesIO
import PyPDF2



BROWSER_CHALLENGE_PHRASES = [
    "just a moment",
    "checking your browser",
    "verify you are a human",
    "please wait while we verify",
    "access to this page has been denied",
    "challenge verification",
    "ddos protection",
    "security check"
]


def process_code_text(text: str) -> str:
    """ Process code text to preserve boundaries between statements, imports, etc """
    # Handle Python imports - add spaces between import statements
    text = re.sub(r'(import\s+\w+)', r'\1 ', text)
    # Add space after common code punctuation that might not have spaces
    text = re.sub(r'([;{}()])', r'\1 ', text)
    # Fix common patterns in code that run together
    text = re.sub(r'(\w)(if|for|while|def|class|return|import|from)', r'\1 \2', text, flags=re.IGNORECASE)
    # Clean up any double spaces created
    text = re.sub(r'\s{2,}', ' ', text)
    return text


#& MOSTLY WRITTEN BY LLM - needs review, but it's been tested and it's working for now though
def extract_main_content(soup: BeautifulSoup, max_tokens: int = 1000) -> str:
    """ Extract main content from HTML using structural heuristics
        1. Removes boilerplate elements
        2. Prioritizes content based on HTML structure
        3. Weights text by tag importance
    """
    # split up further later
    # Remove script, style, and non-content elements
    for element in soup(["script", "style", "noscript", "nav", "footer", "header",
                         "aside", "form", "iframe", "button"]):
        element.decompose()
    # Content container candidates - check for common article containers first
    content_containers = soup.select("article, .article, .post, .content, main, #main-content, .main-content")
    # If we found potential content containers, use the largest one
    main_container = None
    if content_containers:
        main_container = max(content_containers, key=lambda x: len(x.get_text(strip=True)))
    # If no content containers found, use the whole document
    container = main_container or soup
    # Extract text with weights by tag importance
    weighted_text = []
    # Title gets highest weight
    if soup.title:
        title_text = soup.title.get_text(strip=True)
        if title_text:
            weighted_text.append((title_text, 3.0))
    # Meta description is very important
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and 'content' in meta_desc.attrs:
        weighted_text.append((meta_desc['content'], 2.5))
    # Headers get high weights (h1 highest, h6 lowest)
    for i in range(1, 7):
        for header in container.find_all(f'h{i}'):
            text = header.get_text(strip=True)
            if text:
                weight = 2.5 - ((i-1) * 0.3)  # h1=2.5, h2=2.2, ..., h6=1.0
                weighted_text.append((text, weight))
    # Main content - paragraphs and list items
    for tag in container.find_all(['p', 'li']):
        text = tag.get_text(strip=True)
        if text and len(text.split()) >= 3:  # Skip very short fragments
            weighted_text.append((text, 1.0))
    # Other potentially useful content with lower weights
    for tag in container.find_all(['div', 'section', 'span']):
        # Skip if it contains other elements we've already processed
        if tag.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            continue
        text = tag.get_text(strip=True)
        # Only include divs with substantial text that look like content
        if text and len(text.split()) >= 5:
            weighted_text.append((text, 0.7))
    # add special handling for code blocks
    for code_block in container.find_all(['code', 'pre']):
        text = code_block.get_text(strip=True)
        if text and len(text) > 15:  # some meaningful code block
            # process code blocks to preserve imports and statements
            processed_code = process_code_text(text)
            weighted_text.append((processed_code, 1.2))  # Give code slightly higher priority
    # Sort by weight then join
    weighted_text.sort(key=lambda x: x[1], reverse=True)
    # Join pieces with their weights to create final text
    all_text = []
    token_count = 0
    for text, weight in weighted_text:
        # add period to end of text if not already present to ensure sentence separation
        if text and not text.endswith(('.', '!', '?', ':', ';')):
            text = text + '.'
        tokens = text.split()
        # If this piece would exceed our max tokens, truncate it
        if token_count + len(tokens) > max_tokens:
            tokens = tokens[:max_tokens - token_count]
            all_text.append(' '.join(tokens))
            break
        all_text.append(text)
        token_count += len(tokens)
        if token_count >= max_tokens:
            break
    #print(" [SCRAPER] number of tokens removed from original text: ", text_length_init - token_count)
    return ' '.join(all_text)



#& may keep this here or move it to the general TextCleaningFilter class later
def fetch_full_text(soup: BeautifulSoup, max_tokens: int = 1000) -> str:
    """ Extract raw visible text from the full HTML page, ignoring scripts/styles """
    # remove script and style elements first:
    for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        element.decompose()
    # for efficiency, use islice over the iterator of significant strings to limit the number of tokens returned
    text = list(islice(soup.stripped_strings, 0, max_tokens))
    #print("SCRAPED TEXT LENGTH: ", len(text))
    return " ".join(text) # split into tokens and limit to max_tokens

################################ requests-related functions ################################

def has_browser_safeguard(resp: requests.Response, url: str) -> bool:
    # quick check before parsing to avoid unnecessary processing
    content_sample = resp.text[:1000].lower()
    if any(phrase in content_sample for phrase in BROWSER_CHALLENGE_PHRASES):
        write_failed_fetch_log(url, "SKIPPING: Detected browser challenge page")
        return True  # Return empty for challenge pages
    return False


def attempt_document_extraction(resp: requests.Response, url: str, content_type: str = None) -> Optional[str]:
    # PDF handling - extract text if possible, otherwise log failure and return empty string
    if not content_type:
        content_type = resp.headers.get("Content-Type", "").lower()
    if 'application/pdf' in content_type:
        #print("PDF detected, attempting to extract text...")
        try:
            pdf = PyPDF2.PdfReader(BytesIO(resp.content))
            text = ' '.join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text
        except Exception as e:
            write_failed_fetch_log(url, f"PDF extraction failed: {str(e)}")
            return "" # returning an empty string when it is actually a PDF but the extraction failed
            #raise e(f"[PDF EXTRACTION ERROR: {str(e)}]")
    # plaintext handling
    if 'text/plain' in content_type:
        #print("Plain text detected. Returning raw text...")
        return resp.text
    # if NoneType returned, then the content type isn't PDF or plaintext
    return None






# intended to replace `default_html_fetcher` on a trial basis
def enhanced_html_fetcher(url: str) -> str:
    """ Enhanced HTML fetcher with content type detection and special handling """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        #'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        #'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1'  # Do Not Track
    }
    resp = requests.get(url, headers=headers, timeout=5)
    #resp.raise_for_status() # leads to a lot more errors, which is good to catch, but I'm handling it with logging instead
        # Instead of raising exceptions, check status explicitly
    if resp.status_code >= 400:
        write_failed_fetch_log(url, f"HTTP Error: {resp.status_code}")
        return ""
    if has_browser_safeguard(resp, url):
        return ""
    # attempt to extract document content if applicable
    content_type = resp.headers.get("Content-Type", "").lower()
    doc_text = attempt_document_extraction(resp, url, content_type)
    if doc_text is not None:
        return doc_text
    # set correct encoding for HTML and parser types
    encoding = resp.apparent_encoding or 'utf-8'
    parser_type = "xml" if "xml" in content_type else "html.parser"
    # parse the HTML content while ignoring warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        try:
            decoded = resp.content.decode(encoding, errors="replace")
        except Exception:
            decoded = resp.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(decoded, parser_type)
        # UPDATED: earlier unicode normalization - remove need for later steps
        text_content = extract_main_content(soup)
        text_content = unicodedata.normalize('NFKD', text_content).encode('ascii', 'ignore').decode('ascii')
    # extract main content with the enhanced extraction function
    return text_content #extract_main_content(soup)








################################ older webscraping functions ################################

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
        text = enhanced_html_fetcher(url) #default_html_fetcher(url)
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