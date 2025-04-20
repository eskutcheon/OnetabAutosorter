
import asyncio
import aiohttp
from urllib.parse import urlparse
from collections import defaultdict, deque
from typing import List, Dict, Optional
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import time
import threading
# local imports
from onetab_autosorter.scraper.scraper_utils import safe_fetch, fetch_full_text, extract_main_content, write_failed_fetch_log

#! The multi-threaded approach is faster now but doesn't include rate limiting, so this should still be the preferred option with the Python approach

class WebScraper:
    """ utility class for fetching web pages with support for rate limiting, domain interleaving, and both synchronous and asynchronous operations
        Args:
            rate_limit_delay (float): minimum delay (in sec) between requests to the same domain
            max_workers (int): maximum number of concurrent workers for fetching URLs
            interleave_domains (bool): whether to interleave URLs by domain to evenly distribute requests
    """
    def __init__(self, rate_limit_delay: float = 2.0, max_workers: int = 10, interleave_domains: bool = True):
        self.rate_limit_delay = rate_limit_delay
        self.max_workers = max_workers
        self.interleave_domains = interleave_domains
        self.last_request_time = defaultdict(lambda: 0.0) # track the last request time for each domain
        self.domain_lock = threading.Lock()               # lock to ensure thread-safe updates to last_request_time

    def _enforce_rate_limit(self, domain: str):
        """ enforce rate limit for the given domain with a lock mechanism """
        with self.domain_lock:
            now = time.time()
            elapsed = now - self.last_request_time[domain]
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
            self.last_request_time[domain] = time.time()

    def _rate_limited_fetch(self, url: str) -> str:
        """ fetch while respecting domain rate limit """
        domain = urlparse(url).netloc
        self._enforce_rate_limit(domain)
        return safe_fetch(url)

    def interleave_by_domain(self, urls: List[str]) -> List[str]:
        """ interleaves a list of URLs by their domains to distribute more requests evenly using a round-robin approach """
        domain_map = defaultdict(deque)
        for url in urls:
            domain = urlparse(url).netloc
            domain_map[domain].append(url)
        result = []
        while any(domain_map.values()):
            for q in domain_map.values():
                if q:
                    result.append(q.popleft())
        return result

    #& UNUSED - saves HTML
    def fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        """ Multi-threaded batch fetcher with optional domain interleaving and rate limiting. """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ordered_urls = self.interleave_by_domain(urls) if self.interleave_domains else urls
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._rate_limited_fetch, url): url for url in ordered_urls}
            for future in tqdm(as_completed(futures), total=len(urls), desc="Fetching", colour="cyan"):
                url = futures[future]
                #? NOTE: exceptions from fetching are currently handled by safe_fetch
                results[url] = future.result()
        return results

    async def async_safe_fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        """ Asynchronously fetches a URL while respecting the rate limit for its domain """
        domain = urlparse(url).netloc
        now = time.time()
        elapsed = now - self.last_request_time[domain]
        if elapsed < self.rate_limit_delay:
            #? NOTE: await call doesn't acquire a lock, which could lead to race conditions when updating self.last_request_time
            #! FIXME: consider using asyncio.Lock() for thread safety
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time[domain] = time.time()
        try:
            async with session.get(url, timeout=10) as resp:
                text = await resp.text()
                soup = BeautifulSoup(text, "html.parser")
                return extract_main_content(soup) #fetch_full_text(soup)
        except Exception as e:
            write_failed_fetch_log(url, e)
            return ""

    async def _async_fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        """ Coroutine to fetch all URLs asynchronously using aiohttp. """
        connector = aiohttp.TCPConnector(limit_per_host=self.max_workers)
        headers = {"User-Agent": "Mozilla/5.0"}
        results = {}
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            tasks = [self.async_safe_fetch(session, url) for url in urls]
            #! FIXME: might need to use return_exceptions=True to handle exceptions properly and raise them to be caught by the calling function
            texts = await asyncio.gather(*tasks)
            results = dict(zip(urls, texts))
        return results

    def run_async_fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        """ Fetches a batch of URLs asynchronously, handling the event loop appropriately """
        if not asyncio.get_event_loop().is_running():
            return asyncio.run(self._async_fetch_batch(urls))
        else:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._async_fetch_batch(urls))
