
from typing import Optional, Dict
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional
from termcolor import colored, cprint
# local imports
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter, EnhancedTextCleaningFilter
from onetab_autosorter.scraper.scraper_utils import fetch_full_text, extract_main_content




class TextPreprocessingHandler:
    """ Aggregates multiple text preprocessing steps into a single pipeline manager:
        1) Parse/clean HTML with BeautifulSoup
        2) Possibly run advanced text filter for non-English chars, LaTeX, etc.
        3) Use DomainBoilerplateFilter for domain-based repeated phrase removal
        4) Return final cleaned text
    """

    def __init__(
        self,
        domain_filter: DomainBoilerplateFilter = None,
        cleaning_filter: TextCleaningFilter = None,
        max_tokens: int = 400,
    ):
        """
            :param domain_filter: an instance of DomainBoilerplateFilter (optional).
            :param cleaning_filter: an instance of TextCleaningFilter (optional).
            :param max_tokens: how many tokens to keep after extraction from soup.
        """
        self.domain_filter = domain_filter
        self.cleaning_filter = cleaning_filter
        self.max_tokens = max_tokens
        ### assuming that the domain filter has been trained already (need to iron out that logic):
        self.domain_names = self.domain_filter.get_present_domains() if domain_filter else []

    def process_html(
        self,
        raw_html: str,
        domain: Optional[str] = None,
        use_domain_filter: bool = True
    ) -> str:
        """ Main method: Takes raw HTML, does soup-based extraction, applies a pipeline of optional filtering, and returns final text. """
        # parse text from HTML
        soup = BeautifulSoup(raw_html, "html.parser")
        text = self._extract_from_soup(soup)
        # basic, regex, and domain filtering
        return self.process_text(text, domain, use_domain_filter)

    def is_text_empty(self, text: str, step_name: str) -> bool:
        if text == "":
            print(f" [HANDLER ({step_name})] WARNING: No text to process")
            return True
        return False

    def is_text_overcleaned(self, text: str, initial_length: int, step_name: str = "") -> bool:
        """ Check if the text has been over-cleaned (e.g., lost too much content) """
        OVERFILTERED_THRESHOLD = 0.05
        if len(text) / initial_length < OVERFILTERED_THRESHOLD:
            percent = int((1 - OVERFILTERED_THRESHOLD) * 100)
            cprint(f" [HANDLER] WARNING: Lost over {percent}% of text ({step_name})", "yellow", attrs=["bold", "underline"])
            return True
        return False

    def process_text(
        self,
        text: str,
        domain: Optional[str] = None,
        use_domain_filter: bool = True
    ) -> str:
        """ Overload for the case where we already have plain text (no HTML) """
        # TODO: going to move some of the warnings from the domain and cleaning filters to their respective classes
        if self.is_text_empty(text, "before cleaning"):
            return text
        initial_length = len(text)
        # advanced filtering
        if self.cleaning_filter:
            text = self.cleaning_filter.filter(text, self.max_tokens)
        if self.is_text_empty(text, "after cleaning"):
            return text
        if self.is_text_overcleaned(text, initial_length, step_name="after cleaning"):
            return text
        # domain-based
        if use_domain_filter and self.domain_filter and domain:
            # call `add_entry_text` if building up the domain filter or call `.filter_boilerplate` if domain is locked
            text = self._apply_domain_filter(domain, text)
        if text == "":
            print(" [HANDLER (after domain filter)] WARNING: No text after domain filter")
        return text


    def _apply_domain_filter(self, domain: str, text: str) -> str:
        # if domain not locked, add or if locked, filter
        if domain not in self.domain_names:
            return text
        data = self.domain_filter.get_domain_data(domain)
        if data and data.locked:
            # domain is locked => filter
            return self.domain_filter.filter_boilerplate(domain, text)
        else:
            # domain not locked => accumulate text so the domain filter can finalize eventually
            self.domain_filter.add_entry_text(domain, text)
        return text  # we don't do removal yet if not locked


    def _extract_from_soup(self, soup: BeautifulSoup) -> str:
        """ Clean raw HTML data and get text while optionally truncating to self.max_tokens """
        fetch_full_text(soup, self.max_tokens)



    # #& currently unused - determine whether to keep later
    # def process_batch_html(self, html_map: Dict[str, str], domain_map: Dict[str, str]) -> Dict[str, str]:
    #     """ Process multiple HTML documents in batch.
    #         Args:
    #             html_map: Dict mapping URLs to HTML content
    #             domain_map: Dict mapping URLs to domain names
    #         Returns:
    #             Dict mapping URLs to processed text
    #     """
    #     results = {}
    #     for url, html in tqdm(html_map.items(), desc="Processing HTML batch"):
    #         domain = domain_map.get(url, "")
    #         processed = self.process_html(html, domain)
    #         results[url] = processed
    #     return results

    # #& currently unused - determine whether to keep later
    # def process_batch_text(self, text_map: Dict[str, str], domain_map: Dict[str, str]) -> Dict[str, str]:
    #     """ Process multiple text documents in batch.
    #         Args:
    #             text_map: Dict mapping URLs to text content
    #             domain_map: Dict mapping URLs to domain names
    #         Returns:
    #             Dict mapping URLs to processed text
    #     """
    #     results = {}
    #     for url, text in tqdm(text_map.items(), desc="Processing text batch"):
    #         domain = domain_map.get(url, "")
    #         processed = self.process_text(text, domain)
    #         results[url] = processed
    #     return results


    # def process_entries(self, entries: List[Dict[str, Any]], content_map: Dict[str, str]) -> List[Dict[str, Any]]:
    def process_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Process a list of entries with their content.
            Args:
                entries: List of entry dictionaries
                content_map: Dict mapping URLs to content (HTML or text)
            Returns:
                The updated list of entries with processed content
        """
        for idx, entry in enumerate(tqdm(entries, desc="Processing entries")):
            #url = entry["url"]
            domain = entry.get("domain", "")
            #content = content_map.get(url, "")
            #content = entry.get("scraped", None)
            content = entry.pop("scraped", None)
            if content:
                # Determine if content is HTML or plain text
                is_html = bool(content.strip().startswith(("<html", "<!DOCTYPE", "<doc")))
                if is_html:
                    print(" [HANDLER] Something went wrong - didn't expect this result in the current implementation")
                processed = self.process_html(content, domain) if is_html else self.process_text(content, domain)
                entries[idx]["clean_text"] = processed
                #print("[HANDLER] Processed text length for entry:", len(entries[idx]["clean_text"]))
            else:
                print(colored(f" [HANDLER] No content found for entry '{entries[idx]['url']}'", "yellow"))
                entries[idx]["clean_text"] = ""
        return entries




class EnhancedTextPreprocessingHandler(TextPreprocessingHandler):
    """ Enhanced text preprocessing handler with improved content extraction """
    def __init__(
        self,
        domain_filter: DomainBoilerplateFilter = None,
        cleaning_filter: TextCleaningFilter = None,
        max_tokens: int = 400,
    ):
        self.domain_filter = domain_filter
        self.cleaning_filter = cleaning_filter or EnhancedTextCleaningFilter()  # Use enhanced filter by default
        self.max_tokens = max_tokens
        self.domain_names = self.domain_filter.get_present_domains() if domain_filter else []
        self.content_quality_stats = {}  # Track quality metrics per domain

    def process_html(
        self,
        raw_html: str,
        domain: Optional[str] = None,
        url: Optional[str] = None,
        use_domain_filter: bool = True
    ) -> str:
        """ Process HTML with enhanced content extraction. """
        # Parse and identify main content block
        soup = BeautifulSoup(raw_html, "html.parser")
        # Extract relevant text with priority on important sections
        text = extract_main_content(soup, self.max_tokens)
        # Process using text pipeline
        return self.process_text(text, domain, use_domain_filter)

    def process_text(
        self,
        text: str,
        domain: Optional[str] = None,
        use_domain_filter: bool = True
    ) -> str:
        """ Process text with enhanced cleaning and filtering. """
        if self.is_text_empty(text, "before cleaning"):
            return text
        initial_length = len(text)
        initial_token_count = len(text.split())
        # Apply text cleaning filter
        if self.cleaning_filter:
            text = self.cleaning_filter.filter(text, self.max_tokens)
        if self.is_text_empty(text, "after cleaning"):
            self._update_quality_stats(domain, 'overcleaned', True)
            return text
        # Check if overcleaned
        if self.is_text_overcleaned(text, initial_length, "after cleaning"):
            self._update_quality_stats(domain, 'overcleaned', True)
        # Apply domain-based filtering
        if use_domain_filter and self.domain_filter and domain:
            pre_domain_filter_length = len(text)
            text = self._apply_domain_filter(domain, text)
            # Track domain filter impact
            if text and pre_domain_filter_length > 0:
                reduction_pct = 1 - (len(text) / pre_domain_filter_length)
                self._update_quality_stats(domain, 'domain_filter_reduction', reduction_pct)
        if text == "":
            self._update_quality_stats(domain, 'final_empty', True)
            print(" [HANDLER] WARNING: No text after all processing")
        return text

    def _update_quality_stats(self, domain: str, metric: str, value):
        """Track quality statistics for later analysis."""
        if not domain:
            return
        if domain not in self.content_quality_stats:
            self.content_quality_stats[domain] = {}
        if metric not in self.content_quality_stats[domain]:
            self.content_quality_stats[domain][metric] = []
        self.content_quality_stats[domain][metric].append(value)

    def get_quality_report(self):
        """Generate a report of content quality metrics."""
        report = {}
        for domain, metrics in self.content_quality_stats.items():
            report[domain] = {}
            for metric, values in metrics.items():
                if metric == 'overcleaned' or metric == 'final_empty':
                    report[domain][metric] = sum(values) / len(values) if values else 0
                else:
                    report[domain][metric] = sum(values) / len(values) if values else 0
        return report

    def process_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Enhanced entry processing with quality tracking. """
        for idx, entry in enumerate(tqdm(entries, desc="Processing entries")):
            domain = entry.get("domain", "")
            content = entry.pop("scraped", None)
            if content:
                is_html = bool(content.strip().startswith(("<html", "<!DOCTYPE", "<doc")))
                processed = self.process_html(content, domain, entry.get("url", "")) if is_html else self.process_text(content, domain)
                entries[idx]["clean_text"] = processed
                # Track content quality
                self._update_quality_stats(domain, 'final_length', len(processed) if processed else 0)
                self._update_quality_stats(domain, 'final_token_count', len(processed.split()) if processed else 0)
            else:
                print(colored(f" [HANDLER] No content found for entry '{entries[idx]['url']}'", "yellow"))
                entries[idx]["clean_text"] = ""
        return entries