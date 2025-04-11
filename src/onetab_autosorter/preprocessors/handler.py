
from typing import Optional, Dict
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

# Suppose we import or define these somewhere:
from .domain_filter import DomainBoilerplateFilter
from .text_filters import TextCleaningFilter



# TODO: might still wrap stuff in a WebScraper class and pass it here for unified preprocessing directly from the HTML from BeautifulSoup
    # - would allow for easier management of all the actual scraping logic like adding rate-limiting, 
    # might end up removing the check for whether the user has an internet connection unless I find a much easier way

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
        max_tokens=200,
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
        self.domain_names = self.domain_filter.get_present_domains()

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

    def process_text(
        self,
        text: str,
        domain: Optional[str] = None,
        use_domain_filter: bool = True
    ) -> str:
        """ Overload for the case where we already have plain text (no HTML) """
        # advanced filtering
        # TODO: might want to move the maximum number of tokens to the cleaning filter itself
        if self.cleaning_filter:
            text = self.cleaning_filter.filter(text, self.max_tokens)
        # domain-based
        if use_domain_filter and self.domain_filter and domain:
            # call `add_entry_text` if building up the domain filter or call `.filter_boilerplate` if domain is locked
            text = self._apply_domain_filter(domain, text)
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


    #& might pass of this task to the HTML2TextFilter class later, assuming this class has an instance as a member variable
    def _extract_from_soup(self, soup: BeautifulSoup) -> str:
        """ Clean raw HTML data and get text while optionally truncating to self.max_tokens """
        remove_tags = ["script", "style", "noscript"]
        for tagname in remove_tags:
            for t in soup.find_all(tagname):
                t.decompose()
        # If you want to remove certain CSS selectors:
        # for sel in [".footnote", "header nav", "aside"]:
        #     for t in soup.select(sel):
        #         t.decompose()
        # now get text
        text = soup.get_text(separator="\n", strip=True)
        tokens = text.split()
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        return " ".join(tokens)

    # # !!! Seemingly not used - need to do a check on the domain filter's data dictionary and call it if needed
    # def finalize_domain_filter(self):
    #     """ forcibly finalize domain filter if needed. """
    #     if self.domain_filter:
    #         self.domain_filter.force_finalize_all()

    def process_batch_html(self, html_map: Dict[str, str], domain_map: Dict[str, str]) -> Dict[str, str]:
        """ Process multiple HTML documents in batch.
            Args:
                html_map: Dict mapping URLs to HTML content
                domain_map: Dict mapping URLs to domain names
            Returns:
                Dict mapping URLs to processed text
        """
        results = {}
        for url, html in tqdm(html_map.items(), desc="Processing HTML batch"):
            domain = domain_map.get(url, "")
            processed = self.process_html(html, domain)
            results[url] = processed
        return results

    def process_batch_text(self, text_map: Dict[str, str], domain_map: Dict[str, str]) -> Dict[str, str]:
        """ Process multiple text documents in batch.
            Args:
                text_map: Dict mapping URLs to text content
                domain_map: Dict mapping URLs to domain names
            Returns:
                Dict mapping URLs to processed text
        """
        results = {}
        for url, text in tqdm(text_map.items(), desc="Processing text batch"):
            domain = domain_map.get(url, "")
            processed = self.process_text(text, domain)
            results[url] = processed
        return results

    def process_entries(self, entries: List[Dict[str, Any]], content_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """ Process a list of entries with their content.
            Args:
                entries: List of entry dictionaries
                content_map: Dict mapping URLs to content (HTML or text)
            Returns:
                The updated list of entries with processed content
        """
        for entry in tqdm(entries, desc="Processing entries"):
            url = entry["url"]
            domain = entry.get("domain", "")
            content = content_map.get(url, "")
            if content:
                # Determine if content is HTML or plain text
                is_html = bool(content.strip().startswith(("<html", "<!DOCTYPE", "<doc")))
                if is_html:
                    processed = self.process_html(content, domain)
                else:
                    processed = self.process_text(content, domain)
                entry["processed_text"] = processed
            else:
                entry["processed_text"] = ""
        return entries