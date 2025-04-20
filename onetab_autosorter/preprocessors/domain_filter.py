
import os
import json
import re
import termcolor
import uuid
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from collections import defaultdict, Counter
#from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
#import numpy as np
from typing import List, Dict, Set, Callable, Optional, Union, Any
# local imports
#from onetab_autosorter.utils.clean_utils import assign_line_bin, extract_phrases_by_line


# might integrate back into the class later
def filter_disjoint_boilerplate(phrases: List[str]) -> Set[str]:
    """ Filters boilerplate phrases to retain only disjoint sets of tokens, prioritizing longer tokens first.
        Args:
            phrases (List[str]): List of boilerplate phrases sorted from largest to smallest.
        Returns:
            List[str]: Filtered list of disjoint boilerplate phrases.
    """
    phrases = set(sorted(phrases, key=len, reverse=True))
    selected_phrases = set()
    tokens_seen = set()
    for phrase in phrases:
        phrase_tokens = set(phrase.lower().split())
        if not tokens_seen.intersection(phrase_tokens):
            selected_phrases.add(phrase)
            tokens_seen.update(phrase_tokens)
    #print("number of n-grams, before and after: ", (num_phrases, len(selected_phrases)))
    return selected_phrases




# def _fix_code_spacing(text: str) -> str:
#     """Fix common code spacing issues after pattern application."""
#     # Fix Python imports and common syntax running together
#     text = re.sub(r'(?<=[a-z])(import|from|as)(?=[a-z])', r' \1 ', text, flags=re.IGNORECASE)
#     # Fix other common Python syntax issues
#     python_keywords = ['def', 'class', 'for', 'in', 'if', 'else', 'elif',
#                       'while', 'return', 'with', 'try', 'except', 'finally']
#     for keyword in python_keywords:
#         text = re.sub(f'(?<=[a-z]){keyword}(?=[a-z])', f' {keyword} ', text, flags=re.IGNORECASE)
#     # fix very long words that are likely run-together tokens that keep happening to code blocks like "importmatplotlibpyplotaspltimportnumpyasnp"
#     long_word_pattern = re.compile(r'\b[a-zA-Z]{20,}\b')
#     for match in long_word_pattern.finditer(text):
#         long_word = match.group(0)
#         # Try splitting by common pattern boundaries
#         fixed_word = long_word
#         # Insert spaces before capital letters
#         fixed_word = re.sub(r'([a-z])([A-Z])', r'\1 \2', fixed_word)
#         # Try splitting at common boundaries in code
#         for boundary in ['import', 'from', 'as', 'plt', 'np', 'pd', 'def', 'class']:
#             fixed_word = fixed_word.replace(boundary, f' {boundary} ')
#         # clean up any double spaces created
#         fixed_word = re.sub(r' {2,}', ' ', fixed_word).strip()
#         # replace original long word with the fixed version
#         if fixed_word != long_word:
#             text = text.replace(long_word, fixed_word)
#     # replace any double spaces created
#     text = re.sub(r' {2,}', ' ', text)
#     return text



@dataclass
class DomainFilterData:
    """ Holds domain-specific boilerplate detection data """
    texts: List[str] = field(default_factory=list)
    boilerplate: Set[str] = field(default_factory=set)
    locked: bool = False
    no_valid_text: bool = False
    #?NOTE: may still want to add sample_count back in

    def flush_memory(self):
        self.texts.clear()

    def __bool__(self):
        return bool(self.boilerplate)

#? beware of inconsistent domain names since sometime's the base url root is extracted like `base_domain = sorted(domain.lower().split('.'), key=len)[-1]``


# TODO: need to add an additional function to pass through after `filter_disjoint_boilerplate` to remove non-english words from the set to filter out

class DomainBoilerplateFilter:
    """ Domain-wide TF-IDF based boilerplate detection and filtering. """
    MIN_TEXT_LENGTH = 3
    def __init__(self, min_domain_samples=8, min_phrase_freq=0.75, ngram_range=(2, 10), max_features=1000):
        self.min_domain_samples = min_domain_samples
        self.min_phrase_freq = min_phrase_freq
        #self.ngram_range = ngram_range
        self.phrase_min_length = ngram_range[0]
        self.phrase_max_length = ngram_range[1]
        self.max_features = max_features
        self.stopwords = set(stopwords.words("english"))
        self.domain_data_map: Dict[str, DomainFilterData] = defaultdict(DomainFilterData)

    def run_preliminary_search(self, domain_map: Dict[str, List[str]], min_domain_size: int = 2, html_fetcher_fn: Optional[Callable] = None):
        domain_counts = {dom: len(entries) for dom, entries in domain_map.items()}
        # sort domains by the number of entries to prioritize larger domains first
        sorted_domains = sorted(domain_map.keys(), key=lambda d: len(domain_map[d]), reverse=True)
        sorted_domains = [dom for dom in sorted_domains if domain_counts[dom] >= min_domain_size]
        # since some domains might already be locked from a prior run, only fetch if not locked
        for domain in tqdm(sorted_domains, desc="Domain Boilerplate Detection Step"):
            domain_data = self.get_domain_data(domain)
            if domain_data.locked: #or domain_counts[domain] < min_domain_size: <- redundant since sorted_domains was filtered for that
                continue  # already locked from previous runs
            # fill only as many domain entries as needed (FILTER_THRESHOLD)
            sampling_size = min(domain_counts[domain], self.min_domain_samples)
            # fetch a only as many pages as needed
            domain_entries: List[Dict[str, Any]] = domain_map[domain][:sampling_size]
            try:
                # aggregate text and filter out empty strings simultaneously
                text_map = {e["url"]: text for e in domain_entries if (text := e.get("scraped", "")) != ""}
                if not text_map:
                    print(termcolor.colored(f" [DOMAIN FILTER] No content found for domain: {domain}", "yellow"))
                    continue
            except KeyError: # keeping this here just for backward compatibility with the old format for now
                print(termcolor.colored(f" [DOMAIN FILTER] WARNING: domain {domain} has no scraped data!", color="yellow"))
                urls = [e["url"] for e in domain_entries]
                text_map = html_fetcher_fn(urls) if html_fetcher_fn else {url: "" for url in urls}
            for url, raw_text in text_map.items():
                #? NOTE: an internal trigger to add_entry_text finalizes phrases after enough are encountered
                if raw_text:
                    self.add_entry_text(domain, raw_text)
        # drop any domains that only consisted of stopwords or garbage text
        for domain in list(self.domain_data_map.keys()):
            if self.domain_data_map[domain].no_valid_text:
                _ = self.domain_data_map.pop(domain, None)
        # forcefully finalize any domain that didn't meet min_domain_samples but was greater than min_domain_size (so text is in the buffer)
        self.force_finalize_all()


    def filter_boilerplate(self, domain: str, text: str) -> str:
        """ Remove boilerplate patterns from text while preserving word boundaries """
        domain_data = self.get_domain_data(domain)
        if not domain_data.boilerplate: # or not domain_data.locked:
            return text
        if not text or text.isspace():
            print(termcolor.colored(f" [DOMAIN FILTER] Warning: text passed to domain filter is empty!", color="yellow"))
            return text
        # IMPROVED: Add word boundary detection and space preservation
        try:
            patterns = sorted(domain_data.boilerplate, key=len, reverse=True)
            # use a two-pass approach for better space preservation
            for pattern in patterns:
                # First add markers where we'll remove text
                marker = f" {uuid.uuid4().hex} "  # Unique marker with spaces on both sides
                # Use word boundary detection when possible
                if len(pattern.split()) > 1:  # Multi-word pattern
                    safe_pattern = re.escape(pattern)
                    text = re.sub(f"\\b{safe_pattern}\\b", marker, text, flags=re.IGNORECASE)
                else:  # Single word - use standard replacement
                    text = text.replace(pattern, marker)
            # Remove markers but keep their spaces
            filtered_text = re.sub(r' [0-9a-f]{32} ', ' ', text)
            # clean up excess whitespace but preserve paragraph structure
            filtered_text = re.sub(r' {2,}', ' ', filtered_text)        # replace multiple spaces with a single space
            filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)   # keep paragraph breaks
            return filtered_text.strip()
        except Exception as e:
            print(f"Error filtering text for {domain}: {e}")
            return text

    def add_entry_text(self, domain: str, text: str):
        domain_data = self.get_domain_data(domain)
        if domain_data.locked:
            return
        domain_data.texts.append(text)
        # internally triggers finalization of the set of phrases for a domain when min_domain_samples is reached
        if len(domain_data.texts) >= self.min_domain_samples:
            self._finalize_boilerplate(domain)

    def _finalize_boilerplate(self, domain: str):
        """ Process collected texts to identify boilerplate phrases by domain via the following strategies:
            1. Sentence-level common phrases
            2. Line-prefix and line-suffix patterns
            3. Common n-grams across documents
        """
        domain_data = self.get_domain_data(domain)
        #text_batch = [text for text in domain_data.texts if text != "" and not is_only_stopwords(text, self.stopwords)]
        text_batch = [text for text in domain_data.texts if text and len(text) > self.MIN_TEXT_LENGTH]
        if not text_batch:
            domain_data.no_valid_text = True
            domain_data.locked = True
            domain_data.flush_memory()
            print(termcolor.colored(f" [DOMAIN FILTER] WARNING: domain {domain} has no valid text data for boilerplate detection!", color="yellow"))
            return
        # 1. Extract common sentences and phrases
        sentence_phrases = self._extract_common_phrases(text_batch)
        # 2. Extract common line prefixes/suffixes
        line_patterns = self._extract_line_patterns(text_batch)
        # 3. Extract common word n-grams
        common_ngrams = self._extract_common_ngrams(text_batch)
        # Combine all detected patterns
        all_patterns = set(sentence_phrases + line_patterns + common_ngrams)
        # Filter out stopword-only or very short patterns
        ###filtered_patterns = self._filter_patterns(all_patterns)
        # Update the domain data
        domain_data.boilerplate.update(all_patterns) #filtered_patterns)
        domain_data.locked = True
        domain_data.flush_memory()



    def get_domain_data(self, domain: str) -> DomainFilterData:
        #? NOTE: relies on defaultdict values if domain doesn't yet exist
        return self.domain_data_map[domain]

    def get_present_domains(self):
        return list(self.domain_data_map.keys())

    def force_finalize_all(self):
        for domain, data in self.domain_data_map.items():
            # if initial domain filtering has concluded but texts for more than the min number of samples remains, run the boilerplate detection anyway
            if not data.locked and data.texts:
                self._finalize_boilerplate(domain)

    def _is_mostly_stopwords(self, text: str) -> bool:
        """ Check if text consists only of stopwords. """
        # clean and tokenize
        text = re.sub(r"[^\w\s]", "", text.lower()) # remove punctuation and special characters
        words = text.split()
        # check if most words are stopwords
        non_stopwords = [w for w in words if w not in self.stopwords]
        # Valid if at least 10% of words are not stopwords
        return len(non_stopwords) < (0.1 * len(words))

    def _is_common_phrase(self, phrase: str, count: int, min_count: int, test_stopwords = True, test_length = False):
        flag = count >= min_count
        if test_stopwords:
            flag &= not self._is_mostly_stopwords(phrase)
        if test_length:
            flag &= self.phrase_min_length <= len(phrase) <= self.phrase_max_length
        return flag

    def _filter_by_counter(self, texts: List[str], counter: Counter, test_stopwords = True, test_length = False):
        min_count = max(2, int(len(texts) * self.min_phrase_freq))
        return [p for p, n in counter.items() if self._is_common_phrase(p, n, min_count, test_stopwords=test_stopwords, test_length=test_length)]

    def _extract_common_phrases(self, texts: List[str]) -> List[str]:
        """Extract phrases that occur across multiple documents."""
        # Split into sentences
        all_sentences = []
        for text in texts:
            # Simple sentence splitting at punctuation
            sentences = re.split(r'[.!?]+', text)
            all_sentences.extend([s.strip() for s in sentences if len(s.strip()) >= self.phrase_min_length])
        # Count frequencies
        sentence_counter = Counter(all_sentences)
        # Keep sentences that appear in more than min_phrase_freq proportion of documents
        return self._filter_by_counter(texts, sentence_counter, test_length=True)


    def _extract_line_patterns(self, texts: List[str]) -> List[str]:
        """Extract common line prefixes and suffixes."""
        all_lines = []
        for text in texts:
            lines = text.split('\n')
            all_lines.extend([line.strip() for line in lines if line.strip()])
        # Process beginnings and endings of lines
        beginnings = []
        endings = []
        for line in all_lines:
            # Get first ~30 characters for beginnings
            if len(line) > 15:
                prefix = line[:30].strip()
                if len(prefix) >= self.phrase_min_length:
                    beginnings.append(prefix)
            # Get last ~30 characters for endings
            if len(line) > 15:
                suffix = line[-30:].strip()
                if len(suffix) >= self.phrase_min_length:
                    endings.append(suffix)
        # Count frequencies
        beginning_counter = Counter(beginnings)
        ending_counter = Counter(endings)
        # Select common patterns
        common_beginnings = self._filter_by_counter(texts, beginning_counter)
        common_endings = self._filter_by_counter(texts, ending_counter)
        return common_beginnings + common_endings

    def _extract_common_ngrams(self, texts: List[str]) -> List[str]:
        """ Extract common word n-grams appearing across documents. """
        # Extract and count all 2-6 word n-grams
        ngram_counter = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            # Generate n-grams of size 2-6
            for n in range(self.phrase_min_length, self.phrase_max_length):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    if len(ngram) >= self.phrase_min_length:
                        ngram_counter[ngram] += 1
        # Keep n-grams that appear across multiple documents
        return self._filter_by_counter(texts, ngram_counter, test_length=True)


    # def _filter_patterns(self, patterns: Set[str]) -> Set[str]:
    #     """ Filter out poor quality boilerplate patterns. """
    #     # First sort by length (descending) to prioritize longer patterns
    #     sorted_patterns = sorted(patterns, key=len, reverse=True)
    #     # Filter out duplicate information
    #     result = []
    #     seen_tokens = set()
    #     for pattern in sorted_patterns:
    #         # Clean up pattern
    #         pattern = pattern.strip()
    #         tokens = set(pattern.lower().split())
    #         # Skip if pattern contains mostly seen tokens
    #         if len(tokens.intersection(seen_tokens)) > 0.75 * len(tokens):
    #             continue
    #         # Skip very common web phrases
    #         if pattern.lower() in {'cookie policy', 'terms of service', 'privacy policy',
    #                               'accept all cookies', 'all rights reserved',
    #                               'sign in', 'sign up', 'log in', 'copyright'}:
    #             continue
    #         # Keep this pattern
    #         if len(pattern) >= self.phrase_min_length:
    #             result.append(pattern)
    #             seen_tokens.update(tokens)
    #     return set(result)


    def save_boilerplate_map(self, json_path: str, override: bool = False):
        if override or not os.path.exists(json_path):
            out_data = {domain: sorted(list(data.boilerplate)) for domain, data in self.domain_data_map.items()}
            if not out_data:
                print(termcolor(" [DOMAIN FILTER] WARNING: domain boilerplate data empty when saving to JSON!", color="yellow"))
                return
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as fptr:
                json.dump(out_data, fptr, indent=2)
            print(termcolor.colored(f" [DOMAIN FILTER] Saved extracted domain boilerplate to {json_path}.", color="green"))

    @classmethod
    def load_boilerplate_map(cls, json_path: str, **kwargs):
        obj = cls(**kwargs)
        if not os.path.isfile(json_path):
            return obj
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = dict(json.load(f))
        for domain, phrases in loaded.items():
            domain_data: DomainFilterData = obj.domain_data_map[domain]
            domain_data.boilerplate = set(phrases)
            domain_data.locked = True
        return obj






