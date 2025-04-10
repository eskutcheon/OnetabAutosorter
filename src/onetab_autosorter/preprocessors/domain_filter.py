import os
import json
import re
import termcolor
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from collections import defaultdict #, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
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
    # #! FOR DEBUGGING - remove later:
    # num_phrases = len(phrases)
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



@dataclass
class DomainFilterData:
    """Holds domain-specific boilerplate detection data."""
    texts: List[str] = field(default_factory=list)
    boilerplate: Set[str] = field(default_factory=set)
    locked: bool = False
    #?NOTE: may still want to add sample_count back in

    def flush_memory(self):
        self.texts.clear()

    def __bool__(self):
        return bool(self.boilerplate)

#? beware of inconsistent domain names since sometime's the base url root is extracted like `base_domain = sorted(domain.lower().split('.'), key=len)[-1]``


# TODO: need to add an additional function to pass through after `filter_disjoint_boilerplate` to remove non-english words from the set to filter out

class DomainBoilerplateFilter:
    """ Domain-wide TF-IDF based boilerplate detection and filtering. """
    def __init__(self, min_domain_samples=5, min_df_ratio=0.8, ngram_range=(2, 10), max_features=500):
        self.min_domain_samples = min_domain_samples
        self.min_df_ratio = min_df_ratio
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.domain_data_map: Dict[str, DomainFilterData] = defaultdict(DomainFilterData)

    def run_preliminary_search(self, domain_map: Dict[str, List[str]], min_domain_size: int = 2, html_fetcher_fn: Optional[Callable] = None):
        domain_counts = {dom: len(entries) for dom, entries in domain_map.items()}
        sorted_domains = sorted(domain_map.keys(), key=lambda d: len(domain_map[d]), reverse=True)
        sorted_domains = [dom for dom in sorted_domains if domain_counts[dom] >= min_domain_size]
        # since some domains might already be locked from a prior run, only fetch if not locked
        for domain in tqdm(sorted_domains, desc="Domain Boilerplate Detection Step"):
            domain_data = self.get_domain_data(domain)
            if domain_data.locked or domain_counts[domain] < min_domain_size:
                continue  # already locked from previous runs
            # fill only as many domain entries as needed (FILTER_THRESHOLD)
            sampling_size = min(domain_counts[domain], self.min_domain_samples)
            # fetch a only as many pages as needed
            domain_entries = domain_map[domain][:sampling_size]
            urls = [e["url"] for e in domain_entries]
            # fetch HTML information from HTTP requests in a batch
            #? NOTE: if I want this method to be integrated into `DomainBoilerplateFilter`, I should pass this function and the urls
            text_map = html_fetcher_fn(urls) if html_fetcher_fn else {url: "" for url in urls}
            for url, raw_text in text_map.items():
                #? NOTE: an internal trigger to add_entry_text finalizes phrases after enough are encountered
                self.add_entry_text(domain, raw_text)
        # forcefully finalize any domain that didn't meet min_domain_samples but was greater than min_domain_size (so text is in the buffer)
        self.force_finalize_all()


    def get_domain_data(self, domain: str) -> DomainFilterData:
        #? NOTE: relies on defaultdict values if domain doesn't yet exist
        return self.domain_data_map[domain]

    def get_present_domains(self):
        return list(self.domain_data_map.keys())

    def add_entry_text(self, domain: str, text: str):
        domain_data = self.get_domain_data(domain)
        if domain_data.locked:
            return
        domain_data.texts.append(text)
        # internally triggers finalization of the set of phrases for a domain when min_domain_samples is reached
        if len(domain_data.texts) >= self.min_domain_samples:
            self._finalize_boilerplate(domain)

    def force_finalize_all(self):
        for domain, data in self.domain_data_map.items():
            # if initial domain filtering has concluded but texts for more than the min number of samples remains, run the boilerplate detection anyway
            if not data.locked and data.texts:
                self._finalize_boilerplate(domain)

    def _finalize_boilerplate(self, domain: str):
        #domain_data = self.domain_data_map[domain]
        # if domain_data.locked or len(domain_data.texts) < self.min_domain_samples:
        #     return
        vectorizer = TfidfVectorizer(
            #! fails on some domains that may be only stopwords - their docs even say to rethink using `stopwords="english"`
            #stop_words="english",
            strip_accents="unicode",
            ngram_range=self.ngram_range,
            #max_features=self.max_features,
            min_df = self.min_df_ratio,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
        )
        _ = vectorizer.fit_transform(self.domain_data_map[domain].texts)
        repeated_phrases = vectorizer.get_feature_names_out()
        repeated_phrases = filter_disjoint_boilerplate(repeated_phrases)
        self.domain_data_map[domain].boilerplate.update(repeated_phrases)
        self.domain_data_map[domain].locked = True
        #print(f"Domain {domain} locked with {len(repeated_phrases)} phrases.")
        self.domain_data_map[domain].flush_memory()

    def filter_boilerplate(self, domain: str, text: str) -> str:
        domain_data = self.get_domain_data(domain)
        if not domain_data.boilerplate: # or not domain_data.locked:
            return text
        # Efficiently filter text using precompiled regex
        pattern = re.compile('|'.join(re.escape(bp) for bp in sorted(domain_data.boilerplate, key=len, reverse=True)), re.IGNORECASE)
        filtered_text = pattern.sub('', text)
        if text != "" and filtered_text == text:
            print(termcolor.colored(f"WARNING '{domain}' filtered text is the same as original text!", color="yellow"))
        return filtered_text.strip()

    def save_boilerplate_map(self, json_path: str):
        out_data = {domain: sorted(list(data.boilerplate)) for domain, data in self.domain_data_map.items()}
        if not out_data:
            print(termcolor("WARNING: domain boilerplate data empty when saving to JSON!", color="yellow"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(termcolor.colored(f"Saved extracted domain boilerplate to {json_path}.", color="green"))

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

