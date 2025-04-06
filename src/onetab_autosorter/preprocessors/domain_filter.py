import os
import json
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Union, Any
# local imports
from onetab_autosorter.utils.clean_utils import assign_line_bin, extract_phrases_by_line


@dataclass
class DomainFilterData:
    """ Holds all relevant data for a single domain's boilerplate detection. """
    # list of raw page HTML/text built while still collecting
    pages: List[str] = field(default_factory=list)
    # Counter that tracks doc frequency of each phrase in that bin
    bin_phrases: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # set of repeated phrases for that bin (once locked)
    boilerplate: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # indicates whether the domain is finalized and the boilerplate is decided
    locked: bool = False
    # number of samples (webpages) added for this domain
    sample_count: int = 0

    def flush_memory(self):
        """Remove large in-memory structures no longer needed after locking."""
        self.pages.clear()
        self.bin_phrases.clear()

    def __bool__(self):
        """ Allow checking if the domain data is empty or not """
        return bool(self.boilerplate) # should maybe do this with self.locked, but was just thinking of testing for `{}` at the time


class DomainBoilerplateFilter:
    """ Tracks repeated text snippets for each domain, determines repeated boilerplate, and locks it in to filter out those phrases
        from text scraped from websites of that domain in the future.
        - Tracks n-gram phrases by line offset bins across multiple pages
        - If a phrase appears in at least 'min_repeat_count' docs in the same bin, consider it boilerplate for that bin.
    """
    def __init__(self, min_domain_samples=5, min_repeat_count=2, ngram_range=(2,3), num_bins=5):
        """
            :param min_domain_samples: Minimum number of samples (entries) from a domain before we finalize its boilerplate phrases.
            :param min_repeat_count:   How many times to observe a phrase (across multiple entries) to be considered boilerplate.
        """
        self.min_domain_samples = min_domain_samples
        self.min_repeat_count = min_repeat_count
        self.ngram_range = ngram_range
        self.num_bins = num_bins # might want this to be variable later
        self.domains: Dict[str, DomainFilterData] = defaultdict(DomainFilterData)


    def get_domain_data(self, domain: str) -> DomainFilterData:
        """ Returns the DomainFilterData object for a given domain, or None if it doesn't exist """
        return self.domains.get(domain, None)


    def add_entry_text(self, domain: str, text: str):
        """ Adds raw text from an entry to the domain's text list (if domain is not locked) """
        #d = self.domains[domain]
        d = self.domains.setdefault(domain, DomainFilterData())
        if d.locked:
            return
        d.pages.append(text)
        d.sample_count += 1
        # if there are enough examples for this domain, trigger finalize boilerplate text for reference
        if d.sample_count >= self.min_domain_samples:
            self._finalize_boilerplate(domain)

    def force_finalize_all(self):
        """ Forcibly finalize boilerplate for any domains not yet locked (to call after gathering all the text) """
        for domain, data in self.domains.items():
            if not data.locked:
                self._finalize_boilerplate(domain)

    def _finalize_boilerplate(self, domain: str):
        """ once there are enough domain samples, determine repeated phrases and lock the domain to avoid new text appends """
        data = self.domains[domain]
        if data.locked:
            return
        # Mark locked
        data.locked = True
        # for each page, break into lines, assign each line a bin, extract n-grams
        for page_text in data.pages:
            lines = page_text.splitlines()
            self._build_ngram_bins(lines, data) # build n-gram bins for this page
        # Now, for each bin, figure out which phrases appear in >= min_repeat_count pages
        for bin_id, phrase_counter in data.bin_phrases.items():
            repeated = {ph for ph, freq in phrase_counter.items() if freq >= self.min_repeat_count}
            data.boilerplate[bin_id] = repeated
        # flush memory
        data.flush_memory()


    def _build_ngram_bins(self, lines: str, domain_data: DomainFilterData):
        """ build a set of phrases per (bin_id) so we only count each phrase once per doc+bin """
        total_lines = len(lines)
        bin_doc_sets: Dict[int, Set[str]] = defaultdict(set)
        for i, line in enumerate(lines):
            bin_id = assign_line_bin(i, total_lines) # using constant num_bins of 5 for simplicity for now
            phrases = extract_phrases_by_line(line, self.ngram_range)
            bin_doc_sets[bin_id].update(phrases)
        # now update doc frequency in domain_bin_phrases
        # TODO: collapse these loops to avoid multiple dict lookups and speed things up:
        for bin_id, phrase_set in bin_doc_sets.items():
            for ph in phrase_set:
                domain_data.bin_phrases[bin_id][ph] += 1


    def filter_boilerplate(self, domain: str, text: str) -> str:
        """ Removes known boilerplate from the text if the domain is locked, else returns text as-is. """
        data = self.domains[domain]
        if not data.locked or domain not in data.boilerplate:
            return text  # domain not locked yet, or no known boilerplate for a domain, return original text
        lines = text.splitlines()
        total_lines = len(lines)
        filtered_lines = []
        for i, line in enumerate(lines):
            bin_id = assign_line_bin(i, total_lines, self.num_bins)
            # check each phrase in that bin's set
            repeated_phrases = data.boilerplate.get(bin_id, set())
            # remove line if it contains any repeated phrase
            if any(rp in line.lower() for rp in repeated_phrases):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)


    def save_boilerplate_map(self, json_path: str):
        """ Save the domain -> boilerplate_lines map to disk as { "domain": [line, ...], ... } """
        # convert each set(...) to a sorted list for JSON
        out_data = {}
        for domain, data in self.domains.items():
            # only if locked do we have final boilerplate
            if not isinstance(data.boilerplate, dict):
                raise ValueError(f"Invalid data for domain '{domain}': expected dict, got {type(data.boilerplate)}")
            if data.locked:
                domain_map = {}
                for bin_id, phrases in data.boilerplate.items():
                    # convert sets to sorted lists for JSON serialization
                    # TODO: might just want to import the PythonSetEncoder to handle the conversion away from sets
                    domain_map[bin_id] = sorted(list(phrases))
                out_data[domain] = domain_map
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)


    @classmethod
    def load_boilerplate_map(cls, json_path: str, **kwargs) -> "DomainBoilerplateFilter":
        """ Load an existing domain -> list-of-lines from disk. The JSON structure
            is assumed to be: { "domain1": [...], "domain2": [...], ... }
            Returns:
                DomainBoilerplateFilter (with domain_locked=True for all loaded domains)
        """
        obj = cls(**kwargs)  # create a new instance with the same parameters
        if not os.path.isfile(json_path):
            return obj
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = dict(json.load(f))
        # should be ` # loaded = { domain: { bin_id: [phrase, phrase,...], ... }, ...}`
        for domain, bin_map in loaded.items():
            d = obj.domains[domain]
            d.locked = True
            for bin_id_str, phrase_list in bin_map.items():
                bin_id = int(bin_id_str)
                d.boilerplate[bin_id] = set(phrase_list)
        return obj