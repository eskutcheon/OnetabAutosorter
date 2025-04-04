import re
import json
import os
from collections import defaultdict, Counter
from thefuzz import fuzz
from typing import List, Dict, Any


IGNORE_PATTERNS = [
    r"javascript.*required",
    r"enable.*javascript",
    r"cookies.*enable",
    r"(404|403) error",
    r"access denied",
    r"your browser.*not supported",
    r"you are being redirected",
]

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        norm = line.lower().strip()
        # norm.startswith("#") or norm.startswith("//") or norm.startswith("<!--") or norm.endswith("-->")
        # drop short lines or lines matching known spam patterns
        if not norm or any(re.search(pattern, norm) for pattern in IGNORE_PATTERNS) or (len(norm.split()) < 3):
            continue
        cleaned.append(line.strip())
    return "\n".join(cleaned)


def remove_near_duplicates(lines: list[str], threshold: int = 85) -> list[str]:
    if not lines:
        return []
    output = [lines[0]]
    for line in lines[1:]:
        if all(fuzz.ratio(line, seen) < threshold for seen in output):
            output.append(line)
    return output


def preprocess_html_text(raw: str) -> str:
    # remove near-duplicate lines, then drop short or spammy lines
    lines = raw.splitlines()
    lines = remove_near_duplicates(lines)
    return clean_text("\n".join(lines))


def get_base_title_text(entry: Dict[str, Any]) -> str:
    """ strips out domain tokens from the page title to reduce noise in KeyBERT extraction """
    title = entry.get("title", "")
    domain = entry.get("domain", "")
    if not domain: # really shouldn't happen, but just in case
        return title.strip()
    # Remove domain tokens from title before keyword extraction
    base_domain = sorted(domain.lower().split('.'), key=len)[-1]
    pattern = r'\b' + re.escape(base_domain) + r'\b'
    return re.sub(pattern, '', title, flags=re.IGNORECASE).strip()




class DomainBoilerplateFilter:
    """ Tracks repeated text snippets for each domain. Once a domain has enough samples
        to determine its repeated boilerplate, it locks that boilerplate for future filtering.
    """
    def __init__(self, min_domain_samples=5, min_repeat_count=2):
        """
            :param min_domain_samples: Minimum number of samples (entries) from a domain
                                    before we finalize boilerplate phrases for that domain.
            :param min_repeat_count:   How many times a phrase must be observed (across
                                    multiple entries) to be considered boilerplate.
        """
        self.min_domain_samples = min_domain_samples
        self.min_repeat_count = min_repeat_count
        # domain -> list of raw text from each entry
        self.domain_texts: Dict[str, List[str]] = defaultdict(list)
        # domain -> set of boilerplate phrases (after locking)
        self.boilerplate_map: Dict[str, set] = {}
        # track whether a domain has been locked
        self.domain_locked: Dict[str, bool] = defaultdict(bool)
        self.domain_sample_count: Counter = Counter()

    def add_entry_text(self, domain: str, text: str):
        """ Adds raw text from an entry to the domain's text list (if domain is not locked) """
        if self.domain_locked[domain]:
            return
        self.domain_texts[domain].append(text)
        self.domain_sample_count[domain] += 1
        # if there are enough examples for this domain, finalize boilerplate text for reference
        if self.domain_sample_count[domain] >= self.min_domain_samples:
            self._finalize_boilerplate(domain)

    def force_finalize_all(self):
        """ Forcibly finalize boilerplate for any domains not yet locked (to call after gathering all the text) """
        for domain, locked in self.domain_locked.items():
            if not locked:
                self._finalize_boilerplate(domain)

    def _finalize_boilerplate(self, domain: str):
        """ once there are enough domain samples, determine repeated phrases and lock the domain to avoid new text appends """
        if self.domain_locked[domain]:
            return  # already finalized
        all_texts = self.domain_texts[domain]  # list of entire text from each entry
        # count occurrences across all entries, then consider repeated lines as boilerplate.
        lines_counter = Counter()
        for text in all_texts:
            unique_lines = set([l.strip() for l in text.splitlines() if l.strip()])
            for line in unique_lines:
                lines_counter[line] += 1
        # Build the set of lines that occur in enough pages to be considered boilerplate
        repeated = {line for (line, cnt) in lines_counter.items() if cnt >= self.min_repeat_count}
        self.boilerplate_map[domain] = repeated
        self.domain_locked[domain] = True
        self._flush_text_buffer(domain)  # free memory by dropping the raw text buffer

    def _flush_text_buffer(self, domain):
        """ drop large text buffers for a domain for memory efficiency """
        del self.domain_texts[domain]

    def filter_boilerplate(self, domain: str, text: str) -> str:
        """ Removes known boilerplate from the text if the domain is locked, else returns text as-is. """
        if not self.domain_locked[domain]:
            return text  # domain not locked yet, no known boilerplate
        # Either do line-by-line removal or try substring removal later
        repeated = self.boilerplate_map.get(domain, set())
        filtered_lines = [ln for ln in text.splitlines() if ln.strip() not in repeated]
        return "\n".join(filtered_lines)

    @classmethod
    def load_boilerplate_map(cls, json_path: str, min_domain_samples=5, min_repeat_count=2):
        """ Load an existing domain -> list-of-lines from disk. The JSON structure
            is assumed to be: { "domain1": [...], "domain2": [...], ... }
            Returns:
                DomainBoilerplateFilter (with domain_locked=True for all loaded domains)
        """
        obj = cls(min_domain_samples=min_domain_samples, min_repeat_count=min_repeat_count)
        if not os.path.isfile(json_path):
            return obj
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for domain, lines in data.items():
            if not isinstance(lines, list):
                continue
            # treat 'lines' as a set of boilerplate lines
            obj.boilerplate_map[domain] = set(lines)
            obj.domain_locked[domain] = True
        return obj

    def save_boilerplate_map(self, json_path: str):
        """ Save the domain -> boilerplate_lines map to disk as { "domain": [line, ...], ... } """
        # convert each set(...) to a sorted list for JSON
        data = {dom: sorted(list(lines)) for dom, lines in self.boilerplate_map.items()}
        # TODO: might just want to import the PythonSetEncoder to handle this
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)