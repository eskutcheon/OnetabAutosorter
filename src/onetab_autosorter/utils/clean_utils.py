
import re
import os
from thefuzz import fuzz
from typing import List, Dict, Union, Set, Optional, Any


IGNORE_PATTERNS = [
    r"javascript.*required",
    r"enable.*javascript",
    r"cookies.*enable",
    r"(404|403) error",
    r"access denied",
    r"your browser.*not supported",
    r"you are being redirected",
]


#& UNUSED - general utility function to check if a string is mostly stopwords - should move to the TextCleaningFilter class later
# def is_mostly_stopwords(ngram: str, stopwords: List[str], threshold_ratio=0.75) -> bool:
#     tokens = ngram.split()
#     if not tokens:
#         return False
#     sw_count = sum(1 for t in tokens if t in stopwords)
#     return (sw_count / len(tokens)) >= threshold_ratio

#& only used by `extract_phrases_by_line` to determine repeated phrases in n-grams - should become part of the domain filter if anything
def _get_ngrams(tokens: List[str], n: int) -> Set[str]:
    """ Return the set of n-grams from a list of tokens. """
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}



# TODO: if adding more of these functions to a new Filtering class, this would be the default cleaning operations (w/IGNORE_PATTERNS)
#& general text cleaning step for removing spammy lines and short lines (under 3 words) from raw text - should move to the TextCleaningFilter class later
def _clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        norm = line.lower().strip()
        # norm.startswith("#") or norm.startswith("//") or norm.startswith("<!--") or norm.endswith("-->")
        # drop short lines or lines matching known spam patterns
        if not norm or any(re.search(pattern, norm) for pattern in IGNORE_PATTERNS) or (len(norm.split()) < 3):
            continue
        cleaned.append(norm) #line.strip())
    return "\n".join(cleaned)

#& only used by preprocess_html_text() to remove near-duplicate lines based on Levenshtein similarity ratio
def _remove_near_duplicates(lines: list[str], threshold: int = 85) -> list[str]:
    if not lines:
        return []
    output = [lines[0]]
    for line in lines[1:]:
        # TODO: look into using a partial ratio or token_sort_ratio for better near-duplicate detection
        if all(fuzz.ratio(line, seen) < threshold for seen in output):
            output.append(line)
    return output

#& called by the webscraping function before passing back cleaned text to the main processing pipeline - should be moved to the TextCleaningFilter class later
def preprocess_html_text(raw: Union[str, List[str]]) -> str:
    # remove near-duplicate lines, then drop short or spammy lines
    lines = raw.splitlines() if isinstance(raw, str) else raw
    lines = _remove_near_duplicates(lines)
    text = _clean_text("\n".join(lines))
    return text


#& general text cleaning step for removing redundant domain titles from entry data titles - should move to the TextCleaningFilter class later
def get_base_title_text(entry: Dict[str, Any]) -> str:
    """ strips out domain tokens from the page title to reduce noise in KeyBERT extraction """
    title = entry.get("title", "")
    domain = entry.get("domain", "")
    if not domain: # really shouldn't happen, but just in case
        return title.strip()
    # Remove domain tokens from title before keyword extraction
    # TODO: replace with a argmax approach later for efficiency
    base_domain = sorted(domain.lower().split('.'), key=len)[-1]
    pattern = r'\b' + re.escape(base_domain) + r'\b'
    return re.sub(pattern, '', title, flags=re.IGNORECASE).strip()


#& relevant to the n-gram extraction and filtering logic for text lines within the domain filter - might move to `preprocessors/domain_filter.py` later
def extract_phrases_by_line(line: str, ngram_range=(2,3)) -> Set[str]:
    """ Tokenize a line, produce n-grams in the given range (e.g. 2-3), and filter out n-grams that are mostly stopwords """
    # Simple tokenization by non-alphabetic
    tokens = re.findall(r"[A-Za-z0-9]+", line.lower())
    phrases = set()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams = _get_ngrams(tokens, n)
        for ng in ngrams:
            phrases.add(ng.strip())
            # if not is_mostly_stopwords(ng, stopwords=stopwords):
            #     phrases.add(ng)
    return phrases

#& relevant to the domain filter and its boilerplate removal logic - might move to `preprocessors/domain_filter.py` later
def assign_line_bin(line_idx: int, num_lines: int, num_bins: int = 5) -> int:
    """ Assigns a line index to a bin based on the total number of lines and the specified number of bins. """
    if num_bins <= 0:
        raise ValueError("Number of bins must be greater than 0.")
    # simple approach for creating unequal bins based on num_lines and num_bins
    if num_lines == 0:
        return 0
    num_bins = min(num_lines, num_bins)  # ensure at least one line per bin (or num_lines >= num_bins)
    bin_size = num_lines // num_bins
    for i in range(num_bins):
        if line_idx < (i + 1) * bin_size:
            return i
    if num_lines % bin_size != 0:  # if there's a remainder, assign to the last bin
        return num_bins - 1 # since it's the bin index starting at 0


#& general text cleaning step - should move to the TextCleaningFilter class later
def get_stopword_list(file_path: str):
    if not os.path.isfile(file_path):
        return []  # return empty list if the file doesn't exist
    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()
    stopwords = [sw.strip().lower() for sw in stopwords if sw.strip()]  # clean and lower-case
    return stopwords

