
import re
from thefuzz import fuzz
from typing import List, Dict, Union, Set, Optional, Any


#& only used by preprocess_html_text() to remove near-duplicate lines based on Levenshtein similarity ratio
def _remove_near_duplicates(lines: List[str], threshold: int = 85) -> list[str]:
    if not lines:
        return []
    # returns just the first line if it's only one
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
    #!!! Consider removing in favor of TF-IDF approach
    lines = _remove_near_duplicates(lines)
    text = "\n".join(lines) # _clean_text(lines)
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

