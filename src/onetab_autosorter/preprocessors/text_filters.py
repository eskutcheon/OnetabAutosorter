
import os
import re
# from itertools import islice
from typing import Optional, Union, List, Any, Set
from bs4 import BeautifulSoup, Tag


# Some default regex patterns for ignoring noise or spam:
DEFAULT_IGNORE_PATTERNS = [
    r"^javascript.*required$",
    r"enable\s+javascript",
    r"cookies\s+.*enable",
    r"(404|403)\s+error",
    r"access\s+denied",
    r"your\s+browser.*not\s+supported",
    r"you\s+are\s+being\s+redirected"
]



def latex_remover(tag: Tag) -> bool:
    # Example: remove tags that might store LaTeX code
    # or if 'class' in tag.attrs and 'latex' in tag['class'] ...
    if tag.name == "span" and "latex" in (tag.get("class") or []):
        return True
    return False


#? not sure if I'm going to be using this at all
def strip_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """ Remove script, style, noscript tags and other non-visible elements from the BeautifulSoup object """
    #! literally just does the same thing as soup.get_text(), but might be worthwhile later
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = str(soup.prettify(formatter="minimal")).splitlines()
    text = ""
    for line in lines:
        if not line.strip().startswith("<"):
            text += line.strip() + "\n"
    return text



class TextCleaningFilter:
    """ class for the most general text cleaning steps (after HTML extraction)
        - Removes lines matching certain ignore patterns
        - Removes lines that are shorter than a min word count
        - Removes optional advanced patterns (like LaTeX, citations, or non-English chars)
        - Performs stopword removal
    """
    DEFAULT_STOPWORDS_FILE = r"resources/nltk_stopwords.txt"  # default path to stopwords file
    # support for removing unwanted patterns like LaTeX expressions, citations in brackets, and Non-English characters (non-ASCII or a certain set)
    # TODO: may add these to a global `OPTIONAL_IGNORE_PATTERNS` list later
    #LATEX_PATTERN = re.compile(r"\$[^$]+\$") # TODO: add more robust LaTeX detection if needed (like \begin{*}...\end{*} or \[...\] etc.)
    LATEX_PATTERN = re.compile(r"\$[^$]+\$", re.MULTILINE)
    CITATION_PATTERN = re.compile(r"\[[^\]]*\]", re.MULTILINE)  # Matches citations like [1], [2,3], or [1, 2, 3] etc.
    # extremely naive approach, removing non-(a-z0-9 punctuation) chars
    NON_ENGLISH_PATTERN = re.compile(r"[^a-zA-Z0-9\s\.,!\?]", re.MULTILINE)  # Matches any character that is not a letter, digit, space, or common punctuation

    def __init__(
        self,
        min_word_count: int = 3,  # minimum number of words in a line to keep it (for spammy line removal)
        ignore_patterns: List[Union[str, re.Pattern]] = None,
        remove_latex: bool = True,
        remove_citations: bool = True,
        remove_non_english: bool = True,
        # TODO: change this to accept the set directly instead of a file path (or input NLTK stopwords directly from the `nltk` module)
        stopword_file: Optional[str] = None
    ):
        self.min_word_count = min_word_count
        self.ignore_patterns: List[re.Pattern] = []
        if ignore_patterns:
            self._register_ignore_patterns(ignore_patterns)
        self._register_ignore_patterns(DEFAULT_IGNORE_PATTERNS)  # always register default ignore patterns (let it come after user-defined ones)
        if remove_latex:
            self._register_ignore_patterns(self.LATEX_PATTERN)  # add LaTeX pattern to ignore list if enabled
        if remove_citations:
            self._register_ignore_patterns(self.CITATION_PATTERN)
        if remove_non_english:
            self._register_ignore_patterns(self.NON_ENGLISH_PATTERN)
        stopword_file = stopword_file or os.path.join(os.path.dirname(__file__), self.DEFAULT_STOPWORDS_FILE)
        self.stopwords = self._get_stopword_set(stopword_file)


    def filter(self, text: str, max_num_tokens: int = 1000) -> str:
        """ Main method to clean text by removing stopwords. """
        if not text:
            return text
        for pattern in self.ignore_patterns:
            text = re.sub(pattern, "", text)  # remove lines matching ignore patterns
        # strip extra whitespace
        #text = re.sub(r'\s+', ' ', text).strip()  # replace multiple spaces with a single space
        # remove lines that are too short
        lines = text.splitlines()
        cleaned_lines = []
        total_tokens = 0
        for line in lines:
            # breaks from the loop and ends text filtering if total_tokens exceeds max_num_tokens
            if total_tokens >= max_num_tokens:
                break
            ln_stripped = line.strip()
            tokens = ln_stripped.split()
            # if line is empty after stripping whitespace or if the number of distinct tokens is less than min_word_count, skip this line
            if not ln_stripped or len(tokens) < self.min_word_count:
                continue
            # Check ignore patterns
            #!! GPT wrote this and it may be wrong since this checks within the line, but the regex pattern specifies multiline
            # if self._matches_ignore_pattern(ln_stripped):
            #     continue
            # remove stopwords in line
            final_tokens = self._filter_stopwords_from_tokens(tokens)  # filter out stopwords
            if not final_tokens: # if empty list after stopword removal
                continue
            cleaned_lines.append(" ".join(final_tokens))
            total_tokens += len(final_tokens)
        return "\n".join(cleaned_lines)


    def _register_ignore_patterns(self, patterns: Union[str, re.Pattern, List[Union[str, re.Pattern]]]) -> None:
        if isinstance(patterns, str):
            self.ignore_patterns.append(re.compile(patterns, re.IGNORECASE))
        elif isinstance(patterns, re.Pattern):
            self.ignore_patterns.append(patterns)
        elif isinstance(patterns, list):
            assert(all(isinstance(p, (str, re.Pattern)) for p in patterns)), "All items in ignore_patterns list must be strings or regex patterns."
            self.ignore_patterns.extend([re.compile(p, re.IGNORECASE) if isinstance(p, str) else p for p in patterns])
        else:
            raise ValueError(f"ignore patterns must be a string, regex pattern, or a list of strings/regex patterns; got {type(patterns)}")

    def _get_stopword_set(self, file_path: str) -> Set[str]:
        if not os.path.isfile(file_path):
            return []  # return empty list if the file doesn't exist
        with open(file_path, "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        stopwords = [sw.strip().lower() for sw in stopwords if sw.strip()]  # clean and lower-case
        return set(stopwords)


    def _matches_ignore_pattern(self, line: str) -> bool:
        for pattern in self.ignore_patterns:
            if pattern.search(line.lower()):
                return True
        return False

    def _filter_stopwords_from_text(self, text: str) -> str:
        """ Filter out stopwords from a text string. """
        tokens = self._filter_stopwords_from_tokens(text.split())
        if not tokens:
            return ""
        return " ".join(tokens)  # filter out stopwords

    def _filter_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """ Filter out stopwords from a list of tokens. """
        return [t for t in tokens if t.lower() not in self.stopwords]