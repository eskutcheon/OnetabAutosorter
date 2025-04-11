
import os
import re
import nltk
from typing import Optional, Union, List, Any, Set
# NLP tools
# from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# local imports




class TextCleaningFilter:
    """ class for the most general text cleaning steps (after HTML extraction)
        - Removes lines matching certain ignore patterns
        - Removes lines that are shorter than a min word count
        - Removes optional advanced patterns (like LaTeX, citations, or non-English chars)
        - Performs stopword removal
    """
    def __init__(
        self,
        min_word_count: int = 3,  # minimum number of words in a line to keep it (for spammy line removal)
        ignore_patterns: List[Union[str, re.Pattern]] = None,
        lang_filter = "en"
    ):
        # Make sure resources are available
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.min_word_count = min_word_count
        self.lang_filter = lang_filter
        # TODO: reference a language mapping to be consistent with `lang_filter`
        self.stopwords: Set[str] = set(stopwords.words("english"))
        self.ignore_patterns: List[re.Pattern] = []
        if ignore_patterns:
            self._register_ignore_patterns(ignore_patterns)
        self.tokenizer = word_tokenize

    # might want to apply the regex replacements over the whole text at once
    def filter(self, text: str, max_num_tokens: int = 200) -> str:
        """ Clean and return a normalized, truncated text string """
        if not text:
            return ""
        # Comprehensive text cleaning via regex (all at once for efficiency)
        for pattern in self.ignore_patterns:
            text = pattern.sub('', text)
        # Tokenize and filter stopwords
        #? NOTE: This can be done within the tokenizer, but it's kind of preferable to limit the tokens without stopwords
        tokens = [t for t in self.tokenizer(text) if t.lower() not in self.stopwords]
        if len(tokens) > max_num_tokens:
            tokens = tokens[:max_num_tokens]
        if len(tokens) < self.min_word_count:
            return ""
        return self.clean_text(' '.join(tokens))


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

    @staticmethod
    def clean_text(text: str) -> str:
        # # Remove metadata patterns
        # for pattern in pattern_cfg.NAVIGATION_PATTERNS:
        #     text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        # # Remove unwanted formatting
        # for pattern in pattern_cfg.FORMATTING_PATTERNS:
        #     text = re.sub(pattern, '', text)
        # Remove orphaned punctuation and excessive newlines
        # TODO: integrate into filter patterns or just remove entirely - seems like something any tokenizer should handle
        text = re.sub(r'(\s[.,;:!?])|([.,;:!?]\s)', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text).strip()
        return text