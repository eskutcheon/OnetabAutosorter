
import os
import re
import nltk
import unicodedata
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
        #lang_filter = "en" #! FIXME: not in use right now but needs to be incorporated for language detection and stopword filtering
    ):
        # Make sure resources are available
        nltk.download('punkt', quiet=True)
        # TODO: move stopwords to the enclosing handler and pass to both this and the domain filter
        nltk.download('stopwords', quiet=True)
        self.min_word_count = min_word_count
        #self.lang_filter = lang_filter
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
            text = pattern.sub(' ', text)
        # Tokenize and filter stopwords
        #? NOTE: This can be done within the tokenizer, but it's kind of preferable to limit the tokens without stopwords
        tokens = [t for t in self.tokenizer(text) if t.lower() not in self.stopwords]
        if len(tokens) > max_num_tokens:
            tokens = tokens[:max_num_tokens]
        if len(tokens) < self.min_word_count:
            return ""
        return ' '.join(tokens)


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




class EnhancedTextCleaningFilter(TextCleaningFilter):
    """
    Enhanced text cleaning filter with more advanced text normalization and better token handling.
    """
    def __init__(
        self,
        min_word_count: int = 3,
        ignore_patterns: List[Union[str, re.Pattern]] = None,
        preserve_entities: bool = True,
        use_pos_filtering: bool = True
    ):
        super().__init__(min_word_count, ignore_patterns)
        self.preserve_entities = preserve_entities
        self.use_pos_filtering = use_pos_filtering
        self.pos_tagger = None
        # Download required NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        if use_pos_filtering:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            # initialize Perceptron Tagger if use_pos_filtering is enabled
            try:
                from nltk.tag import PerceptronTagger
                self.pos_tagger = PerceptronTagger()
                print("Successfully loaded perceptron tagger")
            except (ImportError, LookupError) as e:
                print(f"Warning: Could not load POS tagger: {e}")
                print("Falling back to basic filtering...")
                self.use_pos_filtering = False

    def _is_valid_token(self, token: str) -> bool:
        """ Check if a token is valid based on length and content. """
        return len(token) > 1 and not token.isdigit() and token.lower() not in self.stopwords

    def filter(self, text: str, max_num_tokens: int = 400) -> str:
        """ Clean and normalize text with advanced filtering techniques. """
        if not text:
            return ""
        # Apply ignore patterns first for efficiency
        for pattern in self.ignore_patterns:
            text = pattern.sub(' ', text)
        # Text normalization preprocessing
        text = self._normalize_text(text)
        # Tokenize the text
        tokens = self.tokenizer(text)
        # Apply POS filtering if enabled
        if self.use_pos_filtering and self.pos_tagger:
            tokens = self._filter_by_pos_tags(tokens)
        else:
            # Simple stopword filtering
            # TODO: make a function for the conditional - used elsewhere as well
            tokens = [t for t in tokens if self._is_valid_token(t)]
        # Limit tokens if needed
        if len(tokens) > max_num_tokens:
            tokens = tokens[:max_num_tokens]
        # Check if we have enough tokens
        if len(tokens) < self.min_word_count:
            return ""
        return ' '.join(tokens)

    def _normalize_text(self, text: str) -> str:
        """ Apply text normalization techniques """
        # convert Unicode to ASCII
        # TODO: move this step earlier to the webscraping stage
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        # patterns for common typographical issues
        text = re.sub(r'[\u2018\u2019\u201c\u201d]', '"', text)  # Smart quotes
        text = re.sub(r'[\u2013\u2014]', '-', text)  # Em/en dashes
        # preserve paragraph breaks but normalize other whitespace
        text = re.sub(r'(\n\s*\n)', '\n\n', text)
        # add space around punctuation that might connect words
        text = re.sub(r'([;:,.!?()])', r' \1 ', text)
        # fix camelCase and PascalCase words that might be running together
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # # remove URLs completely (adding a space to preserve word boundaries)
        # text = re.sub(r'https?://\S+', ' ', text)
        # text = re.sub(r'www\.\S+', ' ', text)
        # horizontal whitespace only (preserving line breaks)
        text = re.sub(r'[ \t]+', ' ', text)
        # final cleanup of excess spaces
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _filter_by_pos_tags(self, tokens: List[str]) -> List[str]:
        """ Filter tokens based on part-of-speech tags using NLTK's PerceptronTagger to keep only content-rich words
            Keep token if:
                1. It's a content word (matches one of our POS tags)
                2. Not a stopword
                3. Not just a number
                4. Not too short
        """
        try:
            # generate part-of-speech tags from the tokens
            pos_tags = self.pos_tagger.tag(tokens) if self.pos_tagger else nltk.pos_tag(tokens)
            # Keep only content-bearing parts of speech (NN: noun, JJ: adjective, VB: verb, RB: adverb)
            content_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']
            filtered_tokens = []
            for token, tag in pos_tags:
                # check if the token is a content word and meets other criteria
                # TODO: make this a function too
                if tag[:2] in content_pos and self._is_valid_token(token):
                    filtered_tokens.append(token)
            return filtered_tokens
        except Exception as e:
            # If POS tagging fails, fall back to basic filtering
            print(f"POS tagging failed: {e}. Falling back to basic filtering.")
            return [t for t in tokens if self._is_valid_token(t)]