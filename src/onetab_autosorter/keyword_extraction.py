import re
import os
import json
from tqdm import tqdm
import numpy as np
from termcolor import colored
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterable
# NLP topic models and backends
from keybert import KeyBERT
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
# local imports
from onetab_autosorter.utils.utils import is_internet_connected
from onetab_autosorter.scraper.scraper_utils import default_html_fetcher, default_html_fetcher_batch
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.utils.clean_utils import get_base_title_text


class BaseKeywordModel:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        preprocessor: Optional[TextPreprocessingHandler] = None,
        fetcher_fn: Optional[Callable] = None
    ):
        self.model_name = model_name
        self.preprocessor = preprocessor
        self.fetcher_fn = fetcher_fn
        self.is_connected = is_internet_connected()
        if not self.is_connected:
            print(colored('WARNING: No internet connection detected. Supplemental HTML fetching will be disabled.', color="yellow"))

    @staticmethod
    def refine_keywords(keywords: List[Tuple[str, float]], min_score: float = 0.2) -> Dict[str, float]:
        # TODO: add Levenshtein similarity ratios back in to filter extremely similar keywords
            # e.g. for Wikipedia's "Koch Snowflake" page with "koch snowflake limits" and "koch snowflake limit" - problem only appeared when using maxsum with KeyBERT
        """ Refines a list of (keyword, score) pairs by removing low scores, redundancy, and token overlap. """
        seen_tokens = set()
        refined = {}
        for kw, score in sorted(keywords, key=lambda x: -x[1]):
            if score < min_score:
                continue
            normalized = re.sub(r'\b(\w+)( \1\b)+', r'\1', kw.lower())  # remove internal keyword repetition
            tokens = frozenset(normalized.split()) # split by spaces and create a frozenset of tokens
            # skip if all tokens already covered by a previously accepted phrase
            if tokens & seen_tokens == tokens:
                continue
            # update the refined dictionary with the normalized keyword and its score
            refined[normalized] = round(score, 4)
            seen_tokens.update(tokens)
        return refined

    @staticmethod
    def dump_json(data: Dict, path: str, label: str = ""):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding="utf-8") as fptr:
                json.dump(data, fptr, indent=2, ensure_ascii=False)
            print(colored(f"{label} dumped to {path}", "green"))
        except Exception as e:
            print(colored(f"[ERROR] Failed to dump {label} to {path}: {e}", "red"))


    def _prep_for_extraction(self, entry: Dict[str, Any], raw_text: str) -> str:
        """ combine the entry's title with the domain-filtered text then apply token truncation """
        domain = entry.get("domain", "")
        title_str = get_base_title_text(entry)
        # merge the cleaned/filtered text
        combined = (title_str + " " + raw_text).strip()
        final_text = self._preprocess_text(domain, combined)
        return final_text.strip()  # ensure no leading/trailing whitespace is left

    def _preprocess_text(self, domain: str, full_text: str) -> str:
        """ Apply domain-level boilerplate filtering (if domain is locked) and then truncate to max_tokens """
        if self.preprocessor:
            # If domain is locked or partially locked, do the filtering - filter_boilerplate() will be a no-op if domain isn't locked yet
            full_text = self.preprocessor.process_text(full_text, domain, use_domain_filter=True)
        return full_text

    def run(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method.")





class KeyBertKeywordModel(BaseKeywordModel):
    """ wrapper class for KeyBERT - retrieves keywords from a given text (supplemented with response headers) using KeyBERT """
    MAX_PHRASE_LENGTH = 3

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        top_k=10,
        preprocessor: Optional[TextPreprocessingHandler] = None,
        fetcher_fn : Optional[Callable] = None
    ):
        """
            :param model_name: e.g. "all-MiniLM-L6-v2" (for the underlying sentence-transformer in KeyBERT)
            :param prefetch_factor: number of entries to prefetch in parallel (default: 1)
            :param preprocessor: optional TextPreprocessingHandler for domain filtering and text cleaning
            :param fetcher_fn: function to fetch summary text for each URL in batch
        """
        self.model = KeyBERT(model=model_name)
        self.top_k = top_k
        self.preprocessor = preprocessor
        # if fetcher_fn is None, set it to default fetcher based on prefetch_factor
        self.fetcher_fn = fetcher_fn #or (default_html_fetcher if self.prefetch_factor == 1 else default_html_fetcher_batch)
        self.is_connected = is_internet_connected()
        if not self.is_connected:
            print(colored('WARNING: No internet connection detected. Supplemental HTML fetching will be disabled.', color="yellow"))


    def generate(self, entry: Dict[str, Any], supplement_text: Optional[str] = None) -> Dict[str, Any]:
        text = self._prep_for_extraction(entry, supplement_text)
        if not text:
            entry["keywords"] = {}
            return entry
        raw = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, self.MAX_PHRASE_LENGTH),
            use_mmr=True,
            #use_maxsum=True,
            nr_candidates=20,   # TODO: might want to make this derived from the text length in some way
            stop_words="english",
            top_n=self.top_k
        )
        # filtering domain keywords since they could be present from the supplementary text (keep singleton domain name though)
        refined = self.refine_keywords(raw)
        entry["keywords"] = refined
        return entry


    def _generate_stream(self, entries: List[Dict[str, Any]], summaries: Optional[Dict[str, str]] = None):
        for entry in entries:
            summary = summaries.get(entry["url"], "") if summaries else None
            yield self.generate(entry, supplement_text = summary)

    def generate_stream(self, entries: List[Dict[str, Any]]) -> deque:
        """ Process entries using a generator, no prefetching """
        return deque(self._generate_stream(tqdm(entries, desc="Processing entries", unit="entry")))

    def generate_batch(self, entries: List[Dict[str, Any]], summaries: Dict[str, str]) -> deque:
        """ Process entries in batch with pre-fetched supplemental summaries (with either Python or Java backend) """
        return deque(self._generate_stream(entries, summaries))

    #? NOTE: chunking will be slower than single fetches until I remove the conditional logic in `_append_supplementary_text`
    def generate_with_chunking(self, entries: List[Dict[str, Any]], chunk_size: int = 20) -> deque:
        """ Chunk entries and process with batch fetching """
        full_results = deque()
        for i in tqdm(range(0, len(entries), chunk_size), desc="Keyword extraction of entries in chunks", unit="chunk"):
            chunk = entries[i:i+chunk_size]
            summaries = None
            if self.is_connected and self.fetcher_fn:
                summaries = self.fetcher_fn([e["url"] for e in chunk])
                #print("SUMMARY KEYS: ", summaries.keys() if summaries else "No summaries fetched")
            full_results.extend(self.generate_batch(chunk, summaries))
        return full_results


    def run(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Unified runner that fetches, extracts keywords, and prepares data for downstream embedding or clustering """
        # providing this to have a common interface with BERTTopic - should still call one of the generator functions
        if not self.fetcher_fn:
            raise RuntimeError("No fetcher_fn provided to fetch supplemental text.")
        urls = [entry["url"] for entry in entries]
        summaries = self.fetcher_fn(urls)
        if not summaries:
            print(colored("Warning: No summaries returned from fetcher.", "yellow"))
            summaries = {url: "" for url in urls}
        updated = []
        for entry in tqdm(entries, desc="Running KeyBERT", unit="entry"):
            summary = summaries.get(entry["url"], "")
            updated_entry = self.generate(entry, supplement_text=summary)
            updated.append(updated_entry)
        # TODO (later): convert to DataFrame + embed + cluster
        return updated




class BERTTopicKeywordModel(BaseKeywordModel):
    """ BERTTopic-based wrapper class that:
        1. Optionally fetches text for each entry (with a fetcher_fn)
        2. Applies domain-based boilerplate filtering (DomainBoilerplateFilter)
        3. Concatenates the result with the entry's title
        4. Truncates text to a max token limit
        5. Runs BERTopic .fit_transform(...) over the entire corpus
        6. Assigns topic labels and representative topic keywords to each entry
        #!!! Unlike KeyBERT, BERTTopic is not doc-by-doc. It needs the entire corpus in one pass. !!!
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        nr_topics: Union[int, str, None] = None,
        preprocessor: Optional[TextPreprocessingHandler] = None,
        # MIGHT NOT BE ABLE TO USE THIS SINCE THE CONFIDENCE IS OFTEN ABOUT THIS LOW
        sort_confidence_threshold = 0.01,  # threshold for sorting topic confidence scores #! setting threshold low while experimenting - update later
        fetcher_fn: Optional[Callable] = None
    ):
        """
            :param model_name: e.g. "all-MiniLM-L6-v2" (for the underlying sentence-transformer in BERTopic)
            :param nr_topics: If you want to reduce the number of topics, you can set nr_topics to something like "auto" or an integer.
            :param prefetch_factor: parallels KeyBertKeywordModel usage for chunking/fetching
            :param fetcher_fn: function to fetch summary text for each URL in batch
        """
        # TODO: try a more powerful backbone model if needed, e.g. "paraphrase-multilingual-MiniLM-L12-v2" or "all-distilroberta-v1"
            # try embedding_model = "allenai/scibert_scivocab_uncased" # for scientific papers or domain-specific texts
        # TODO: also experiment with `zeroshot_topic_list` or `seed_topic_list`
        # TODO: experiment with `ctfidf_model` for custom ClassTfidfTransformer and
            # `representation_model` to fine-tune the topic representations from c-TF-IDF. Models from bertopic.representation
        # TODO: also try replacing the default vectorizer model, like sklearn.feature_extraction.text.CountVectorizer
            # custom_vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[A-Za-z]+\b")
        self.topic_model = BERTopic(
            nr_topics = nr_topics, # can be an integer for number of topics or "auto" for automatic topic reduction
            embedding_model = SentenceTransformer(model_name),
            calculate_probabilities = True,
            n_gram_range = (1, 3),  # unigrams, bigrams, and trigrams
            verbose = True
        )
        # optionally store a preprocessor that handles text cleaning and domain filtering
        self.preprocessor = preprocessor
        # set the confidence threshold for topic probabilities
        self.topic_prob_threshold = sort_confidence_threshold
        # for optional text fetching in a single batch
        self.fetcher_fn = fetcher_fn or default_html_fetcher_batch
        self.is_connected = is_internet_connected()
        if not self.is_connected:
            # TODO: might be better to raise a Runtime error since we otherwise would have practically nothing to go on
            print(colored('WARNING: No internet connection detected. Supplemental HTML fetching will be disabled.', color="yellow"))

    def _add_topics_to_entries(self, entries: List[Dict[str, Any]], all_topics: Dict[str, Tuple[str, float]], topics: List[int], probs: np.ndarray) -> None:
        for entry, topic_id, prob in zip(entries, topics, probs):
            # Calculate the highest topic probability
            topic_prob = float(round(prob, 4))
            # Mark as outlier if below the threshold
            if topic_prob < self.topic_prob_threshold:
                entry["topic_id"] = -1
                entry["topic_prob"] = 0.0
                entry["topic_keywords"] = {}
            else:
                # Assign topic ID and probability
                entry["topic_id"] = topic_id
                entry["topic_prob"] = topic_prob
                # Retrieve and refine keywords for the topic
                word_tuples = all_topics.get(topic_id, [])
                entry["topic_keywords"] = self.refine_keywords(word_tuples, min_score=self.topic_prob_threshold)


    def fit_transform_entries(self, entries: List[Dict[str, Any]]) -> None:
        """
            1. Optionally fetch text for each entry (if we have an internet connection).
            2. Construct a doc for each entry, applying domain filtering and max token truncation.
            3. Run BERTTopic .fit_transform(...) on the entire corpus.
            4. Assign each entry's topic_id + top topic keywords in the 'topic_keywords' field.
        """
        # 1) Possibly fetch all text in one batch
        summaries_map = {}
        #print(colored("Fetching webpage content of all urls...", color="green"))
        if self.is_connected and self.fetcher_fn:
            urls = [e["url"] for e in entries]
            # fetch text for each URL (e.g. domain-filtered or raw HTML)
            summaries_map = self.fetcher_fn(urls)
        #self.dump_json(summaries_map, "output/sample_text_log10.json", label="Raw summaries")
        # 2) Build corpus
        for e in tqdm(entries, desc="Cleaning Entries for Corpus"):
            # Get the text from summaries_map if present, else empty
            raw_text = summaries_map.get(e["url"], "")
            doc_text = self._prep_for_extraction(e, raw_text)
            summaries_map[e["url"]] = doc_text  # update the map with the cleaned text
        self.dump_json(summaries_map, "output/sample_cleaned_log11.json", label="Cleaned summaries")
        corpus = list(summaries_map.values())  # list of cleaned text for each entry
        # 3) Run .fit_transform
        print(colored("Fitting corpus to the topic model...", color="green"))
        # topics: List[int], probs: np.ndarray
        topics, probs = self.topic_model.fit_transform(corpus)
        all_topics = self.topic_model.get_topics()  # dict: {topic_id -> [(word, weight), ...], ...}
        # 4) Store topic_id & topic keywords
        self._add_topics_to_entries(entries, all_topics, topics, probs)

    ##############################################################################################################################
    # to mirror the KeyBert approach, define a generate method. In practice, BERTTopic is best used in one shot on all docs.
    ##############################################################################################################################

    def run(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ convenience method that calls fit_transform_entries(...) and returns the updated entries """
        self.fit_transform_entries(entries)
        return entries

    def generate(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """ Satisfies the 'BaseKeywordModel' interface. But BERTTopic doesn't do doc-by-doc extraction, so do nothing here. """
        return entry