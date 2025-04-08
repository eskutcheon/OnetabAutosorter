import re
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
from onetab_autosorter.scraper.scraper_utils import default_html_fetcher, default_html_fetcher_batch, SupplementFetcher
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.utils.clean_utils import get_base_title_text


class BaseKeywordModel:
    def generate(self, entry: Dict) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")


class KeyBertKeywordModel(BaseKeywordModel):
    """ wrapper class for KeyBERT - retrieves keywords from a given text (supplemented with response headers) using KeyBERT """
    MAX_PHRASE_LENGTH = 3

    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=10, prefetch_factor: int = 1, fetcher_fn : Optional[SupplementFetcher] = None):
        self.model = KeyBERT(model=model_name)
        self.top_k = top_k
        # TODO: might just remove this argument later since an equivalent argument is passed to `generate_with_chunking`
        self.prefetch_factor = prefetch_factor  # not used in this class, but can be useful for async fetching
        # if fetcher_fn is None, set it to default fetcher based on prefetch_factor
        self.supplement_fetcher = fetcher_fn or (default_html_fetcher if self.prefetch_factor == 1 else default_html_fetcher_batch)
        self.is_connected = is_internet_connected()
        if not self.is_connected:
            print(colored('WARNING: No internet connection detected. Supplemental HTML fetching will be disabled.', color="yellow"))


    def _refine_keywords(self, keywords: List[Tuple[str, float]], min_score=0.2) -> Dict[str, float]:
        """ refines keywords by removing duplicates and low-scoring keywords """
        seen_tokens = set()
        refined = {}
        for kw, score in sorted(keywords, key=lambda x: -x[1]):
            if score < min_score:
                continue
            normalized = re.sub(r'\b(\w+)( \1\b)+', r'\1', kw.lower())  # remove internal repetition
            tokens = frozenset(normalized.split()) # split by spaces and create a frozenset of tokens
            # skip if all tokens already covered by a previously accepted phrase
            if tokens & seen_tokens == tokens:
                continue
            # update the refined dictionary with the normalized keyword and its score
            refined[normalized] = round(score, 4)
            seen_tokens.update(tokens)
        return refined


    # TODO: add to some Webcrawler class that prepares the entries for keyword extraction, so that it can handle both fetching and cleaning
    def _prepare_content_for_extraction(self, entry: Dict[str, Any], supplement_text: Optional[str] = None) -> str:
        """ combine the (possibly domain-filtered) supplement_text with a simplified version of the title """
        text = get_base_title_text(entry)
        text = (text + " " + (supplement_text or "")).strip()
        return text


    def generate(self, entry: Dict[str, Any], supplement_text: Optional[str] = None) -> Dict[str, Any]:
        text = self._prepare_content_for_extraction(entry, supplement_text)
        if not text:
            entry["keywords"] = {}
            return entry
        raw = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, self.MAX_PHRASE_LENGTH),
            use_maxsum=True,
            nr_candidates=20,   # TODO: might want to make this derived from the text length in some way
            stop_words="english",
            top_n=self.top_k
        )
        # filtering domain keywords since they could be present from the supplementary text (keep singleton domain name though)
        refined = self._refine_keywords(raw)
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
            if self.is_connected and self.supplement_fetcher:
                summaries = self.supplement_fetcher([e["url"] for e in chunk])
                #print("SUMMARY KEYS: ", summaries.keys() if summaries else "No summaries fetched")
            full_results.extend(self.generate_batch(chunk, summaries))
        return full_results




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
    LOG_FINAL_TEXT = True #! should be temporary but may include it later if it's helpful and doesn't eat up all disk space
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        nr_topics: Union[int, str, None] = None,
        preprocessor: Optional[TextPreprocessingHandler] = None,
        sort_confidence_threshold = 0.01,  # threshold for sorting topic confidence scores #! setting threshold low while experimenting - update later
        fetcher_fn: Optional[SupplementFetcher] = None
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


    def _preprocess_text(self, domain: str, full_text: str) -> str:
        """ Apply domain-level boilerplate filtering (if domain is locked) and then truncate to max_tokens """
        if self.preprocessor:
            # If domain is locked or partially locked, do the filtering - filter_boilerplate() will be a no-op if domain isn't locked yet
            full_text = self.preprocessor.process_text(full_text, domain, use_domain_filter=True)
        return full_text

    # will only be relevant using the tabular dataframe with output keywords for clustering later
    def _simulate_data_as_text(self, entry: Dict[str, Any]) -> str:
        """ embed domain and group info into the text """
        # TODO: move upstream since it's more key-dependent than I'd prefer
        domain_label = entry["domain"].split(".")[-2]  # "wikipedia" or "linuxfoundation"
        group_label = f"group_{entry['group_ids'][0]}" if entry["group_ids"] else ""
        artificial_text = f"domain_{domain_label} - {group_label}"
        return artificial_text.strip()  # return the simulated text for the entry, if no real text is available


    def _prepare_doc_for_entry(self, entry: Dict[str, Any], raw_text: str) -> str:
        """ combine the entry's title with the domain-filtered text then apply token truncation """
        domain = entry.get("domain", "")
        title_str = get_base_title_text(entry)
        # spoof common tokens so that the embedding model considers the entries' internal variables like group ID and domains
        # title_str += self._simulate_data_as_text(entry)
        # merge the cleaned/filtered text
        combined = (title_str + " " + raw_text).strip()
        final_text = self._preprocess_text(domain, combined)
        return final_text.strip()  # ensure no leading/trailing whitespace is left


    def _add_topics_to_entries(self, entries: List[Dict[str, Any]], all_topics: Dict[str, Tuple[str, float]]) -> None:
        for entry in entries:
            tid = entry["topic_id"]
            if tid == -1:
                # outlier doc
                entry["topic_keywords"] = {}
            else:
                word_tuples = all_topics.get(tid, [])
                # convert to e.g. {"word": weight} mapping
                kw_dict = {w: float(round(score, 4)) for (w, score) in word_tuples}
                entry["topic_keywords"] = kw_dict


    def fit_transform_entries(self, entries: List[Dict[str, Any]]) -> None:
        """
            1. Optionally fetch text for each entry (if we have an internet connection).
            2. Construct a doc for each entry, applying domain filtering and max token truncation.
            3. Run BERTTopic .fit_transform(...) on the entire corpus.
            4. Assign each entry's topic_id + top topic keywords in the 'topic_keywords' field.
        """
        def dump_summaries_map(file_path: str, summaries_map: Dict[str, str]) -> None:
            """ Helper function to dump the summaries map to a JSON file for debugging purposes """
            with open(file_path, 'w', encoding="utf-8") as fptr:
                json.dump(summaries_map, fptr, indent=4, ensure_ascii=False)
            print(colored(f"Summaries map dumped to {file_path}", color="green"))
        # 1) Possibly fetch all text in one batch
        summaries_map = {}
        #print(colored("Fetching webpage content of all urls...", color="green"))
        if self.is_connected and self.fetcher_fn:
            urls = [e["url"] for e in entries]
            # fetch text for each URL (e.g. domain-filtered or raw HTML)
            summaries_map = self.fetcher_fn(urls)
        #! [DEBUG] Dump fetched summaries map to a file for debugging - REMOVE LATER
        dump_summaries_map("output/sample_text_log9.json", summaries_map)
        # 2) Build corpus
        for e in tqdm(entries, desc="Cleaning Entries for Corpus"):
            # Get the text from summaries_map if present, else empty
            raw_text = summaries_map.get(e["url"], "")
            doc_text = self._prepare_doc_for_entry(e, raw_text)
            summaries_map[e["url"]] = doc_text  # update the map with the cleaned text
        #! [DEBUG] Dump fetched summaries map to a file for debugging - REMOVE LATER
        dump_summaries_map("output/sample_cleaned_log9.json", summaries_map)
        corpus = list(summaries_map.values())  # list of cleaned text for each entry
        # 3) Run .fit_transform
        print(colored("Fitting corpus to the topic model...", color="green"))
        # topics: List[int], probs: np.ndarray
        topics, probs = self.topic_model.fit_transform(corpus)
        # 4) Store topic_id & topic keywords
        for entry, topic_id, prob in zip(entries, topics, probs):
            entry["topic_id"] = topic_id
            topic_prob = float(round(max(prob), 4)) if isinstance(prob, (list, np.ndarray)) and len(prob) > 0 else 0.0
            if topic_prob < self.topic_prob_threshold:
                entry["topic_id"] = -1  # mark as outlier if below threshold
                topic_prob = 0.0
            # save highest topic probability for the entry, rounded to 4 decimal places
            entry["topic_prob"] = topic_prob
        all_topics = self.topic_model.get_topics()  # dict: {topic_id -> [(word, weight), ...], ...}
        self._add_topics_to_entries(entries, all_topics)

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