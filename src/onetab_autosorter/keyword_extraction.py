import re
import os
import json
from tqdm import tqdm
import numpy as np
from termcolor import colored
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterable
# NLP topic models and backends
from nltk.corpus import stopwords
from keybert import KeyBERT
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer



# TODO: revise KeyBERT to accept all "documents" in one go, rather than one-by-one, since it should give slightly better results



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


class BaseKeywordModel:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        candidate_labels: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.candidate_labels = candidate_labels


    @staticmethod
    def refine_keywords(keywords: List[Tuple[str, float]], min_score: float = 0.2) -> Dict[str, float]:
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

    # might remove this in favor of a more general-purpose JSON saving utility function, but this can still be used for now
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
        return combined
        # final_text = self._preprocess_text(domain, combined)
        # return final_text.strip()  # ensure no leading/trailing whitespace is left

    def run(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method.")





class KeyBertKeywordModel(BaseKeywordModel):
    """ wrapper class for KeyBERT - retrieves keywords from a given text (supplemented with response headers) using KeyBERT """
    MAX_PHRASE_LENGTH = 2  # max number of words in a phrase to extract (e.g. 1 for unigrams, 2 for bigrams, etc.)

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        candidate_labels: Optional[List[str]] = None,
        top_k=10,
        # preprocessor: Optional[TextPreprocessingHandler] = None,
        # fetcher_fn : Optional[Callable] = None
    ):
        """
            :param model_name: e.g. "all-MiniLM-L6-v2" (for the underlying sentence-transformer in KeyBERT)
            :param prefetch_factor: number of entries to prefetch in parallel (default: 1)
            :param preprocessor: optional TextPreprocessingHandler for domain filtering and text cleaning
            :param fetcher_fn: function to fetch summary text for each URL in batch
        """
        self.model = KeyBERT(model=model_name)
        self.candidate_labels = candidate_labels if candidate_labels else None
        self.stopwords = stopwords.words("english")
        self.top_k = top_k
        print(f"[KEYWORD EXTRACTOR] Using KeyBERT model: {model_name}, candidate labels: {candidate_labels}, top_k: {top_k}")

    # TODO: make a type alias for Tuple[str, float] for keyword tuples later
    def extract_keywords_from_text(self, text: Union[str, List[str]]) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        return self.model.extract_keywords(
            text,
            candidates=self.candidate_labels,
            keyphrase_ngram_range=(1, self.MAX_PHRASE_LENGTH),
            use_mmr=True,
            #diversity=0.75,
            nr_candidates=20,
            highlight=True,
            stop_words=self.stopwords,
            top_n=self.top_k
        )

    def generate(self, entry: Dict[str, Any], supplement_text: Optional[str] = None) -> Dict[str, Any]:
        text = self._prep_for_extraction(entry, supplement_text)
        #print(f"[KEYWORD EXTRACTOR] {entry['url']} text: {text}")
        if not text:
            entry["keywords"] = {}
            return entry
        raw = self.extract_keywords_from_text(text)
        # filtering domain keywords since they could be present from the supplementary text (keep singleton domain name though)
        refined = self.refine_keywords(raw)
        entry["keywords"] = refined
        return entry

    #? NOTE: results seem a LOT better using all the text at once rather than one-by-one
    def generate_batch(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        corpus = [self._prep_for_extraction(e, e.get("clean_text", "")) for e in entries]
        if not corpus:
            raise RuntimeError("[KEYWORD EXTRACTION] Something went wrong - corpus of prepared text is empty")
        keywords = self.extract_keywords_from_text(corpus) # order of output SHOULD be preserved
        for idx in range(len(entries)):
            entries[idx]["keywords"] = self.refine_keywords(keywords[idx])
        return entries


    def _generate_stream(self, entries: List[Dict[str, Any]], summaries: Optional[Dict[str, str]] = None):
        for entry in entries:
            summary = summaries.get(entry["url"], "") if summaries else None
            yield self.generate(entry, supplement_text = summary)

    def generate_stream(self, entries: List[Dict[str, Any]]) -> deque:
        """ Process entries using a generator, no prefetching """
        return deque(self._generate_stream(tqdm(entries, desc="Processing entries", unit="entry")))

    def run(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Unified runner that fetches, extracts keywords, and prepares data for downstream embedding or clustering """
        # for idx, entry in tqdm(enumerate(entries), desc="Running KeyBERT", unit="entry"):
        #     entries[idx]["keywords"] = self.generate(entry, entry.get("clean_text", None))
        # return entries
        return self.generate_batch(entries)






class BERTopicKeywordModel(BaseKeywordModel):
    """ BERTopic-based wrapper class that:
        1. Optionally fetches text for each entry (with a fetcher_fn)
        2. Applies domain-based boilerplate filtering (DomainBoilerplateFilter)
        3. Concatenates the result with the entry's title
        4. Truncates text to a max token limit
        5. Runs BERTopic .fit_transform(...) over the entire corpus
        6. Assigns topic labels and representative topic keywords to each entry
        #!!! Unlike KeyBERT, BERTopic is not doc-by-doc. It needs the entire corpus in one pass. !!!
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        candidate_labels: Optional[List[str]] = None,
        nr_topics: Union[int, str, None] = None,
        # MIGHT NOT BE ABLE TO USE THIS SINCE THE CONFIDENCE IS OFTEN ABOUT THIS LOW
        sort_confidence_threshold = 0.01,  # threshold for sorting topic confidence scores #! setting threshold low while experimenting - update later
    ):
        print(colored(f" [KEYWORD EXTRACTOR] WARNING: BERTopic model currently gives much worse results than KeyBERT!", "yellow"))
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
        candidate_labels = candidate_labels if candidate_labels else None # convert empty list to None
        self.topic_model = BERTopic(
            nr_topics = nr_topics, # can be an integer for number of topics or "auto" for automatic topic reduction
            # seed topic list would be better to use, but it requires separation by topics
            zeroshot_topic_list=candidate_labels,  # list of candidate labels for zero-shot topic modeling
            embedding_model = SentenceTransformer(model_name),
            calculate_probabilities = True,
            n_gram_range = (1, 3),  # unigrams, bigrams, and trigrams
            verbose = True
        )
        # set the confidence threshold for topic probabilities
        self.topic_prob_threshold = sort_confidence_threshold


    def _add_topics_to_entries(self, entries: List[Dict[str, Any]], all_topics: Dict[str, Tuple[str, float]], topics: List[int], probs: np.ndarray) -> None:
        print("shape of probabilities: ", probs.shape)
        for entry, topic_id, prob in zip(entries, topics, probs):
            # Calculate the highest topic probability
            print("probability (in loop): ", prob)
            #topic_prob = round(float(prob), 4)
            topic_prob = round(float(prob.max()), 4) if topic_id != -1 else 0.0
            # Mark as outlier if below the threshold
            if topic_id == -1 or topic_prob < self.topic_prob_threshold:
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
            3. Run BERTopic .fit_transform(...) on the entire corpus.
            4. Assign each entry's topic_id + top topic keywords in the 'topic_keywords' field.
        """
        corpus = []
        for e in tqdm(entries, desc="Cleaning Entries for Corpus"):
            # Get the text from summaries_map if present, else empty
            #raw_text = e.get("scraped_text", "")
            raw_text = e.get("clean_text", "")
            doc_text = self._prep_for_extraction(e, raw_text)
            ###### summaries_map[e["url"]] = doc_text  # update the map with the cleaned text
            corpus.append(doc_text)
        #self.dump_json(summaries_map, "output/sample_cleaned_log11.json", label="Cleaned summaries")
        ###### corpus = list(summaries_map.values())  # list of cleaned text for each entry
        # Run .fit_transform
        print(colored("Fitting corpus to the topic model...", color="green"))
        # topics: List[int], probs: np.ndarray
        topics, probs = self.topic_model.fit_transform(corpus)
        all_topics = self.topic_model.get_topics()  # dict: {topic_id -> [(word, weight), ...], ...}
        # Store topic_id & topic keywords
        self._add_topics_to_entries(entries, all_topics, topics, probs)

    ##############################################################################################################################
    # to mirror the KeyBert approach, define a generate method. In practice, BERTopic is best used in one shot on all docs.
    ##############################################################################################################################

    def run(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ convenience method that calls fit_transform_entries(...) and returns the updated entries """
        self.fit_transform_entries(entries)
        return entries

    def generate(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """ Satisfies the 'BaseKeywordModel' interface. But BERTopic doesn't do doc-by-doc extraction, so do nothing here. """
        return entry