import re
from tqdm import tqdm
#from copy import deepcopy
from xtermcolor import colorize
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
# NLP topic models
from keybert import KeyBERT
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
# from numpy import percentile
from onetab_autosorter.utils import is_internet_connected
from onetab_autosorter.scraper.scraper_utils import default_html_fetcher, default_html_fetcher_batch, SupplementFetcher
from onetab_autosorter.text_cleaning import get_base_title_text


#~ if I extract all the filtering and fetching code, I should group entries by their domain
    #~ I could avoid referencing blacklists by just comparing fetched supplemental text against others in the domain,
    #~ since everything that's duplicated is likely to be boilerplate

# TODO: might be better to extract the filtering functions to new utility functions to comply with single responsibility principle
    # if they were joined to a single class, they'd be the ones containing the IGNORE_KEYWORDS and IGNORED_DOMAINS lists
    # since the filtering functions currently come one after another, it should follow a pipeline-like structure (like torch.nn.Sequential)
    # consider some differentiation of preprocessing (e.g. `_get_base_title_text`) and postprocessing (e.g. `_refine_keywords`) utilities



class BaseKeywordModel:
    def generate(self, entry: Dict) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")


class KeyBertKeywordModel(BaseKeywordModel):
    """ wrapper class for KeyBERT - retrieves keywords from a given text (supplemented with response headers) using KeyBERT """
    MAX_PHRASE_LENGTH = 3
    #MIN_SENTENCE_LENGTH = 6  # minimum number of words in a sentence to be considered for keyword extraction
    # IGNORED_KEYWORDS = {"privacy simplified", "non javascript", "url", "link", "website", "webpage", "page", "site", "bookmark", "tab"}
    # IGNORED_DOMAINS = {"google", "bing", "duckduckgo", "yahoo", "baidu", "yandex", "ask", "aol", "msn"}

    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=10, prefetch_factor: int = 1, fetcher_fn : Optional[SupplementFetcher] = None):
        self.model = KeyBERT(model=model_name)
        self.top_k = top_k
        # TODO: might just remove this argument later since an equivalent argument is passed to `generate_with_chunking`
        self.prefetch_factor = prefetch_factor  # not used in this class, but can be useful for async fetching
        # if fetcher_fn is None, set it to default fetcher based on prefetch_factor
        self.supplement_fetcher = fetcher_fn or (default_html_fetcher if self.prefetch_factor == 1 else default_html_fetcher_batch)
        self.is_connected = is_internet_connected()
        if not self.is_connected:
            print(f"{colorize('WARNING: No internet connection detected. Supplemental HTML fetching will be disabled.', rgb='yellow')}")


    def _refine_keywords(self, keywords: List[Tuple[str, float]], min_score=0.2) -> Dict[str, float]:
        """ refines keywords by removing duplicates and low-scoring keywords """
        seen_tokens = set()
        refined = {}
        for kw, score in sorted(keywords, key=lambda x: -x[1]):
            if score < min_score:
                continue
            normalized = re.sub(r'\b(\w+)( \1\b)+', r'\1', kw.lower())  # remove internal repetition
            # tokens = frozenset([*normalized.split(), normalized]) # split by spaces and create a frozenset of tokens
            tokens = frozenset(normalized.split()) # split by spaces and create a frozenset of tokens
            # strip any keywords found within self.IGNORED_KEYWORDS
            # if any(ignored in tokens for ignored in self.IGNORED_KEYWORDS):
            #     continue
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
        MAX_TOKENS = 2000  # limit the text length to avoid excessive processing time and memory usage
        text = get_base_title_text(entry)
        text = (text + " " + (supplement_text or "")).strip()
        # base this on recurring text strings in all supplementary texts from the same domain
        #text = self._append_supplementary_text(text, entry["url"], supplement_text)
        #return text
        tokens = text.split()
        if len(tokens) > MAX_TOKENS:
            text = " ".join(tokens[:MAX_TOKENS])
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
        ###raw_filtered = self._filter_domain_keywords(raw, entry.get("domain", ""))
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
    """ BERTTopic-based wrapper that can act as a drop-in replacement for the KeyBERT approach in terms of pipeline usage
        Note that BERTTopic requires a corpus-level approach:
            1. gather text for all entries
            2. run `fit_transform` once to get topic assignments
            3. store each entry's topic info back in the entries
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        nr_topics: Optional[int] = None,
        prefetch_factor: int = 1,
        fetcher_fn: Optional[SupplementFetcher] = None,
    ):
        """
        :param model_name: e.g. "all-MiniLM-L6-v2" (for the underlying sentence-transformer in BERTopic)
        :param nr_topics: If you want to reduce the number of topics, you can set nr_topics to something like "auto" or an integer.
        :param prefetch_factor: parallels KeyBertKeywordModel usage for chunking/fetching
        :param fetcher_fn: function to fetch summary text for each URL in batch
        """
        # Create a Sentence-Transformer embedding model for BERTopic
        embedding_model = SentenceTransformer(model_name)
        self.topic_model = BERTopic(nr_topics=nr_topics, embedding_model=embedding_model)
        self.prefetch_factor = prefetch_factor
        self.supplement_fetcher = fetcher_fn or (
            default_html_fetcher if prefetch_factor == 1 else default_html_fetcher_batch
        )
        self.is_connected = is_internet_connected()
        if not self.is_connected:
            print(f"{colorize('WARNING: No internet connection detected. Supplemental HTML fetching will be disabled.', rgb='yellow')}")

    def _prepare_document(self, entry: Dict[str, Any], supplement_text: Optional[str] = None) -> str:
        """ Combine the (possibly domain-filtered) supplement_text with the page title or other metadata to form the input for BERTTopic """
        base_title = get_base_title_text(entry)
        doc = (base_title + " " + (supplement_text or "")).strip()
        return doc if doc else " "  # fallback to space if empty


    def generate_corpus(self, entries: List[Dict[str, Any]], summaries: Optional[Dict[str, str]] = None) -> List[str]:
        """ Construct the corpus from all entries. If 'summaries' is given (url->text), include that text in the doc. """
        docs = []
        for e in entries:
            url = e.get("url", "")
            if summaries:
                text = summaries.get(url, "")
            else:
                text = ""
            doc = self._prepare_document(e, text)
            docs.append(doc)
        return docs

    def fit_transform(self, entries: List[Dict[str, Any]], docs: List[str]) -> None:
        """ Run BERTTopic .fit_transform() on the entire corpus (docs). Then attach the discovered topic info back to each entry """
        topics, _ = self.topic_model.fit_transform(docs)
        # for each doc, store the assigned topic
        for entry, topic_id in zip(entries, topics):
            entry["topic_id"] = topic_id
        # get top n keywords from each discovered topic
        all_topics = self.topic_model.get_topics() # list[(word, weight), ...]
        # store them as "topic_keywords" per doc
        for entry in entries:
            tid = entry["topic_id"]
            if tid == -1:
                # -1 means "outlier" or "no topic assigned" in BERTopic
                entry["topic_keywords"] = {}
            else:
                # get the top words for that topic
                word_tuples = all_topics.get(tid, [])
                # convert e.g. [("deep", 0.45), ("learning", 0.42), ...] to a dict with { "word": weight }
                kw_dict = {w: float(round(score, 4)) for (w, score) in word_tuples}
                entry["topic_keywords"] = kw_dict

    ##########################################################################
    # To mirror the KeyBert approach, define some generate_* methods.
    # But in practice, BERTTopic is best used in one shot on all docs.
    ##########################################################################

    def generate_stream(self, entries: List[Dict[str, Any]]) -> deque:
        """ actually a "batch" method despite the name. Fetch text for all entries, run BERTTopic on them once, and yield results """
        # If you want to fetch supplementary text in one go
        summaries = {}
        if self.is_connected and self.supplement_fetcher:
            urls = [e["url"] for e in entries]
            summaries = self.supplement_fetcher(urls)
        docs = self.generate_corpus(entries, summaries)
        self.fit_transform(entries, docs)
        return deque(entries)

    def generate_with_chunking(self, entries: List[Dict[str, Any]], chunk_size: int = 20) -> deque:
        """ BERTTopic doesn't support chunk-by-chunk topic discovery since it needs the whole corpus, so this is equivalent to generate_stream. """
        return self.generate_stream(entries)

    def generate_batch(self, entries: List[Dict[str, Any]], summaries: Dict[str, str]) -> deque:
        """ if summaries are provided for each entry, skip re-fetching.
            BERTTopic still needs to run .fit_transform(...) on the entire doc set.
        """
        docs = self.generate_corpus(entries, summaries)
        self.fit_transform(entries, docs)
        return deque(entries)

    def generate(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """ Not meaningful for BERTTopic, since we do batch topic modeling. Do nothing.
        """
        # For a "single doc" approach, you'd need to have a previously-trained BERTTopic model,
        # then do .transform([doc]) to get the topic. But that requires the model be fitted 
        # to a larger corpus already.
        return entry