import os
import json
from re import Pattern
from collections import defaultdict
from typing import Optional, Dict, List, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.config.config import Config, CheckpointSettings #, get_cfg_from_cli
from onetab_autosorter.utils.utils import detect_bookmark_format, deduplicate_entries
from onetab_autosorter.utils.io_utils import cache_stage, compute_hash, get_hashed_path_from_string, load_json, save_json
# TODO: replace with custom type hints later - not needed as an object here
from onetab_autosorter.parsers import BaseParser
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel, BERTopicKeywordModel
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter


# TODO: add keyword model classes to a registry somewhere else later
KEYWORD_EXTRACTOR_REGISTRY = {
    "keybert": KeyBertKeywordModel,
    "bertopic": BERTopicKeywordModel
}


"""
# ROADMAP

    ## OVERVIEW:
    organize functions here for use in each pipeline stage, organized into four categories with prefixed naming convention:
        1. create_* - functions that create objects or classes (e.g. parsers, keyword models, etc.)
        2. run_* - functions that run a specific task (e.g. parsing, webscraping, etc.)
        3. load_* - functions that load data from a file or cache (e.g. JSON, CSV, etc.)
        4. save_* - functions that save data to a file or cache (e.g. JSON, CSV, etc.)

    ###? checking off items below once functions are implemented (first check) and the stage is complete (second check)
    ## STAGES TO IMPLEMENT:
        1. [][] Initial Bookmark HTML Parsing
            a. [x][] create a parser object based on the input file type (HTML, JSON, etc.)
            b. [x][] run it, and return entries (expected format: List[Dict[str, Any]])
            c. [][] (optionally) load entries from a file or cache (as expected format) and skip this stage
            d. [][] (optionally) save entries parsed from HTML to JSON - mutually exclusive with (c)
        2. [][] Webscraping Stage
            a. [x][] create a fetcher function based on the scraper type (Java, Async, etc.)
            b. [x][] run it to get supplementary text for each entry and return a dictionary (returned as Dict[str, str] w/ url keys)
            c. [][] (optionally) load scraped data (as expected format in part b) from a file or cache and skip this stage
                OR (optionally) load cleaned scraped data if available and skip stages 2, 3, and 4
            d. [][] (optionally) save scraped data (cleaned or not) to a file or cache - mutually exclusive with (c)
        3. [][] Domain Boilerplate Filter Creation
            a. [x][] create and immediately fit a domain filter object on the webscraped data
            b. [][] (optionally) load domain filter phrases previously fit to each domain from a file or cache and skip this stage
            c. [][] (optionally) save (append to existing) domain filter phrases to a file or cache - mutually exclusive with (b)
        4. [][] Filter Creation
            a. [x][] create a TextCleaningFilter object based on the configuration settings
            b. [x][] create a TextPreprocessingHandler instance with the cleaning filter and domain filter
            c. [x][] run the filter to clean all scraped text for usable input to the keyword models
            d. [][] (optionally) save the cleaned text to a file or cache (as Dict[str, str] w/ url keys)
        5. [][] Keyword Extraction
            a. [x][] create a keyword model object based on the configuration settings
            b. [][] run it on the cleaned text and filter resulting keywords and scores (for BERTopic, also save topic ID results)
            c. [][] (optionally) load all entries (List[Dict] format) along with keywords from a file or cache and skip this stage
            d. [][] (optionally) save the keywords to a file or cache (as part of original entry data format List[Dict[str, Any]]) - mutually exclusive with (c)
        6. [][] Embedding Stage
            a. [][] create an embedding model object based on the configuration settings and initialized with the keywords, the cleaned text, or all entry data
            b. [][] (optionally) load all entries (format TBD, depending on `polars`) with embeddings from a file or cache and skip this stage
            c. [][] (optionally) save the embeddings to a file or cache (format TBD, depending on `polars`) - mutually exclusive with (b)
        7. [][] Clustering Stage
            a. [][] create a clustering model object based on the configuration settings
            b. [][] run it with all available embedded data, preferably as a multimodal input (embedding of all fields in tabular dataframe)
                NOTE: probably not allowing loading at this stage since it makes no sense
        8. [][] Cluster Label Generation Stage
            a. [][] create a label generation model object based on the configuration settings
            b. [][] run it with simple keyword input from all data in a cluster to generate a label describing the cluster's nodes
                NOTE: probably not allowing loading at this stage since it makes no sense
        9. [][] Final Output Stage
            a. [][] create final labeled bins with the generated labels and (URL, title) in bins
            b. [][] save as a simple JSON file, etc OR generate a new bookmark import-ready HTML file with the new bins

    ## IMPLEMENTATION DETAILS:
        - saving data after each stage is optional and will be configured as a callback in the pipeline stage itself
        - loading data from a file or cache will be handled by a `loading_factory` static method in all pipeline stages to load, normalize, and return the data
        - the pipeline factory will be responsible for instantiating pipeline stages using configuration options
            - need to write so that config deepcopies are passed to stage constructors and functions in this file aren't always receiving the whole config object
        - at times, coupling certain stages (e.g. webscraping and domain filtering) may be unavoidable, so might just write a custom pipeline stage for that
        - embedding and clustering details aren't fully fleshed out yet
        - text cleaning still needs some work to ensure we're not passing garbage to the keyword extractor models

"""





##########################################################################################################################
# (BELOW) NEWER METHODS that are UNTESTED and are not yet fully integrated into the pipeline
##########################################################################################################################

def merge_entries_with_scraped(entries: List[Dict[str, Any]], scraped_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #scraped_dict: Dict[str, Dict[str, Any]] = {item['url']: item for item in scraped_data}
    scraped_urls = list(scraped_data.keys())
    for entry in entries:
        url: str = entry['url']
        if url in scraped_urls:
            #! Think this should be raw_text - track down usage everywhere (definitely a problem in `run_text_filtering`)
            entry['scraped'] = scraped_data[url] #.get('raw_html', "")
    return entries


# def load_entries(file_path: str, config: Config) -> List[Dict[str, Any]]:
#     """ load entries from a file, either from cache or by parsing the original HTML file """
#     from onetab_autosorter.utils.io_utils import compute_hash, cache_path
#     cache_settings = config.checkpoints
#     parsed_hash = compute_hash(file_path)
#     cache_file = cache_path(cache_settings.cache_dir, f"parsed_{parsed_hash}")
#     if cache_settings.reuse_parsed and os.path.exists(cache_file):
#         with open(cache_file, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     return create_and_run_parser(file_path, cache_file, cache_settings, config.deduplicate)


##########################################################################################################################
# (BELOW) NEW METHODS (with `create_*`, `run_*`, `load_*`, and `save_*` atomized functions)
    #   not yet fully integrated into the pipeline - for now working on the lower level logic
##########################################################################################################################


# Loaders (High-level wrapper functions)
#& UNTESTED + UNFINISHED + UNREVIEWED
def load_cached_data(stage_name: str, config: Config, fallback_loader: Callable):
    cache_file = get_hashed_path_from_string(config.input_file, stage_name, config.checkpoints.cache_dir)
    if getattr(config.checkpoints, f"reuse_{stage_name}") and os.path.exists(cache_file):
        return load_json(cache_file)
    else:
        data = fallback_loader()
        if getattr(config.checkpoints, f"save_{stage_name}"):
            save_json(cache_file, data)
        return data


#& UNTESTED + UNFINISHED
def run_text_filtering(preprocessor: TextPreprocessingHandler, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for entry in entries:
        #! FIXME: there is no "raw_html" key set in the entry dict anywhere at the moment
        raw_html = entry.get("raw_html", "")
        domain = entry["domain"]
        #! FIXME: this key also doesn't exist - we should expect to call process_text() instead of process_html()
        entry["cleaned_text"] = preprocessor.process_html(raw_html, domain)
    return entries


##########################################################################################################################


def create_parser(file_path: str) -> BaseParser:
    from onetab_autosorter.parsers import OneTabParser, JSONParser, NetscapeBookmarkParser
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".html":
        html_type = detect_bookmark_format(file_path)
        if html_type == "onetab":
            return OneTabParser()
        elif html_type == "netscape":
            return NetscapeBookmarkParser()
    elif ext == ".json":
        return JSONParser()
    raise ValueError(f"Unsupported input format: {ext}")


def run_parser(parser: BaseParser, file_path: str, deduplicate: bool = True) -> List[Dict[str, Any]]:
    entries = parser.parse(file_path)
    if deduplicate:
        entries = deduplicate_entries(entries) # using the default max url length of 200 to skip testing long urls
    if not entries:
        if deduplicate:
            raise RuntimeError("No entries found in parsed entries after deduplication.")
        raise RuntimeError("No entries found in the input file.")
    return entries


def create_and_run_parser(
    file_path: str,
    cache_file: str,
    cache_settings: CheckpointSettings,
    deduplicate: bool = True
) -> List[Dict[str, Any]]:
    """ get parser object depending on file input type and run the parser to get the initial entries """
    # TODO: separate creation of the parser from the parsing of the file
    parser = create_parser(file_path)
    entries = run_parser(parser, file_path, deduplicate)
    # separating saving logic from the parsing logic
    if cache_settings.save_parsed:
        os.makedirs(cache_settings.cache_dir, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f)
    return entries


def create_fetcher(scraper_type: Union[Callable, Literal["none", "java", "naive", "limited", "async"]] = "limited", **kwargs) -> Callable:
    """ returns a "fetcher" function for webscraping supplementary text based on `scraper_type`
        Args:
            scraper_type (str): one of ["none", "webscraper", "default", "java", "async"]
            kwargs: additional parameters passed to WebScraper constructor
        Returns:
            # TODO: replace with a typing.Protocol definition for a callable that accepts List[str] -> Dict[str, str]
            Callable: a callable that accepts List[str] -> Dict[str, str]
    """
    scraper_type = scraper_type.lower() if isinstance(scraper_type, str) else scraper_type
    if callable(scraper_type):
        #? NOTE: may be better to pass a wrapper function that handles the kwargs rather than passing them to the callable directly
        print("WARNING: (in `create_fetcher`) `scraper_type` passed as a callable is untested and may lead to unexpected behavior.")
        return scraper_type
    if scraper_type == "none":
        return None
    elif scraper_type in ["limited", "async"]:
        from onetab_autosorter.scraper.webscraper import WebScraper
        scraper = WebScraper(**kwargs) #rate_limit_delay=1.2, max_workers=8)
        scrape_func = "run_async_fetch_batch" if scraper_type == "async" else "fetch_batch"
        return getattr(scraper, scrape_func)
    elif scraper_type == "java":
        raise NotImplementedError("Java microservice scraper needs to be refactored for use in the pipeline.")
        from onetab_autosorter.scraper.client import ScraperServiceManager, fetch_summary_batch
        scraper = ScraperServiceManager()
        return scraper.fetch_within_context(fetch_summary_batch)
    elif scraper_type == "naive":
        from onetab_autosorter.scraper.scraper_utils import default_html_fetcher_batch
        return default_html_fetcher_batch
    else:
        raise ValueError(f"Invalid scraper type: {scraper_type}; expected one of ['java', 'naive', 'limited', 'async']")


# TODO: replace json_path default argument with a config object that has the path as an attribute
def create_domain_filter(load_from_file: bool, json_path = r"output/domain_boilerplate.json", **kwargs):
    from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
    if load_from_file:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return DomainBoilerplateFilter.load_boilerplate_map(json_path, **kwargs)
    return DomainBoilerplateFilter(**kwargs)


def get_domain_mapped_urls(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        # populate dictionary of lists with domain names as keys and entries as values
        domain_map[e["domain"]].append(e)
    return domain_map

def create_and_fit_domain_filter(
    load_from_file: bool,
    domain_map: Dict[str, List[Dict]], # dictionary of domain -> list of entries from domain
    json_path = None,
    scraper_fn: Callable = None,
    **kwargs
):
    MIN_REPEAT_COUNT = 3
    # instantiate domain filter object and run the filter #? (we fit the filter first because the keyword extractor expects locked domain keys)
    domain_filter_obj = create_domain_filter(load_from_file, json_path, **kwargs)
    domain_filter_obj.run_preliminary_search(domain_map, MIN_REPEAT_COUNT, scraper_fn) #default_html_fetcher_batch)
    # TODO: add support for appending to the same file and slowly growing the filter phrases over time
    # TODO: remove the saving step from this function and save afterwards in the calling location
    if json_path:
        domain_filter_obj.save_boilerplate_map(json_path)
    return domain_filter_obj


###########################################################################################################
# feels somewhat pointless the way these two functions are written, but the domain filter fitting step is best left separate
#& UNTESTED
def create_text_cleaning_filter(**kwargs) -> TextCleaningFilter:
    # min_word_count: int, compiled_filters: List[Union[str, Pattern]]
    return TextCleaningFilter(**kwargs)

#& UNTESTED
def create_preprocessor(domain_filter, cleaning_filter, max_tokens: int = 200) -> TextPreprocessingHandler:
    return TextPreprocessingHandler(domain_filter, cleaning_filter, max_tokens)
############################################################################################################

#& UNTESTED
# TODO: replace with custom type hint for the return type later
def create_keyword_model(kw_model_name: str, backbone_model: str, seed_kws: List[str], top_k: int, **kwargs) -> Union[KeyBertKeywordModel, BERTopicKeywordModel]:
    kw_model_cls = KEYWORD_EXTRACTOR_REGISTRY.get(kw_model_name, None)
    if kw_model_cls is None:
        raise ValueError(f"Invalid keyword model: {kw_model_name}. Expected one of {list(KEYWORD_EXTRACTOR_REGISTRY.keys())}")
    return kw_model_cls(
        # TODO: just rename these model constructor arguments to ensure consistency
        backbone_model, # model name for both models
        seed_kws,       # candidate_labels for both models
        top_k,          # top_k for KeyBert and nr_topics for BERTopic
        **kwargs
    )



#& UNFINISHED (should support loading from cache as well)
def load_entries(input_file: str) -> List[Dict[str, Any]]:
    # TODO: extend the logic and make this function a dispatcher that calls, among others, load_json
    # load the entries from the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as fptr:
        entries = json.load(fptr)
    return entries







############################################################################################################
# new run functions - still debugging
############################################################################################################

def run_webscraping(fetcher_fn: Callable, urls: List[str]) -> Dict[str, str]:
    return fetcher_fn(urls)

# @cache_stage("scraped")
# def run_webscraping_stage(config, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     fetcher_fn = create_fetcher(config.scraper_type)
#     urls = [entry["url"] for entry in entries]
#     scraped_text = fetcher_fn(urls)
#     for entry in entries:
#         entry["raw_html"] = scraped_text.get(entry["url"], "")
#     return entries

# @cache_stage("cleaned")
# def run_text_filtering_stage(config, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     domain_filter = create_domain_filter(True, "output/domain_boilerplate.json")
#     cleaning_filter = TextCleaningFilter(ignore_patterns=config.compiled_filters)
#     preprocessor = TextPreprocessingHandler(domain_filter, cleaning_filter)
#     for entry in entries:
#         raw_html = entry.get("raw_html", "")
#         entry["cleaned_text"] = preprocessor.process_html(raw_html, entry["domain"])
#     return entries

# @cache_stage("keywords")
# def run_keyword_extraction_stage(config, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     keyword_model = create_keyword_model("keybert", config.model_name, config.seed_kws, config.keyword_top_k)
#     for entry in entries:
#         text = entry.get("cleaned_text", "")
#         entry["keywords"] = keyword_model.generate(entry, text)["keywords"]
#     return entries

def run_text_filtering(preprocessor: TextPreprocessingHandler, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for entry in entries:
        #!! FIXME: handle the difference in the HTML vs text processing methods from here
            #? important to note that for both process_html and process_text, they expect single strings
        raw_html = entry.get("raw_html", "")
        entry["cleaned_text"] = preprocessor.process_html(raw_html, entry["domain"])
    return entries