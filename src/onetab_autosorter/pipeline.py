import os, sys
import json
from collections import defaultdict, Counter
from typing import Optional, Dict, List
from tqdm import tqdm
# local imports
from onetab_autosorter.scraper.scraper_utils import SupplementFetcher, default_html_fetcher_batch
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel
from onetab_autosorter.parsers import OneTabParser, JSONParser
from onetab_autosorter.config import Config, get_cfg_from_cli
from onetab_autosorter.utils.utils import deduplicate_entries, PythonSetEncoder
##from onetab_autosorter.text_cleaning import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter


def get_parser(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".html":
        return OneTabParser()
    elif ext == ".json":
        return JSONParser()
    else:
        raise ValueError(f"Unsupported input format: {ext}")


def load_entries(file_path: str, deduplicate: bool = True, max_url_len: int = 200) -> List[Dict]:
    parser = get_parser(file_path)
    entries = parser.parse(file_path)
    if deduplicate:
        entries = deduplicate_entries(entries, max_length=max_url_len)
    if not entries:
        raise RuntimeError("No entries found in the input file.")
    return entries



# TODO: split these two functions up to use a common interface for running the pipeline

def run_pipeline(config: Config, fetcher_fn: Optional[SupplementFetcher] = None):
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    kwargs = dict(
        model_name=config.model_name,
        prefetch_factor = config.chunk_size,  # number of entries to process in each chunk
        top_k=config.keyword_top_k,
    )
    if fetcher_fn is not None:
        kwargs['fetcher_fn'] = fetcher_fn
    keyword_model = KeyBertKeywordModel(**kwargs)
    if config.chunk_size > 1:
        entries = keyword_model.generate_with_chunking(entries, config.chunk_size)
    else:
        entries = keyword_model.generate_stream(entries)
    #? NOTE: entries is now returned as a deque of dictionaries, so we may need to convert it back to a list
    entries = list(entries)  # convert deque back to list for JSON serialization
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, cls=PythonSetEncoder)

#? NOTE: in the long term, I may want to overhaul a ton of this to be some browser extension
#? or a web app, but for now, I just want to sort my OneTab bookmarks specifically


def run_pipeline_with_scraper(config: Config):
    """ Run the pipeline with the Java-based scraper service for HTML webcrawling. """
    from onetab_autosorter.scraper.client import fetch_summary_batch #, fetch_summary
    from onetab_autosorter.scraper.launcher import ScraperServiceManager
    scraper = ScraperServiceManager()
    print("config output file: ", config.output_json)
    try:
        scraper.start()
        run_pipeline(config, fetch_summary_batch)
    finally:
        scraper.stop()




#################################################################################################
# (NEW) Keyword Extraction + Domain Boilerplate Filtering Pipeline
#################################################################################################

#def get_boilerplate_filter(filter_json_path: str = "", from_file: bool = False, sample_thresh=5, min_count=2) -> DomainBoilerplateFilter:
def get_boilerplate_filter(filter_json_path: str = "", from_file: bool = False, **kwargs) -> DomainBoilerplateFilter:
    # optionally load existing domain filter results from disk
    if from_file and os.path.isfile(filter_json_path):
        return DomainBoilerplateFilter.load_boilerplate_map(
            filter_json_path,
            **kwargs  # pass any additional keyword arguments to the constructor
        )
    return DomainBoilerplateFilter(**kwargs)  # create a new instance with the given parameters

# TODO: add this as a static method within the boilerplate filter itself, with the fetcher function as an argument
def run_boilerplate_filtering(
    sorted_domains,
    domain_filter: DomainBoilerplateFilter,
    domain_map: Dict[str, List[Dict]],
):
    domain_counts = Counter(sorted_domains)
    # since some domains might already be locked from a prior run, only fetch if not locked
    for domain in tqdm(sorted_domains, desc="Domain Boilerplate Detection Step"):
        domain_data = domain_filter.get_domain_data(domain)
        if (domain_data and domain_data.locked) or domain_counts[domain] < domain_filter.min_repeat_count:
            continue  # already locked from previous runs
        # fill only as many domain entries as needed (FILTER_THRESHOLD)
        sampling_size = min(domain_counts[domain], domain_filter.min_domain_samples)
        domain_entries = domain_map[domain][:sampling_size]
        urls = [e["url"] for e in domain_entries]
        # fetch HTML information from HTTP requests in a batch
        text_map = default_html_fetcher_batch(urls)
        for e in domain_entries:
            raw_text = text_map.get(e["url"], "")
            #? NOTE: an internal trigger to add_entry_text finalizes phrases after enough are encountered
            domain_filter.add_entry_text(domain, raw_text)
    # Forcefully finalize any domain that didn't meet min_domain_samples:
    domain_filter.force_finalize_all()


def create_and_run_domain_filter(
    config: Config,
    sorted_domains,
    domain_map: Dict[str, List[Dict]],
    json_path = os.path.join("output", "domain_boilerplate.json"),
    **kwargs
):
    # instantiate domain filter object
    domain_filter_obj = get_boilerplate_filter(
        json_path,
        from_file = config.init_domain_filter,
        **kwargs
    )
    run_boilerplate_filtering(sorted_domains, domain_filter_obj, domain_map)
    #if config.init_domain_filter: # might make this a separate config argument later
    domain_filter_obj.save_boilerplate_map(json_path)
    return domain_filter_obj


# TODO: if we added a way for someone to give their bookmark folder structure,
    # it could be flattened and folder names could be used as the `seed_topic_list` for BERTTopic


def run_pipeline_with_domain_filter(config: Config):
    """
        1) Parse the input file into entries and optionally remove duplicates
        2) Possibly load a saved domain -> boilerplate mapping (if config.init_domain_filter is True)
        3) Group entries by domain; fetch text for each domain to build up the domain filter
        4) Lock in (finalize) boilerplate for each domain
        5) Re-fetch or reuse the text, filter it, run KeyBERT, and produce final JSON
        6) Save updated domain filter if needed
    """
    # TODO: combine all the domain filtering stuff into another convenience function to simplify arguments
    FILTER_THRESHOLD = 5
    MIN_DOMAIN_COUNT = 2
    filter_json_path = os.path.join("output", "domain_boilerplate.json")
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        domain_map[e["domain"]].append(e)
    sorted_domains = sorted(domain_map.keys(), key=lambda d: len(domain_map[d]), reverse=True)
    print("=== PASS 1: Accumulate and lock domain boilerplate. ===")
    # instantiate domain filter object and run the filter #? (we set the filter first because the keyword extractor expects locked domain keys)
    domain_filter_obj = create_and_run_domain_filter(config, sorted_domains, domain_map, FILTER_THRESHOLD, MIN_DOMAIN_COUNT, filter_json_path)
    print("=== PASS 2: Keyword Extraction with domain-based filtering. ===")
    # Re-fetch text; #?should probably store them in memory from the filtering, but it only fetches as many as needed
    # Prepare KeyBERT
    keyword_model = KeyBertKeywordModel(
        model_name=config.model_name,
        top_k=config.keyword_top_k,
        prefetch_factor=1  # or config.chunk_size if you want batch logic
    )
    final_results = []
    #!!! FIXME: no way in hell this runs in the future without rate-limiting - should also try to maximize time between request by the ordering
    # do domain by domain again, or add single pass with chunking later
    print("NOTE: Most frequent sites are processed in chunks first, so the estimated runtime will appear higher at the start.")
    for domain in tqdm(sorted_domains, desc="Domain (Extraction Step)"):
        domain_entries = domain_map[domain]
        urls = [e["url"] for e in domain_entries]
        text_map = default_html_fetcher_batch(urls)
        # TODO: replace with constant-length chunking again later (should really just use `entries` directly)
        for e in domain_entries:
            raw_text = text_map.get(e["url"], "")
            # filter boilerplate lines
            filtered_text = domain_filter_obj.filter_boilerplate(domain, raw_text)
            # run KeyBERT
            processed = keyword_model.generate(e, supplement_text=filtered_text)
            final_results.append(processed)
    # save final JSON with keywords added to all entries
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, cls=PythonSetEncoder)



##############################################################################################################
# TESTING BERTTopic model in new pipeline - still need to consolidate them and handle the chunking logic later
##############################################################################################################

def run_pipeline_with_bertopic(config: Config):
    # TODO: include the domain filtering back in - written by Copilot so it missed some stuff
    from onetab_autosorter.keyword_extraction import BERTTopicKeywordModel
    FILTER_THRESHOLD = 5
    MIN_DOMAIN_COUNT = 2
    filter_json_path = os.path.join("output", "domain_boilerplate.json")
    # Load entries, maybe do domain filtering, etc.
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        domain_map[e["domain"]].append(e)
    sorted_domains = sorted(domain_map.keys(), key=lambda d: len(domain_map[d]), reverse=True)
    domain_filter = create_and_run_domain_filter(
        config,
        sorted_domains,
        domain_map,
        filter_json_path,
        min_domain_samples=FILTER_THRESHOLD,
        min_repeat_count=MIN_DOMAIN_COUNT,
        #? NOTE: scales kinda badly
        ngram_range = (2, 8) # determines the n-gram range for boilerplate detection, e.g. (2,5) means phrases with 2-5 words are considered
    )
    del domain_map, sorted_domains # free memory - variables not needed anymore
    text_cleaner = TextCleaningFilter()
    preprocessor = TextPreprocessingHandler(domain_filter, text_cleaner)
    # Create BERTTopic model with domain filtering + text truncation
    topic_model = BERTTopicKeywordModel(
        model_name=config.model_name,
        nr_topics=None,       # or 'auto', or an integer
        preprocessor=preprocessor,  # use the TextPreprocessingHandler for cleaning
        fetcher_fn=None       # or default_html_fetcher_batch
    )
    # Run BERTTopic on the entire set of entries
    updated_entries = topic_model.run(entries)
    if not updated_entries:
        raise RuntimeError("ERROR: updated entries returned empty")
    # Save or process the updated entries
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(updated_entries, f, indent=2, cls=PythonSetEncoder)



if __name__ == "__main__":
    config = get_cfg_from_cli()
    # defaults to False
    if config.use_java_scraper:
        print("Using Java-based scraper for HTML webcrawling.")
        run_pipeline_with_scraper(config)
    run_pipeline(config)
