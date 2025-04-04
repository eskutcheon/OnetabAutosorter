import os
import json
from collections import defaultdict, Counter
from typing import Optional, Dict, List
from tqdm import tqdm
# local imports
from onetab_autosorter.scraper.scraper_utils import SupplementFetcher, default_html_fetcher_batch
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel
from onetab_autosorter.parsers import OneTabParser, JSONParser
from onetab_autosorter.config import Config, get_cfg_from_cli
from onetab_autosorter.utils import deduplicate_entries, PythonSetEncoder
from onetab_autosorter.text_cleaning import DomainBoilerplateFilter



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

def get_boilerplate_filter(filter_json_path, from_file: bool = False, sample_thresh=5, min_count=2) -> DomainBoilerplateFilter:
    # optionally load existing domain filter results from disk
    if from_file and os.path.isfile(filter_json_path):
        return DomainBoilerplateFilter.load_boilerplate_map(
            filter_json_path,
            min_domain_samples=sample_thresh,
            min_repeat_count=min_count
        )
    return DomainBoilerplateFilter(min_domain_samples=sample_thresh, min_repeat_count=min_count)


def run_boilerplate_filtering(
    sorted_domains,
    domain_filter: DomainBoilerplateFilter,
    domain_map: Dict[str, List[Dict]],
    filter_threshold: int = 5,
    min_domain_count: int = 2
):
    domain_counts = Counter(sorted_domains)
    # since some domains might already be locked from a prior run, only fetch if not locked
    for domain in tqdm(sorted_domains, desc="Domain Boilerplate Detection Step"):
        if domain_filter.domain_locked[domain] or domain_counts[domain] < min_domain_count:
            continue  # already locked from previous runs
        # fill only as many domain entries as needed (FILTER_THRESHOLD)
        sampling_size = min(domain_counts[domain], filter_threshold)
        domain_entries = domain_map[domain][:sampling_size]
        urls = [e["url"] for e in domain_entries]
        # fetch HTML information from HTTP requests in a batch
        text_map = default_html_fetcher_batch(urls)
        for e in domain_entries:
            raw_text = text_map.get(e["url"], "")
            domain_filter.add_entry_text(domain, raw_text)
    # Forcefully finalize any domain that didn't meet min_domain_samples:
    domain_filter.force_finalize_all()



def run_pipeline_with_domain_filter(config: Config):
    """
        1) Parse the input file into entries and optionally remove duplicates
        2) Possibly load a saved domain -> boilerplate mapping (if config.init_domain_filter is True)
        3) Group entries by domain; fetch text for each domain to build up the domain filter
        4) Lock in (finalize) boilerplate for each domain
        5) Re-fetch or reuse the text, filter it, run KeyBERT, and produce final JSON
        6) Save updated domain filter if needed
    """
    FILTER_THRESHOLD = 5
    MIN_DOMAIN_COUNT = 2
    filter_json_path = os.path.join("output", "domain_boilerplate.json")
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    domain_filter_obj = get_boilerplate_filter(
        filter_json_path,
        from_file = config.init_domain_filter,
        sample_thresh=FILTER_THRESHOLD,
        min_count=MIN_DOMAIN_COUNT
    )
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        domain_map[e["domain"]].append(e)
    sorted_domains = sorted(domain_map.keys(), key=lambda d: len(domain_map[d]), reverse=True)
    # initialize (and possibly warm up) the filter by *just fetching text*, ignoring KeyBERT for now
    print("=== PASS 1: Accumulate and lock domain boilerplate. ===")
    run_boilerplate_filtering(sorted_domains, domain_filter_obj, domain_map, FILTER_THRESHOLD, MIN_DOMAIN_COUNT)
    print("=== PASS 2: Keyword Extraction with domain-based filtering. ===")
    # Re-fetch text; #?should probably store them in memory from the filtering, but it only fetches as many as needed
    # Prepare KeyBERT
    keyword_model = KeyBertKeywordModel(
        model_name=config.model_name,
        top_k=config.keyword_top_k,
        prefetch_factor=1  # or config.chunk_size if you want batch logic
    )
    final_results = []
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
    # Optionally, if new domains or new lines have been discovered, re-save the domain filter
    #if config.init_domain_filter: # might make this a separate config argument later
    domain_filter_obj.save_boilerplate_map(filter_json_path)


##############################################################################################################
# TESTING BERTTopic model in new pipeline - still need to consolidate them and handle the chunking logic later
##############################################################################################################

def run_pipeline_with_bertopic(config: Config):
    # TODO: include the domain filtering back in - written by Copilot so it missed some stuff
    from onetab_autosorter.keyword_extraction import BERTTopicKeywordModel
    # 1. Load entries, maybe do domain filtering, etc.
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    # 2. Create BERTTopic model
    # pass in nr_topics if you want to reduce the total number of topics discovered
    topic_model = BERTTopicKeywordModel(
        model_name=config.model_name,
        nr_topics=None,  # or "auto", or some integer
        prefetch_factor=config.chunk_size
    )
    # 3. Fit the model to the entire set of entries
    # We'll just call generate_stream for consistency
    final_results = topic_model.generate_stream(entries)
    final_results = list(final_results)
    # 4. Save final JSON (now each entry has "topic_id" and "topic_keywords")
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)



if __name__ == "__main__":
    config = get_cfg_from_cli()
    # defaults to False
    if config.use_java_scraper:
        print("Using Java-based scraper for HTML webcrawling.")
        run_pipeline_with_scraper(config)
    run_pipeline(config)
