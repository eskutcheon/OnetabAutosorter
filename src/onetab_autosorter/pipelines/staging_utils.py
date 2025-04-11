import os
import json
from typing import Optional, Dict, List, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.config.config import Config, CheckpointSettings #, get_cfg_from_cli
from onetab_autosorter.utils.utils import detect_bookmark_format, deduplicate_entries
from onetab_autosorter.utils.io_utils import compute_hash, cache_path, get_cache_file


# meant to be called by the feature extractor at the moment, but I'm planning to move scraping and cleaning to an earlier pipeline stage
# TODO: split it into two functions
# TODO: add "Config" to a typeshed file for better type hinting
def scrape_and_cache(urls: List[str], fetcher_fn, cache_settings: CheckpointSettings):
    raw_cache_dir = os.path.join(cache_settings.cache_dir, "raw_html")
    cleaned_cache_dir = os.path.join(cache_settings.cache_dir, "cleaned")
    os.makedirs(raw_cache_dir, exist_ok=True)
    os.makedirs(cleaned_cache_dir, exist_ok=True)
    results = {}
    urls_to_fetch = []
    #! FIXME: think this is using the wrong entries, since the cleaned entries are from after cleaning past scraping results
    for url in urls:
        url_hash = compute_hash(url)
        raw_path = cache_path(raw_cache_dir, url_hash, "txt")
        cleaned_path = cache_path(cleaned_cache_dir, url_hash, "txt")
        if cache_settings.reuse_cleaned and os.path.exists(cleaned_path):
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                results[url] = f.read()
        else:
            urls_to_fetch.append(url)
    fetched_results = fetcher_fn(urls_to_fetch)
    for url in urls_to_fetch:
        url_hash = compute_hash(url)
        raw_text = fetched_results.get(url, "")
        raw_path = cache_path(raw_cache_dir, url_hash, "txt")
        if cache_settings.save_scraped:
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
        results[url] = raw_text
    return results



def merge_entries_with_scraped(entries: List[Dict[str, Any]], scraped_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    scraped_dict: Dict[str, Dict[str, Any]] = {item['url']: item for item in scraped_data}
    for entry in entries:
        url: StopAsyncIteration = entry['url']
        if url in scraped_dict:
            entry['raw_html'] = scraped_dict[url].get('raw_html', "")
    return entries


def load_entries(input_file: str, config: Config) -> List[Dict[str, Any]]:
    # TODO: extend the logic and make this function a dispatcher that calls, among others, load_json
    # load the entries from the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as fptr:
        entries = json.load(fptr)
    return entries











def get_parser(file_path: str):
    from onetab_autosorter.parsers import OneTabParser, JSONParser, NetscapeBookmarkParser
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".html":
        html_type = detect_bookmark_format(file_path)
        if html_type == "onetab":
            return OneTabParser()
        elif html_type == "netscape":
            return NetscapeBookmarkParser()
        else:
            raise ValueError(f"Unsupported HTML format: {html_type}")
    elif ext == ".json":
        return JSONParser()
    else:
        raise ValueError(f"Unsupported input format: {ext}")


def get_parsed_entries(
    file_path: str,
    cache_file: str,
    cache_settings: CheckpointSettings,
    deduplicate: bool = True
) -> List[Dict[str, Any]]:
    """ get parser object depending on file input type and run the parser to get the initial entries """
    parser = get_parser(file_path)
    entries = parser.parse(file_path)
    if deduplicate:
        entries = deduplicate_entries(entries) # using the default max url length of 200 to skip testing long urls
    if not entries:
        raise RuntimeError("No entries found in the input file.")
    if cache_settings.save_parsed:
        os.makedirs(cache_settings.cache_dir, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f)
    return entries


#~ later, this will be the first stage of the pipeline
def load_entries(file_path: str, config: Config) -> List[Dict[str, Any]]:
    """ load entries from a file, either from cache or by parsing the original HTML file """
    from onetab_autosorter.utils.io_utils import compute_hash, cache_path
    cache_settings = config.checkpoints
    parsed_hash = compute_hash(file_path)
    cache_file = cache_path(cache_settings.cache_dir, f"parsed_{parsed_hash}")
    if cache_settings.reuse_parsed and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return get_parsed_entries(file_path, cache_file, cache_settings, config.deduplicate)



def get_fetcher_function(scraper_type: Union[Callable, Literal["none", "java", "naive", "limited", "async"]] = "limited", **kwargs) -> Callable:
    """ returns a "fetcher" function for webscraping supplementary text based on `scraper_type`
        Args:
            scraper_type (str): one of ["webscraper", "default", "java", "async"]
            kwargs: additional parameters passed to WebScraper constructor
        Returns:
            Callable: a callable that accepts List[str] -> Dict[str, str]
    """
    scraper_type = scraper_type.lower() if isinstance(scraper_type, str) else scraper_type
    if callable(scraper_type):
        #? NOTE: may be better to pass a wrapper function that handles the kwargs rather than passing them to the callable directly
        print("WARNING: (in `get_fetcher_function`) `scraper_type` passed as a callable is untested and may lead to unexpected behavior.")
        return scraper_type
    if scraper_type == "none":
        return None
    elif scraper_type in ["limited", "async"]:
        from onetab_autosorter.scraper.webscraper import WebScraper
        scraper = WebScraper(**kwargs) #rate_limit_delay=1.2, max_workers=8)
        scrape_func = "run_async_fetch_batch" if scraper_type == "async" else "fetch_batch"
        return getattr(scraper, scrape_func)
    elif scraper_type == "java":
        from onetab_autosorter.scraper.client import ScraperServiceManager, fetch_summary_batch
        scraper = ScraperServiceManager()
        return scraper.fetch_within_context(fetch_summary_batch)
    elif scraper_type == "naive":
        from onetab_autosorter.scraper.scraper_utils import default_html_fetcher_batch
        return default_html_fetcher_batch
    else:
        raise ValueError(f"Invalid scraper type: {scraper_type}; expected one of ['java', 'naive', 'limited', 'async']")


def create_domain_filter(load_from_file: bool, json_path = r"output/domain_boilerplate.json", **kwargs):
    from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
    if load_from_file:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return DomainBoilerplateFilter.load_boilerplate_map(json_path, **kwargs)
    return DomainBoilerplateFilter(**kwargs)


def create_and_run_domain_filter(
    load_from_file: bool,
    domain_map: Dict[str, List[Dict]],
    json_path = r"output/domain_boilerplate.json",
    **kwargs
):
    from onetab_autosorter.scraper.webscraper import WebScraper
    MIN_REPEAT_COUNT = 3
    # instantiate domain filter object and run the filter #? (we fit the filter first because the keyword extractor expects locked domain keys)
    domain_filter_obj = create_domain_filter(load_from_file, json_path, **kwargs)
    scraper = WebScraper() #rate_limit_delay=1.2, max_workers=8)
    domain_filter_obj.run_preliminary_search(domain_map, MIN_REPEAT_COUNT, scraper.fetch_batch) #default_html_fetcher_batch)
    # TODO: add support for appending to the same file and slowly growing the filter phrases over time
    domain_filter_obj.save_boilerplate_map(json_path)
    return domain_filter_obj
