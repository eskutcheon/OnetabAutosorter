import os, sys
import json
from collections import defaultdict #, Counter
from typing import Optional, Dict, List, Literal, Callable, Union, Any
import polars as pl
# local imports
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel
from onetab_autosorter.parsers import OneTabParser, JSONParser, NetscapeBookmarkParser
from onetab_autosorter.config.config import Config, get_cfg_from_cli
from onetab_autosorter.utils.utils import detect_bookmark_format, deduplicate_entries, PythonSetEncoder
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter



def get_parser(file_path: str):
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

#~ later, this will be the first stage of the pipeline
def load_entries(file_path: str, deduplicate: bool = True, max_url_len: int = 200) -> List[Dict[str, Any]]:
    parser = get_parser(file_path)
    entries = parser.parse(file_path)
    if deduplicate:
        entries = deduplicate_entries(entries, max_length=max_url_len)
    if not entries:
        raise RuntimeError("No entries found in the input file.")
    # for e in entries:
    #     print(e)
    return entries



def get_fetcher_function(scraper_type: Union[Callable, Literal["java", "naive", "limited", "async"]] = "limited", **kwargs) -> Callable:
    """ returns a "fetcher" function for webscraping supplementary text based on `scraper_type`
        Args:
            scraper_type (str): one of ["webscraper", "default", "java", "async"]
            kwargs: additional parameters passed to WebScraper constructor
        Returns:
            Callable: a callable that accepts List[str] -> Dict[str, str]
    """
    if callable(scraper_type):
        print("WARNING: scraper_type passed as a callable is untested and may lead to unexpected behavior.")
        return scraper_type
    if scraper_type in ["limited", "async"]:
        from onetab_autosorter.scraper.webscraper import WebScraper
        scraper = WebScraper(**kwargs) #rate_limit_delay=1.2, max_workers=8)
        scrape_func = "run_async_fetch_batch" if scraper_type == "async" else "fetch_batch"
        return getattr(scraper, scrape_func)
    elif scraper_type == "java":
        # TODO: still kinda just want to combine these files
        from onetab_autosorter.scraper.launcher import ScraperServiceManager
        from onetab_autosorter.scraper.client import fetch_summary_batch
        scraper = ScraperServiceManager()
        return scraper.fetch_within_context(fetch_summary_batch)
    elif scraper_type == "naive":
        from onetab_autosorter.scraper.scraper_utils import default_html_fetcher_batch
        return default_html_fetcher_batch
    else:
        raise ValueError(f"Invalid scraper type: {scraper_type}; expected one of ['java', 'naive', 'limited', 'async']")




# TODO: split these two functions up to use a common interface for running the pipeline

def run_pipeline(config: Config, fetcher_fn: Optional[Callable] = None, **fetcher_kw):
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    kwargs = dict(
        model_name=config.model_name,
        prefetch_factor = config.chunk_size,  # number of entries to process in each chunk
        top_k=config.keyword_top_k,
    )
    if fetcher_fn is not None:
        kwargs['fetcher_fn'] = get_fetcher_function(fetcher_fn, **fetcher_kw)
    keyword_model = KeyBertKeywordModel(**kwargs)
    if config.chunk_size > 1:
        entries = keyword_model.generate_with_chunking(entries, config.chunk_size)
    else:
        entries = keyword_model.generate_stream(entries)
    #? NOTE: entries is now returned as a deque of dictionaries, so we may need to convert it back to a list
    entries = list(entries)  # convert deque back to list for JSON serialization
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, cls=PythonSetEncoder)




#################################################################################################
# (NEW) Keyword Extraction + Domain Boilerplate Filtering Pipeline
#################################################################################################


def create_and_run_domain_filter(
    load_from_file: bool,
    domain_map: Dict[str, List[Dict]],
    json_path = os.path.join("output", "domain_boilerplate.json"),
    **kwargs
):
    from onetab_autosorter.scraper.webscraper import WebScraper
    MIN_REPEAT_COUNT = 3
    # instantiate domain filter object and run the filter #? (we set the filter first because the keyword extractor expects locked domain keys)
    domain_filter_obj = None
    if load_from_file:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return DomainBoilerplateFilter.load_boilerplate_map(json_path, **kwargs)
    else:
        domain_filter_obj = DomainBoilerplateFilter(**kwargs)
    scraper = WebScraper() #rate_limit_delay=1.2, max_workers=8)
    domain_filter_obj.run_preliminary_search(domain_map, MIN_REPEAT_COUNT, scraper.fetch_batch) #default_html_fetcher_batch)
    #run_boilerplate_filtering(domain_filter_obj, domain_map)
    domain_filter_obj.save_boilerplate_map(json_path)
    return domain_filter_obj



#! work on getting rid of this next
def run_pipeline_with_keybert(config: Config):
    """
        1. Parse the input file into entries and optionally remove duplicates
        2. Possibly load a saved domain -> boilerplate mapping (if config.init_domain_filter is True)
        3. Group entries by domain; fetch text for each domain to build up the domain filter
        4. Lock in (finalize) boilerplate for each domain
        5. Re-fetch or reuse the text, filter it, run KeyBERT, and produce final JSON
        6. Save updated domain filter if needed
    """
    FILTER_THRESHOLD = 5
    MIN_DOMAIN_COUNT = 2
    filter_json_path = os.path.join("output", "domain_boilerplate.json")
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        domain_map[e["domain"]].append(e)
    # instantiate domain filter object and run the filter (we set the filter first because the keyword extractor expects locked domain keys)
    domain_filter_obj = create_and_run_domain_filter(
        config.init_domain_filter,
        domain_map,
        filter_json_path,
        min_domain_samples=FILTER_THRESHOLD,
        #min_repeat_count=MIN_DOMAIN_COUNT,
        ngram_range = (2, 10) # determines the n-gram range for boilerplate detection, e.g. (2,5) means phrases with 2-5 words are considered
    )
    del domain_map # free memory - variables not needed anymore
    text_cleaner = TextCleaningFilter(ignore_patterns=config.compiled_filters)
    preprocessor = TextPreprocessingHandler(domain_filter_obj, text_cleaner)
    #!####################################################################################################################################
    #! below hasn't been updated to the new structure - need to refactor KeyBertKeywordModel to use the new preprocessor and domain filter
    #!####################################################################################################################################
    # Prepare KeyBERT
    keyword_model = KeyBertKeywordModel(
        model_name=config.model_name,
        candidate_labels=config.seed_kws if config.seed_kws else None,
        top_k=config.keyword_top_k,
        preprocessor=preprocessor,  # use the TextPreprocessingHandler for cleaning
        fetcher_fn = get_fetcher_function("naive")
    )
    final_results = keyword_model.run(entries) #[]
    # TODO: add early parsing to drop empty entries
    # TODO: need to add error catching for empty keywords
    embedding_test(final_results, config)
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, cls=PythonSetEncoder)



##############################################################################################################
# TESTING BERTTopic model in new pipeline - still need to consolidate them and handle the chunking logic later
##############################################################################################################

def run_pipeline_with_bertopic(config: Config):
    # TODO: include the domain filtering back in - written by Copilot so it missed some stuff
    from onetab_autosorter.keyword_extraction import BERTTopicKeywordModel
    FILTER_THRESHOLD = 5
    filter_json_path = os.path.join("output", "domain_boilerplate.json")
    # Load entries, maybe do domain filtering, etc.
    entries = load_entries(config.input_file, deduplicate=config.deduplicate, max_url_len=config.dedupe_url_max_len)
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        domain_map[e["domain"]].append(e)
    domain_filter = create_and_run_domain_filter(
        config.init_domain_filter,
        domain_map,
        filter_json_path,
        min_domain_samples=FILTER_THRESHOLD,
        #min_repeat_count=MIN_DOMAIN_COUNT,
        ngram_range = (2, 10) # determines the n-gram range for boilerplate detection, e.g. (2,5) means phrases with 2-5 words are considered
    )
    del domain_map # free memory - variables not needed anymore
    text_cleaner = TextCleaningFilter(ignore_patterns=config.compiled_filters)
    preprocessor = TextPreprocessingHandler(domain_filter, text_cleaner)
    # Create BERTTopic model with domain filtering + text truncation
    topic_model = BERTTopicKeywordModel(
        model_name=config.model_name,
        candidate_labels=config.seed_kws if config.seed_kws else None,
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




##############################################################################################################
# for the extension to using dataframes later:
##############################################################################################################
# will only be relevant using the tabular dataframe with output keywords for clustering later
def _simulate_data_as_text(entry: Dict[str, Any]) -> str:
    """ embed domain and group info into the text """
    # TODO: move upstream since it's more key-dependent than I'd prefer
    domain_label = entry["domain"].split(".")[-2]  # "wikipedia" or "linuxfoundation"
    group_label = f"group_{entry['group_ids'][0]}" if entry["group_ids"] else ""
    artificial_text = f"domain_{domain_label} - {group_label}"
    return artificial_text.strip()  # return the simulated text for the entry, if no real text is available




def embedding_test(entries: List[Dict[str, Any]], config: Config):
    from onetab_autosorter.embeddings import (
        entries_to_dataframe,
        enrich_with_metadata,
        embed_column,
        concatenate_embeddings,
        cluster_hdbscan,
        inject_cluster_results,
        #generate_cluster_labels_from_keywords,
        generate_cluster_labels_zero_shot,
        #generate_cluster_labels_llm,
        inject_generated_labels
    )
    df = entries_to_dataframe(entries)
    print("top 10 entries: ", df.head(10))
    df_meta = enrich_with_metadata(df)
    print("top 10 metadata: ", df_meta.head(10))
    embeddings = embed_column(df, "keywords_text")
    print("embeddings shape: ", embeddings.shape)
    print("embeddings: ", embeddings[:5])
    combined = concatenate_embeddings(embeddings, df_meta.select(["kw_length", "domain_length", "group_count"]))
    print("combined shape: ", combined.shape)
    cluster_results = cluster_hdbscan(combined, min_cluster_size=5)
    print("cluster results: ", cluster_results)
    df_clustered = inject_cluster_results(df, cluster_results["labels"], cluster_results["scores"])
    print("clustered df: ", df_clustered.head(10))
    # Step 3: summarize
    #df_labeled = generate_cluster_labels_from_keywords(df_clustered)
    #print("labeled df: ", df_labeled.head(10))
    #df_labeled.glimpse(max_items_per_column=5)
    #label_map = generate_cluster_labels_llm(df_clustered, top_k=5)
    label_map: pl.DataFrame = generate_cluster_labels_zero_shot(df_clustered, candidate_labels=config.seed_kws)
    df_labeled = inject_generated_labels(df_clustered, label_map.to_dict())
    df_labeled.glimpse(max_items_per_column=10, max_colname_length=160)


if __name__ == "__main__":
    config = get_cfg_from_cli()
    # defaults to False
    # TODO: make this a config option (to replace `use_java_scraper`) later:
    fetcher_func = "java" if config.use_java_scraper else "limited"
    run_pipeline(config, fetcher_func)
