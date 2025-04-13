import os
import json
from collections import defaultdict #, Counter
from typing import Optional, Dict, List, Literal, Callable, Union, Any
import polars as pl
# local imports
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel
from onetab_autosorter.config.config import Config
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter
from onetab_autosorter.utils.io_utils import PythonSetEncoder, compute_hash
from onetab_autosorter.pipelines.staging_utils import load_entries, create_fetcher, create_and_fit_domain_filter





# should be temporary with the new pipeline structure but I need to do regression testing on the keyword models:
def scrape_then_clean(entries: List[Dict[str, Any]], preprocessor: TextPreprocessingHandler, fetcher_fn: Callable) -> List[Dict[str, Any]]:
    urls = [e["url"] for e in entries]
    # TODO: need to generalize the fetcher_fn across all options to ensure it always returns the same structure
    summaries_map = fetcher_fn(urls)
    # TODO: add multi-threaded batched support for preprocessor (only the non-domain-filtering part)
    for idx in range(len(entries)):
        url = entries[idx]["url"]
        domain = entries[idx]["domain"]
        #entries[url]["scraped_text"] = summaries_map.get(url, "")
        raw_text = summaries_map.get(url, "")
        # If domain is locked or partially locked, do the filtering - filter_boilerplate() will be a no-op if domain isn't locked yet
        full_text = preprocessor.process_text(raw_text, domain, use_domain_filter=True)
        entries[idx]["clean_text"] = full_text # or maybe call it "filtered_text"
    return entries



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
    entries = load_entries(config.input_file, config)
    scraper_fn: Callable = create_fetcher(config.scraper_type)
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        # populate dictionary of lists with domain names as keys and entries as values
        domain_map[e["domain"]].append(e)
    # instantiate domain filter object and run the filter (we set the filter first because the keyword extractor expects locked domain keys)
    domain_filter_obj = create_and_fit_domain_filter(
        config.init_domain_filter,
        domain_map,
        json_path=filter_json_path,
        min_domain_samples=FILTER_THRESHOLD,
        scraper_fn=scraper_fn,
        #min_repeat_count=MIN_DOMAIN_COUNT,
        ngram_range = (2, 10) # determines the n-gram range for boilerplate detection, e.g. (2,5) means phrases with 2-5 words are considered
    )
    del domain_map # free memory - variables not needed anymore
    text_cleaner = TextCleaningFilter(ignore_patterns=config.compiled_filters)
    preprocessor = TextPreprocessingHandler(domain_filter_obj, text_cleaner)
    entries = scrape_then_clean(entries, preprocessor, scraper_fn)
    # Prepare KeyBERT
    keyword_model = KeyBertKeywordModel(
        model_name=config.model_name,
        candidate_labels=config.seed_kws if config.seed_kws else None,
        top_k=config.keyword_top_k,
    )
    final_results = keyword_model.run(entries) #[]
    # TODO: add early parsing to drop empty entries
    # TODO: need to add error catching for empty keywords
    #embedding_test(final_results, config)
    # with open(config.output_json, "w", encoding="utf-8") as f:
    #     json.dump(final_results, f, indent=2, cls=PythonSetEncoder)



##############################################################################################################
# TESTING BERTopic model in new pipeline - still need to consolidate them and handle the chunking logic later
##############################################################################################################

def run_pipeline_with_bertopic(config: Config):
    from onetab_autosorter.keyword_extraction import BERTopicKeywordModel
    FILTER_THRESHOLD = 5
    filter_json_path = os.path.join("output", "domain_boilerplate.json")
    # Load entries, maybe do domain filtering, etc.
    entries = load_entries(config.input_file, config)
    scraper_fn: Callable = create_fetcher(config.scraper_type)
    # select entries by domain then sort by frequency so the most frequent domains get processed first
    domain_map: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        domain_map[e["domain"]].append(e)
    domain_filter: DomainBoilerplateFilter = create_and_fit_domain_filter(
        config.init_domain_filter,
        domain_map,
        json_path=filter_json_path,
        scraper_fn=scraper_fn,
        min_domain_samples=FILTER_THRESHOLD,
        #min_repeat_count=MIN_DOMAIN_COUNT,
        ngram_range = (2, 10) # determines the n-gram range for boilerplate detection, e.g. (2,5) means phrases with 2-5 words are considered
    )
    del domain_map # free memory - variables not needed anymore
    text_cleaner = TextCleaningFilter(ignore_patterns=config.compiled_filters)
    preprocessor = TextPreprocessingHandler(domain_filter, text_cleaner)
    entries = scrape_then_clean(entries, preprocessor, fetcher_fn=scraper_fn)
    # Create BERTopic model with domain filtering + text truncation
    topic_model = BERTopicKeywordModel(
        model_name=config.model_name,
        candidate_labels=config.seed_kws if config.seed_kws else None,
        nr_topics=None,       # or 'auto', or an integer
        # preprocessor=preprocessor,  # use the TextPreprocessingHandler for cleaning
        # fetcher_fn=None       # or default_html_fetcher_batch
    )
    # Run BERTopic on the entire set of entries
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
# def _simulate_data_as_text(entry: Dict[str, Any]) -> str:




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
    #df_labeled = generate_cluster_labels_from_keywords(df_clustered)
    #print("labeled df: ", df_labeled.head(10))
    #label_map = generate_cluster_labels_llm(df_clustered, top_k=5)
    df_labeled: pl.DataFrame = generate_cluster_labels_zero_shot(df_clustered, candidate_labels=config.seed_kws)
    #df_labeled = inject_generated_labels(df_clustered, label_map.to_dict())
    df_labeled.glimpse(max_items_per_column=10, max_colname_length=160)
    df_labeled.write_csv(config.output_json.replace(".json", "_clusters.csv"))


