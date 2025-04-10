import os, sys
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pprint import pprint


DEFAULT_YAML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r"default_filter_order.yaml")


@dataclass
class Config:
    input_file: str
    output_json: str = r"output/cleaned_output.json"
    model_name: str = "all-MiniLM-L6-v2"
    keyword_top_k: int = 10
    deduplicate: bool = False
    dedupe_url_max_len: int = 200
    use_java_scraper: bool = False
    chunk_size: int = 30  # Number of entries to process in each chunk
    init_domain_filter: bool = False # whether to initialize the domain filter results from previous runs
    filter_config_path: Optional[str] = DEFAULT_YAML_PATH # path to the YAML file of ordered regex filter patterns)
    # TODO: add a field to allow the user to load supplemental text from a file without webscraping
    compiled_filters: list = None  # Filled post-init
    seed_kws: List[str] = field(default_factory=list)  # seed keywords for the domain filter
    seed_kws_from_html: Optional[str] = None  # path to the file containing seed keywords for the domain filter

    def __post_init__(self):
        # ensure the output path exists
        output_dir = os.path.dirname(self.output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # ensure chunk_size is a positive integer
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        self._load_candidate_keywords()
        pprint(f"Configuration initialized: {self}", indent=2)
        self.compiled_filters = self._load_filters_from_yaml(self.filter_config_path)

    def _load_candidate_keywords(self):
        if self.seed_kws_from_html:
            if not os.path.exists(self.seed_kws_from_html):
                raise FileNotFoundError(f"Seed keywords file not found: {self.seed_kws_from_html}")
            from onetab_autosorter.parsers import NetscapeBookmarkParser
            folder_tree = NetscapeBookmarkParser.extract_folder_structure_tree(self.seed_kws_from_html)
            # only save terminal nodes (i.e. folders with no subfolders) to the seed keywords list:
            self.seed_kws.extend(folder_tree.extract_as_keywords())

    def _load_filters_from_yaml(self, path: str):
        import onetab_autosorter.config.patterns_registry as registry
        with open(path, 'r') as fptr:
            #? NOTE: should raise an error immediately if the file isn't found - keeping it implicit since it happens so early
            config = yaml.safe_load(fptr)
        pattern_list = []
        for field in config.get("filter_sequence", []):
            patterns = getattr(registry, field, None) # get each pattern from the registry and default to None if not found
            if not patterns:
                continue
            if isinstance(patterns, list):
                pattern_list.extend(patterns)
            elif patterns:
                pattern_list.append(patterns)
        return pattern_list


# TODO: considering moving this to the config.py file so they can reference the same constants
def get_cfg_from_cli():
    parser = argparse.ArgumentParser(description="Bookmark Clustering Pipeline")
    parser.add_argument("input_file", help="Input HTML or JSON file (depending on what you're attempting to parse)")
    parser.add_argument("-o", "--output", default=r"output/output_entries.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="KeyBERT model")
    parser.add_argument("--top_k", type=int, default=10, help="Top K keywords")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate URLs")
    parser.add_argument("--max_url_len", type=int, default=200, help="Max URL length to dedup")
    parser.add_argument("--use_java_scraper", action="store_true", help="Use Java-based scraper for HTML webcrawling (if available)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Number of entries to process in each chunk (for webcrawling)")
    parser.add_argument("--init_domain_filter", action="store_true", help = "Whether to initialize the domain filter with results from previous runs.")
    parser.add_argument("--filter_config", type=str, default=DEFAULT_YAML_PATH, help="Path to YAML file specifying pattern filter order (as read from `patterns_registry.py`)")
    parser.add_argument("--seed_kws_from_html", type=str, default=None, help="Path to the HTML file containing bookmark folders to extract seed keywords for the extractor")
    parser.add_argument("--seed_kws", type=str, nargs='+', default=[], help="List of seed keywords for the domain filter")
    args = parser.parse_args()
    return Config(
        input_file=args.input_file,
        output_json=args.output,
        model_name=args.model,
        keyword_top_k=args.top_k,
        deduplicate=args.deduplicate,
        dedupe_url_max_len=args.max_url_len,
        use_java_scraper=args.use_java_scraper,
        chunk_size=args.chunk_size,
        init_domain_filter=args.init_domain_filter,
        filter_config_path=args.filter_config,
        seed_kws=args.seed_kws,
        seed_kws_from_html=args.seed_kws_from_html,
    )

