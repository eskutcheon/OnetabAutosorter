import os, sys
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pprint import pprint


DEFAULT_YAML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r"default_filter_order.yaml")
SCRAPER_OPTIONS = ["none", "limited", "java", "naive", "async"]
KEYWORD_MODEL_REGISTRY = ["keybert", "bertopic"]
DEFAULT_STAGES = ["parsed", "scraped", "domain_filter", "cleaned", "keywords", "embeddings", "clustered", "final_output"]
DEFAULT_STAGE_SETTINGS = {
    "parsed": {"reuse": False, "save": False},
    "scraped": {"reuse": True, "save": True},
    "domain_filter": {"reuse": True, "save": True},
    "cleaned": {"reuse": False, "save": False},
    "keywords": {"reuse": False, "save": True},
    "embeddings": {"reuse": False, "save": False},
    "clustered": {"reuse": False, "save": False},
    "final_output": {"reuse": False, "save": True},
}

@dataclass
class StageCacheSettings:
    stage_name: str
    load_cache: bool = False
    save_cache: bool = False
    cache_dir: str = "cache"


@dataclass
class CheckpointSettings:
    stage_settings: Dict[str, StageCacheSettings] = field(default_factory=dict)
    cache_dir: str = "cache"

    def __post_init__(self):
        # ensure the cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # Initialize default stages if not provided
        if not self.stage_settings:
            self._init_default_stages()

    def _init_default_stages(self):
        """ Initialize default settings for all pipeline stages """
        for name, setting in DEFAULT_STAGE_SETTINGS.items():
            self.set_stage(name, setting["reuse"], setting["save"])

    def set_stage(self, name: str, reuse: bool = False, save: bool = False):
        """Add a new stage or update an existing one."""
        self.stage_settings[name] = StageCacheSettings(
            stage_name=name,
            load_cache=reuse,
            save_cache=save,
            cache_dir=os.path.join(self.cache_dir, name)
        )



@dataclass
class Config:
    input_file: str
    output_json: str = r"output/cleaned_output.json"
    model_name: str = "all-MiniLM-L6-v2"
    keyword_top_k: int = 10
    deduplicate: bool = True
    scraper_type: str = "limited"  # choose from SCRAPER_OPTIONS
    keyword_model: str = "keybert"  # choose from KEYWORD_MODEL_REGISTRY
    chunk_size: int = 30  # Number of entries to process in each chunk
    max_tokens: int = 400 # max number of tokens to retain in each entry
    #& replacing with CheckpointSettings.override_boilerplate but still using it for now
    init_domain_filter: bool = False # whether to initialize the domain filter results from previous runs
    filter_config_path: Optional[str] = DEFAULT_YAML_PATH # path to the YAML file of ordered regex filter patterns)
    compiled_filters: list = None  # filled during post-init
    seed_kws: List[str] = field(default_factory=list)  # seed keywords for the domain filter
    seed_kws_from_html: Optional[str] = None  # path to the file containing seed keywords for the domain filter
    checkpoints: CheckpointSettings = field(default_factory=CheckpointSettings)

    def __post_init__(self):
        # ensure chunk_size is a positive integer
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if self.max_tokens < 10:
            raise ValueError("max_tokens must be a positive integer >= 10.")
        if self.scraper_type.lower() not in ["none", "limited", "java", "naive", "async"]:
            raise ValueError(f"Invalid scraper type: {self.scraper_type}. Must be one of {SCRAPER_OPTIONS}!")
        if self.keyword_model.lower() not in KEYWORD_MODEL_REGISTRY:
            if self.keyword_model.lower() == "BERTopic":
                raise ValueError("it's spelled BERTopic, not BERTopic - adding this error for clarity")
            self.keyword_model = "keybert"
        # ensure the output path exists
        output_dir = os.path.dirname(self.output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(self.checkpoints.cache_dir):
            os.makedirs(self.checkpoints.cache_dir)
        self._load_candidate_keywords()
        pprint(f"Configuration initialized: {self}", indent=2)
        self.compiled_filters = self._load_filters_from_yaml(self.filter_config_path)

    def _load_candidate_keywords(self):
        """ parse an HTML file of nested bookmark folders and extract folder names for seed keywords """
        if self.seed_kws_from_html:
            if not os.path.exists(self.seed_kws_from_html):
                raise FileNotFoundError(f"Seed keywords file not found: {self.seed_kws_from_html}")
            from onetab_autosorter.parsers import NetscapeBookmarkParser
            folder_tree = NetscapeBookmarkParser.extract_folder_structure_tree(self.seed_kws_from_html)
            self.seed_kws.extend(folder_tree.extract_as_keywords())

    def _load_filters_from_yaml(self, path: str):
        """ Load the filter patterns from the YAML file and return them as a list of compiled regex patterns """
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


# TODO: split this up somewhat for brevity and readability
def get_cfg_from_cli():
    parser = argparse.ArgumentParser(description="Bookmark Clustering Pipeline")
    # only required positional argument is the input file (assuming an input html or json file)
    parser.add_argument("input_file", help="Input HTML or JSON file (depending on what you're attempting to parse)")
    parser.add_argument("-o", "--output", default=r"output/output_entries.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="KeyBERT model")
    parser.add_argument("--top_k", type=int, default=10, help="Top K keywords")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate URLs")
    parser.add_argument("--scraper_type", type=str, default="limited", choices=SCRAPER_OPTIONS, help=f"Type of webscraper to use from {SCRAPER_OPTIONS}")
    parser.add_argument("--keyword_model", type=str, default="keybert", choices=KEYWORD_MODEL_REGISTRY, help=f"Keyword extraction model to use from {KEYWORD_MODEL_REGISTRY}")
    parser.add_argument("--chunk_size", type=int, default=50, help="Number of entries to process in each chunk (for webcrawling)")
    parser.add_argument("--max_tokens", type=int, default=400, help="Max number of tokens to retain in each entry (min: 10)")
    #& may replace with Config.checkpoints.override_boilerplate later, but using it for now while debugging
    parser.add_argument("--init_domain_filter", action="store_true", help = "Whether to initialize the domain filter with results from previous runs.")
    parser.add_argument("--filter_config", type=str, default=DEFAULT_YAML_PATH, help="Path to YAML file specifying pattern filter order (as read from `patterns_registry.py`)")
    parser.add_argument("--seed_kws_from_html", type=str, default=None, help="Path to the HTML file containing bookmark folders to extract seed keywords for the extractor")
    parser.add_argument("--seed_kws", type=str, nargs='+', default=[], help="List of seed keywords for the domain filter")
    # Optional config file load
    parser.add_argument("--opts", type=str, help="Optional YAML file to override CLI args")
    # Group for checkpoint-related flags
    ckpt = parser.add_argument_group("Checkpointing + Caching")
    for stage in DEFAULT_STAGES:
        stage_group = parser.add_argument_group(f"{stage.capitalize()} Stage")
        stage_group.add_argument(f"--reuse_{stage}", action="store_true", help=f"Reuse existing {stage} data")
        stage_group.add_argument(f"--save_{stage}", action="store_true", help=f"Save {stage} data")
    # settings for reusing cached files
    ckpt.add_argument("--cache_dir", default="cache")
    # run the parser and return the config object
    args: argparse.Namespace = parser.parse_args()
    # check if their was a file passed in to override the defaults and set args with its values
    if args.opts and os.path.exists(args.opts):
        with open(args.opts, "r") as f:
            opts = yaml.safe_load(f)
        for key, value in opts.items():
            setattr(args, key, value)
    # Construct CheckpointSettings
    ckpt_cfg = CheckpointSettings(
        cache_dir=args.cache_dir,
    )
    for stage, setting in DEFAULT_STAGE_SETTINGS.items():
        reuse_attr = f"reuse_{stage}"
        save_attr = f"save_{stage}"
        #? NOTE: absence of the attribute would raise an error without a default but they should all be covered by the CLI args
        #? NOTE: peculiar structure here is so that user choices (always true if present) aren't overridden while still letting
            #? us set defaults other than False from the CLI
        reuse = getattr(args, reuse_attr)
        reuse = reuse or setting["reuse"] if not reuse else reuse
        save = getattr(args, save_attr)
        save = save or setting["save"] if not save else save
        ckpt_cfg.set_stage(stage, reuse=reuse, save=save)
    return Config(
        input_file=args.input_file,
        output_json=args.output,
        model_name=args.model,
        keyword_top_k=args.top_k,
        deduplicate=args.deduplicate,
        scraper_type=args.scraper_type,
        keyword_model=args.keyword_model,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        init_domain_filter=args.init_domain_filter,
        filter_config_path=args.filter_config,
        seed_kws=args.seed_kws,
        seed_kws_from_html=args.seed_kws_from_html,
        checkpoints=ckpt_cfg
    )

