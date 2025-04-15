import os #, sys
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict
#from pprint import pprint


# TODO: put all of these in a registry file somewhere - preferably so that it's only accessible from this file
DEFAULT_YAML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r"default_filter_order.yaml")
SCRAPER_OPTIONS = ["none", "limited", "java", "naive", "async"]
KEYWORD_MODEL_NAMES = ["keybert", "bertopic"]
SUPPORTED_MODEL_BACKBONES = [
    "all-MiniLM-L6-v2",     # "sentence-transformers/all-MiniLM-L6-v2"
    "all-MiniLM-L12-v2",    # "sentence-transformers/all-MiniLM-L12-v2"
    "all-distilroberta-v1",
    "all-mpnet-base-v2",
    "allenai/scibert_scivocab_uncased",
    "intfloat/e5-base-v2"
]
DEFAULT_STAGES = ["parsed", "scraped", "domain_filter", "cleaned", "keywords", "embeddings", "clustered", "final_output"]
DEFAULT_STAGE_SETTINGS = {
    "parsed": {"reuse": False, "save": False, "data_dependent": False},
    "scraped": {"reuse": True, "save": True, "data_dependent": True},
    "domain_filter": {"reuse": False, "save": False, "data_dependent": True},
    "cleaned": {"reuse": False, "save": False, "data_dependent": True},
    "keywords": {"reuse": False, "save": True, "data_dependent": False},
    "embeddings": {"reuse": False, "save": False, "data_dependent": False},
    "clustered": {"reuse": False, "save": False, "data_dependent": False},
    "final_output": {"reuse": False, "save": True, "data_dependent": False},
}

@dataclass
class StageCacheSettings:
    stage_name: str
    load_cache: bool = False
    save_cache: bool = False
    only_data_dependent: bool = False  # if True, the stage hash will use a hash built only on the data, not the config
    cache_dir: str = "cache"

    def __repr__(self):
        return f"{self.__class__.__name__}(stage_name={self.stage_name}, load_cache={self.load_cache}, save_cache={self.save_cache})"


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
            self.set_stage(name, setting["reuse"], setting["save"], setting["data_dependent"])

    def set_stage(self, name: str, reuse: bool = False, save: bool = False, data_dep: bool = False):
        """Add a new stage or update an existing one."""
        self.stage_settings[name] = StageCacheSettings(
            stage_name=name,
            load_cache=reuse,
            save_cache=save,
            only_data_dependent=data_dep,
            cache_dir=os.path.join(self.cache_dir, name)
        )


@dataclass
class ModelingSettings:
    """ probably adding all of the keyword extraction, embedding, and clustering settings here and pass to the Pipeline class"""
    model_name: str = "all-MiniLM-L6-v2"
    keyword_model: str = "keybert"  # choose from KEYWORD_MODEL_NAMES
    keyword_top_k: int = 10
    seed_kws: List[str] = field(default_factory=list)  # seed keywords for keyword extraction
    seed_kws_from_html: Optional[str] = None  # path to the file containing seed keywords for keyword extraction

    def __post_init__(self):
        if self.model_name.lower() not in SUPPORTED_MODEL_BACKBONES:
            raise ValueError(f"Unsupported model backbone: {self.model_name}. Supported models are: {SUPPORTED_MODEL_BACKBONES}")
        if self.keyword_model.lower() not in KEYWORD_MODEL_NAMES:
            if self.keyword_model.lower() == "berttopic":
                raise ValueError("it's spelled BERTopic, not BERTTopic - adding this error for clarity")
            self.keyword_model = "keybert"
        self._load_candidate_keywords()

    def _load_candidate_keywords(self):
        """ parse an HTML file of nested bookmark folders and extract folder names for seed keywords """
        if self.seed_kws_from_html:
            print("WARNING: seeding keywords from folders is experimental and may not yet work as expected")
            # TODO: fix problem where KeyBERT wants to match the candidate keywords exactly - might need some word2vec approach for synonyms
            if not os.path.exists(self.seed_kws_from_html):
                raise FileNotFoundError(f"Seed keywords file not found: {self.seed_kws_from_html}")
            from onetab_autosorter.parsers import NetscapeBookmarkParser
            folder_tree = NetscapeBookmarkParser.extract_folder_structure_tree(self.seed_kws_from_html)
            self.seed_kws.extend(folder_tree.extract_as_keywords())


@dataclass
class Config:
    input_file: str
    output_json: str = r"output/cleaned_output.json"
    deduplicate: bool = True
    scraper_type: str = "limited"  # choose from SCRAPER_OPTIONS
    chunk_size: int = 30  # Number of entries to process in each chunk
    max_tokens: int = 400 # max number of tokens to retain in each entry
    filter_config_path: Optional[str] = DEFAULT_YAML_PATH # path to the YAML file of ordered regex filter patterns)
    compiled_filters: list = None  # filled during post-init
    checkpoint_cfg: CheckpointSettings = field(default_factory=CheckpointSettings)
    model_settings: ModelingSettings = field(default_factory=ModelingSettings)

    def __post_init__(self):
        # ensure chunk_size is a positive integer
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if self.max_tokens < 10:
            raise ValueError("max_tokens must be a positive integer >= 10.")
        if self.scraper_type.lower() not in ["none", "limited", "java", "naive", "async"]:
            raise ValueError(f"Invalid scraper type: {self.scraper_type}. Must be one of {SCRAPER_OPTIONS}!")
        # ensure the output path exists
        output_dir = os.path.dirname(self.output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(self.checkpoint_cfg.cache_dir):
            os.makedirs(self.checkpoint_cfg.cache_dir)
        # pprint(f"Configuration initialized: {self}", indent=2)
        self.compiled_filters = self._load_filters_from_yaml(self.filter_config_path)

    # TODO: may move this function outside this class and into utils, or at least make it a staticmethod
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



# TODO: move code below into a new `cli.py` file and import config objects as needed from this file.
# TODO: split this up somewhat for brevity and readability
def get_cfg_from_cli():
    parser = argparse.ArgumentParser(description="Bookmark Clustering Pipeline")
    # only required positional argument is the input file (assuming an input html or json file)
    parser.add_argument("input_file", help="Input HTML or JSON file (depending on what you're attempting to parse)")
    parser.add_argument("-o", "--output", default=r"output/output_entries.json", help="Output JSON path")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate URLs")
    parser.add_argument("--scraper_type", type=str, default="limited", choices=SCRAPER_OPTIONS, help=f"Type of webscraper to use from {SCRAPER_OPTIONS}")
    parser.add_argument("--chunk_size", type=int, default=50, help="Number of entries to process in each chunk (for webcrawling)")
    parser.add_argument("--max_tokens", type=int, default=400, help="Max number of tokens to retain in each entry (min: 10)")
    parser.add_argument("--filter_config", type=str, default=DEFAULT_YAML_PATH, help="Path to YAML file specifying pattern filter order (as read from `patterns_registry.py`)")
    # Optional config file load
    parser.add_argument("--opts", type=str, help="Optional YAML file to override CLI args")
    # Group for modeling (keyword extraction + more?) settings
    model_group = parser.add_argument_group("Modeling Settings")
    model_group.add_argument("--backbone_model", type=str, default="all-MiniLM-L6-v2", choices=SUPPORTED_MODEL_BACKBONES, help=f"KeyBERT or BERTopic model backbone, supporting\n{SUPPORTED_MODEL_BACKBONES}")
    model_group.add_argument("--keyword_model", type=str, default="keybert", choices=KEYWORD_MODEL_NAMES, help=f"Keyword extraction model to use from {KEYWORD_MODEL_NAMES}")
    model_group.add_argument("--top_k", type=int, default=10, help="Top K keywords")
    model_group.add_argument("--seed_kws_from_html", type=str, default=None, help="Path to the HTML file containing bookmark folders to extract seed keywords for the extractor")
    model_group.add_argument("--seed_kws", type=str, nargs='+', default=[], help="List of seed keywords for the domain filter")

    # Group for checkpoint-related flags
    ckpt = parser.add_argument_group("Checkpointing + Caching")
    for stage in DEFAULT_STAGES:
        stage_group = parser.add_argument_group(f"{stage.capitalize()} Stage")
        stage_group.add_argument(f"--reuse_{stage}", action="store_true", help=f"Reuse existing {stage} data")
        stage_group.add_argument(f"--save_{stage}", action="store_true", help=f"Save {stage} data")
        # TODO: RENAME
        stage_group.add_argument(f"--hash_only_data_{stage}", action="store_true", help=f"Only use data-dependent hash for {stage} stage")
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
    def get_stage_attr_value(stage: str, attr: str, default_value: bool) -> StageCacheSettings:
        #? NOTE: peculiar structure here is so that user choices (always true if present) aren't overridden while still letting
            #? us set defaults other than False from the CLI
        attr_name = f"{attr}_{stage}"
        return getattr(args, attr_name) or default_value
    # set the stage settings based on the CLI arguments
    for stage, setting in DEFAULT_STAGE_SETTINGS.items():
        reuse = get_stage_attr_value(stage, "reuse", setting["reuse"])
        save = get_stage_attr_value(stage, "save", setting["save"])
        data_dependent = get_stage_attr_value(stage, "hash_only_data", setting["data_dependent"])
        #? NOTE: absence of the attribute would raise an error without a default but they're all False unless specified from CLI
        ckpt_cfg.set_stage(stage, reuse=reuse, save=save, data_dep=data_dependent)

    modeling_cfg = ModelingSettings(
        model_name=args.backbone_model,
        keyword_model=args.keyword_model,
        keyword_top_k=args.top_k,
        seed_kws=args.seed_kws,
        seed_kws_from_html=args.seed_kws_from_html
    )

    return Config(
        input_file=args.input_file,
        output_json=args.output,
        deduplicate=args.deduplicate,
        scraper_type=args.scraper_type,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        filter_config_path=args.filter_config,
        checkpoint_cfg=ckpt_cfg,
        model_settings=modeling_cfg,
    )

