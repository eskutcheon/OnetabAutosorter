import os #, sys
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict



DEFAULT_YAML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r"default_filter_order.yaml")
SCRAPER_OPTIONS = ["none", "limited", "java", "naive", "async"]
KEYWORD_MODEL_NAMES = ["keybert", "bertopic"]
CLUSTERING_ALGORITHMS = ["hdbscan", "kmeans"] #, "agglomerative", "birch"]
SUPPORTED_MODEL_BACKBONES = [
    "all-MiniLM-L6-v2",     # "sentence-transformers/all-MiniLM-L6-v2"
    "all-MiniLM-L12-v2",    # "sentence-transformers/all-MiniLM-L12-v2"
    "all-distilroberta-v1",
    "all-mpnet-base-v2",
    #"allenai/scibert_scivocab_uncased",
    "intfloat/e5-base-v2",
    "distilroberta-base"
]
DEFAULT_STAGES = ["parsed", "scraped", "domain_filter", "cleaned", "keywords", "embeddings", "clustered", "labeled"]
# "data_dependent" is whether the stage hash is built only on the data, not the config (helpful when the results won't change regardless of input parameters (e.g. scraped webpage data))
DEFAULT_STAGE_SETTINGS = {
    "parsed": {"reuse": False, "save": False, "data_dependent": False},
    "scraped": {"reuse": False, "save": True, "data_dependent": True},
    # change to not data dependent if I add any arguments for the tokenizer later - for now results are only determined by input data
    "domain_filter": {"reuse": False, "save": False, "data_dependent": True},
    "cleaned": {"reuse": False, "save": False, "data_dependent": True},
    "keywords": {"reuse": True, "save": True, "data_dependent": False},
    "embeddings": {"reuse": False, "save": False, "data_dependent": False},
    "clustered": {"reuse": False, "save": True, "data_dependent": False},
    "labeled": {"reuse": False, "save": True, "data_dependent": False},
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
    # checkpoint modes
    checkpoint_mode: str = "minimal"  # Options: "none", "minimal", "all"

    def __post_init__(self):
        # ensure the cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # Initialize default stages if not provided
        if not self.stage_settings:
            self._init_default_stages()

    def _init_default_stages(self):
        """ Initialize default settings for all pipeline stages """
        if self.checkpoint_mode == "none":
            # Don't reuse or save any cache
            for name in DEFAULT_STAGES:
                self.set_stage(name, reuse=False, save=False)
        elif self.checkpoint_mode == "minimal":
            # Only cache computationally expensive stages
            for name in DEFAULT_STAGES:
                self.set_stage(name, **DEFAULT_STAGE_SETTINGS[name])
        elif self.checkpoint_mode == "all":
            for name in DEFAULT_STAGES:
                self.set_stage(name, reuse=True, save=True)


    def set_stage(self, name: str, reuse: bool = False, save: bool = False, data_dependent: bool = None):
        """ Add a new stage or update an existing one """
        # if name in self.stage_settings:
        #     return
        self.stage_settings[name] = StageCacheSettings(
            stage_name=name,
            load_cache=reuse,
            save_cache=save,
            only_data_dependent = data_dependent or DEFAULT_STAGE_SETTINGS[name]["data_dependent"],
            cache_dir=os.path.join(self.cache_dir, name)
        )

    def get_stage(self, name: str) -> StageCacheSettings:
        """ get stage settings by name - accessor method for stage settings """
        if name not in self.stage_settings:
            self.set_stage(name)
        return self.stage_settings[name]



@dataclass
class ModelingSettings:
    """ probably adding all of the keyword extraction, embedding, and clustering settings here and pass to the Pipeline class """
        # only the clustering algorithm argument would need to be from outside DEFAULT_MODEL_BACKBONES
    # keyword extraction settings
    keyword_model: str = "keybert"  # choose from KEYWORD_MODEL_NAMES
    keyword_backbone: str = "all-MiniLM-L6-v2" # base transformer backend models for keyword extraction
    keyword_top_k: int = 10         # number of top k keywords to extract
    # NEW keyword extraction settings to integrate - candidate labels are no longer dependent on whether they exist at all
    use_candidates: bool = False  # use candidate keywords for keyword extraction (from seed_labels)
    # embedding generation settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_features: List[str] = field(default_factory=lambda: ["keyword", "path", "subdomain", "date"])
    # clustering settings
    clustering_algorithm: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int = None  # Default to min_cluster_size if None
    # TODO: add zero-shot model backbone argument
    # labeling settings
    use_zero_shot_labels: bool = False
    label_candidate_count: int = 5
    seed_labels: List[str] = field(default_factory=list)  # seed keywords for keyword extraction
    # TODO: may want to make a user confirmation step at the end of _load_candidate_labels to show candidate and allow dropping them
    labels_from_html: Optional[str] = None  # path to the file containing seed keywords for keyword extraction


    def __post_init__(self):
        if self.keyword_backbone not in SUPPORTED_MODEL_BACKBONES:
            raise ValueError(f"Unsupported model backbone: {self.keyword_backbone}. Supported models are: {SUPPORTED_MODEL_BACKBONES}")
        if self.embedding_model not in SUPPORTED_MODEL_BACKBONES:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}. Supported models: {SUPPORTED_MODEL_BACKBONES}")
        if self.keyword_model.lower() not in KEYWORD_MODEL_NAMES:
            if self.keyword_model.lower() == "berttopic":
                raise ValueError("it's spelled 'BERTopic', not 'BERTTopic' - adding this error for clarity")
            self.keyword_model = "keybert"
        # set min_samples if not specified
        if self.min_samples is None:
            # TODO: add natural number restriction to these later
            self.min_samples = self.min_cluster_size
        if self.labels_from_html:
            self._load_candidate_labels()

    def _load_candidate_labels(self):
        """ parse an HTML file of nested bookmark folders and extract folder names for seed keywords """
        print("WARNING: seeding keywords from folders is experimental and may not yet work as expected")
        # TODO: fix problem where KeyBERT wants to match the candidate keywords exactly - might need some word2vec approach for synonyms
        if not os.path.exists(self.labels_from_html):
            raise FileNotFoundError(f"Seed keywords file not found: {self.labels_from_html}")
        from onetab_autosorter.parsers import NetscapeBookmarkParser
        from onetab_autosorter.utils.utils import prompt_to_drop_labels
        folder_tree = NetscapeBookmarkParser.extract_folder_structure_tree(self.labels_from_html)
        extracted_labels = folder_tree.extract_as_keywords()
        # print(f"Extracted seed keywords from HTML: {self.seed_labels}")
        # sys.exit(0)
        all_labels = list(set(self.seed_labels + extracted_labels))
        all_labels = prompt_to_drop_labels(all_labels)
        self.seed_labels = all_labels



@dataclass
class Config:
    input_file: str
    output_json: str = r"output/cleaned_output.json"
    scraper_type: str = "limited"  # choose from SCRAPER_OPTIONS
    chunk_size: int = 30  # Number of entries to process in each chunk
    max_tokens: int = 200 # max number of tokens to retain in each entry
    filter_config_path: Optional[str] = DEFAULT_YAML_PATH # path to the YAML file of ordered regex filter patterns)
    compiled_filters: list = None  # filled during post-init
    checkpoint_cfg: CheckpointSettings = field(default_factory=CheckpointSettings)
    model_cfg: ModelingSettings = field(default_factory=ModelingSettings)

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

    def _load_filters_from_yaml(self, path: str):
        """ Load the filter patterns from the YAML file and return them as a list of compiled regex patterns """
        import onetab_autosorter.config.patterns_registry as registry
        with open(path, 'r') as fptr:
            #? NOTE: should raise an error immediately if the file isn't found - keeping it implicit since it happens so early
            config = yaml.safe_load(fptr)
        pattern_list = []
        for attr in config.get("filter_sequence", []):
            patterns = getattr(registry, attr, None) # get each pattern from the registry and default to None if not found
            if not patterns:
                continue
            if isinstance(patterns, list):
                pattern_list.extend(patterns)
            elif patterns:
                pattern_list.append(patterns)
        return pattern_list