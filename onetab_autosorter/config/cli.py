
import os
import json
import argparse
import yaml
from typing import Optional, List, Dict, Any
from onetab_autosorter.config.config import (
    Config, ModelingSettings, CheckpointSettings,
    DEFAULT_YAML_PATH, SCRAPER_OPTIONS, KEYWORD_MODEL_NAMES,
    SUPPORTED_MODEL_BACKBONES, DEFAULT_STAGES, CLUSTERING_ALGORITHMS
)



def load_yaml_opts(yaml_path: str) -> Dict[str, Any]:
    """ optionally load options from YAML file """
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}

def create_parser():
    parser = argparse.ArgumentParser(description="Bookmark Clustering Pipeline")
    # only required positional argument is the input file (assuming an input html or json file)
    parser.add_argument("input_file", help="Input HTML or JSON file (depending on what you're attempting to parse)")
    #~ may not keep but may have some flag for whether to create a new sorted bookmark HTML file ready to import to a browser
    parser.add_argument("-o", "--output", default=r"output/output_entries.json", help="Output JSON path")
    # optional config file load
    parser.add_argument("--opts", type=str, help="Optional YAML file to override CLI args")
    prep_group = parser.add_argument_group("Webscraping + Preprocessing Settings")
    prep_group.add_argument("--scraper_type", type=str, default="limited", choices=SCRAPER_OPTIONS, help=f"Type of webscraper to use")
    # TODO: maybe rename to make it more explicit that this is related to scraping chunk sizes
    prep_group.add_argument("--chunk_size", type=int, default=50, help="Number of entries in each chunk while webscraping")
    prep_group.add_argument("--max_tokens", type=int, default=400, help="Max tokens to keep per entry (min: 10)")
    prep_group.add_argument("--filter_config", type=str, default=DEFAULT_YAML_PATH, help="YAML file path with pattern filter order")
    # checkpointing settings
    ckpt_group = parser.add_argument_group("Pipeline Checkpointing")
    ckpt_group.add_argument("--checkpoint_mode", type=str, default="minimal", choices=["none", "minimal", "all"], help="Checkpointing mode")
    ckpt_group.add_argument("--cache_dir", default="cache", help="Root destination directory for cached files")
    ckpt_group.add_argument("--cache_stages", nargs="+", choices=DEFAULT_STAGES, help="Specific stages to cache (overrides mode)")
    # TODO: should make "parsing" the default
    ckpt_group.add_argument("--load_stage", type=str, choices=DEFAULT_STAGES, help="Load from this stage and continue pipeline")
    # (modeling) keyword extraction and embedding generation settings
    model_group = parser.add_argument_group("Keyword Extraction + Embedding Settings")
    model_group.add_argument("--keyword_model", type=str, default="keybert", choices=KEYWORD_MODEL_NAMES, help=f"Keyword extraction model to use from {KEYWORD_MODEL_NAMES}")
    model_group.add_argument("--keyword_backbone", type=str, default="all-MiniLM-L6-v2", choices=SUPPORTED_MODEL_BACKBONES, help=f"KeyBERT or BERTopic model backbone, supporting\n{SUPPORTED_MODEL_BACKBONES}")
    model_group.add_argument("--top_k", type=int, default=10, help="Top K keywords")
    model_group.add_argument("--labels_from_html", type=str, default=None, help="Path to the HTML file containing bookmark folders to extract seed keywords for the extractor")
    model_group.add_argument("--seed_labels", type=str, nargs='+', default=[], help="List of seed keywords for the domain filter")
    model_group.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", choices=SUPPORTED_MODEL_BACKBONES, help="Backbone NLP model for generating embeddings")
    # TODO: edit these defaults before use
    model_group.add_argument("--embedding_features", type=str, nargs="+", default=["keyword", "path", "subdomain", "date"], help="Features to include in embeddings")
    # (sorting/labeling) clustering and final label generation settings
    sorting_group = parser.add_argument_group("Clustering + Labeling Settings")
    sorting_group.add_argument("--clustering_algorithm", type=str, default="hdbscan", choices=CLUSTERING_ALGORITHMS, help="Clustering algorithm")
    sorting_group.add_argument("--min_cluster_size", type=int, default=5, help="Minimum size of clusters")
    sorting_group.add_argument("--min_samples", type=int, help="Min samples for HDBSCAN (defaults to min_cluster_size)")
    sorting_group.add_argument("--use_zero_shot_labels", action="store_true", help="Use zero-shot classification for cluster labels")
    # TODO: RENAME - related to fuzzy clustering
    sorting_group.add_argument("--label_candidate_count", type=int, default=5, help="Number of candidate labels per cluster")
    return parser



def build_config_from_args(args: Optional[argparse.Namespace] = None) -> Config:
    """ build Config object from parsed arguments """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    # override with YAML if specified
    if args.opts:
        yaml_opts = load_yaml_opts(args.opts)
        # update args with YAML values (properly handles nested attributes)
        for key, value in yaml_opts.items():
            if hasattr(args, key):
                setattr(args, key, value)
    # build checkpoint settings
    ckpt_cfg = CheckpointSettings(
        cache_dir=args.cache_dir,
        checkpoint_mode=args.checkpoint_mode,
    )
    # handle cache_stages override
    if args.cache_stages:
        for stage in DEFAULT_STAGES:
            if stage in args.cache_stages:
                ckpt_cfg.set_stage(stage, reuse=True, save=False)
            else:
                ckpt_cfg.set_stage(stage, reuse=False, save=False)
    # handle loading from a specific stage possibly specified by load_stage
    #! FIXME: not really working yet
    if args.load_stage:
        found_start = False
        for stage in DEFAULT_STAGES:
            if stage == args.load_stage:
                found_start = True
            ckpt_cfg.set_stage(stage, reuse=found_start, save=False)#ckpt_cfg.get_stage(stage).save_cache)
    # build modeling settings
    model_cfg = ModelingSettings( # start keyword extraction settings
        keyword_model = args.keyword_model,
        keyword_backbone = args.keyword_backbone,
        keyword_top_k = args.top_k,
        seed_labels = args.seed_labels,
        labels_from_html = args.labels_from_html, # begin embedding settings
        embedding_model = args.embedding_model,
        embedding_features = args.embedding_features, # begin clustering settings
        clustering_algorithm = args.clustering_algorithm,
        min_cluster_size = args.min_cluster_size,
        min_samples = args.min_samples, # begin labeling settings
        use_zero_shot_labels = args.use_zero_shot_labels,
        # TODO: RENAME
        label_candidate_count = args.label_candidate_count
    )
    # build and return the final Config object
    return Config(
        input_file = args.input_file,
        output_json = args.output,
        scraper_type = args.scraper_type,
        chunk_size = args.chunk_size,
        max_tokens = args.max_tokens,
        filter_config_path = args.filter_config,
        checkpoint_cfg = ckpt_cfg,
        model_cfg = model_cfg
    )


def main():
    """ main entry point for the CLI to support running the pipeline """
    config = build_config_from_args()
    # import here to avoid circular imports
    from onetab_autosorter.pipelines.factory import create_pipeline
    # create and run pipeline
    pipeline = create_pipeline(config)
    results = pipeline.run(config.input_file)
    # Save results
    with open(config.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Pipeline run complete - Results saved to {config.output_json}")


if __name__ == "__main__":
    main()