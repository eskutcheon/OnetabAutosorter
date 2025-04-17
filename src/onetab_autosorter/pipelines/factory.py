
from copy import deepcopy
#from typing import Dict, List, Tuple, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.config.config import Config
import onetab_autosorter.pipelines.pipeline_stages as pipe_stages


#DEFAULT_STAGES = ["parsed", "scraped", "domain_filter", "cleaned", "keywords", "embeddings", "clustered", "final_output"]


#? might rename the file to "builder.py" or something since this is actually a builder method
def create_pipeline(config: Config) -> pipe_stages.Pipeline:
    # TODO: still need to add the remaining conditional logic
    stages = []
    # Parsing stage
    parsing_stage = pipe_stages.ParsingStage(
        #? NOTE: refactor demands consistent stage names across the pipeline - reference DEFAULT_STAGES
        file_path=config.input_file,
        deduplicate=config.deduplicate,
        #checkpoints=config.checkpoint_cfg
        stage_settings=config.checkpoint_cfg.stage_settings["parsed"]
    )
    ###stages.append(parsing_stage) # hopefully temporary: parsing stage is separate and acts as the initial loader stage
    # Web scraping stage (conditional)
    if config.scraper_type.lower() != "none":
        scraping_stage = pipe_stages.WebScrapingStage(
            scraper_type=config.scraper_type.lower(),
            #checkpoints=config.checkpoint_cfg
            stage_settings=config.checkpoint_cfg.stage_settings["scraped"]
        )
        stages.append(scraping_stage)
    #! replace with config arguments later
    HARDCODED_DOMAIN_FILTER_KWARGS = {
        "min_domain_samples": 8,
        "min_phrase_freq": 0.75,
        "ngram_range": (2, 10),
        "max_features": 1000,
    }
    # optional domain filter (initial fitting) stage
    domain_filter_stage = pipe_stages.DomainFilterFittingStage(
        #checkpoints=config.checkpoint_cfg,
        stage_settings=config.checkpoint_cfg.stage_settings["domain_filter"],
        **HARDCODED_DOMAIN_FILTER_KWARGS
    )
    stages.append(domain_filter_stage)
    # Text filtering stage
    filtering_stage = pipe_stages.TextPreprocessingStage(
        compiled_filters=config.compiled_filters,
        stage_settings=config.checkpoint_cfg.stage_settings["cleaned"],
        max_tokens=config.max_tokens
    )
    stages.append(filtering_stage)
    # Keyword extraction stage
    keyword_stage = pipe_stages.KeywordExtractionStage(
        stage_settings=config.checkpoint_cfg.stage_settings["keywords"],
        model_type=config.model_settings.keyword_model,
        backbone_model=config.model_settings.model_name,
        seed_kws=config.model_settings.seed_kws,
        top_k=config.model_settings.keyword_top_k
    )
    stages.append(keyword_stage)
    #~ might make the settings_metadata argument into take a variable number of dataclasses and aggregate them into a dict
        #~ primarily meant for hashing the settings to differentiate the same input data with different run parameters
    return pipe_stages.Pipeline(
        stages=stages,
        loader_stage=parsing_stage,
        settings_metadata=[deepcopy(config.model_settings)]
    )
