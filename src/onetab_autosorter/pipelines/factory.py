
#from typing import Dict, List, Tuple, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.config.config import Config
import onetab_autosorter.pipelines.pipeline_stages as pipe_stages


#DEFAULT_STAGES = ["parsed", "scraped", "domain_filter", "cleaned", "keywords", "embeddings", "clustered", "final_output"]



def create_pipeline(config: Config) -> pipe_stages.Pipeline:
    # TODO: still need to add the remaining conditional logic
    stages = []
    # Parsing stage
    parsing_stage = pipe_stages.ParsingStage(
        #? NOTE: refactor demands consistent stage names across the pipeline - reference DEFAULT_STAGES
        file_path=config.input_file,
        deduplicate=config.deduplicate,
        #checkpoints=config.checkpoints
        stage_settings=config.checkpoints.stage_settings["parsed"]
    )
    ###stages.append(parsing_stage) # hopefully temporary: parsing stage is separate and acts as the initial loader stage
    # Web scraping stage (conditional)
    if config.scraper_type.lower() != "none":
        scraping_stage = pipe_stages.WebScrapingStage(
            scraper_type=config.scraper_type.lower(),
            #checkpoints=config.checkpoints
            stage_settings=config.checkpoints.stage_settings["scraped"]
        )
        stages.append(scraping_stage)
    #! replace with config arguments later
    HARDCODED_DOMAIN_FILTER_KWARGS = {
        "min_domain_samples": 5,
        "min_df_ratio": 0.8,
        "ngram_range": (2, 10),
        "max_features": 1000,
    }
    # optional domain filter (initial fitting) stage
    domain_filter_stage = pipe_stages.DomainFilterFittingStage(
        #checkpoints=config.checkpoints,
        stage_settings=config.checkpoints.stage_settings["domain_filter"],
        **HARDCODED_DOMAIN_FILTER_KWARGS
    )
    stages.append(domain_filter_stage)
    # Text filtering stage
    filtering_stage = pipe_stages.TextPreprocessingStage(
        compiled_filters=config.compiled_filters,
        stage_settings=config.checkpoints.stage_settings["cleaned"],
        max_tokens=config.max_tokens
    )
    stages.append(filtering_stage)
    # Keyword extraction stage
    keyword_stage = pipe_stages.KeywordExtractionStage(
        stage_settings=config.checkpoints.stage_settings["keywords"],
        model_type=config.keyword_model,
        backbone_model=config.model_name,
        seed_kws=config.seed_kws,
        top_k=config.keyword_top_k
    )
    stages.append(keyword_stage)
    return pipe_stages.Pipeline(stages=stages, loader_stage=parsing_stage)
