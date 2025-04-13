
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.config.config import Config
import onetab_autosorter.pipelines.pipeline_stages as pipe_stages



# TODO: might move this back to `pipeline_stages` later, just thought this made more sense before splitting utils and pipeline stage classes



class PipelineFactory:
    @staticmethod
    def create_pipeline(config: Config) -> pipe_stages.Pipeline:
        # TODO: still need to add the remaining conditional logic
        stages = []
        # Parsing stage
        parsing_stage = pipe_stages.ParsingStage(
            file_path=config.input_file,
            deduplicate=config.deduplicate,
            checkpoints=config.checkpoints
        )
        ###stages.append(parsing_stage) # hopefully temporary: parsing stage is separate and acts as the initial loader stage
        # Web scraping stage (conditional)
        if config.scraper_type != "none":
            scraping_stage = pipe_stages.WebScrapingStage(
                scraper_type=config.scraper_type,
                checkpoints=config.checkpoints
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
            checkpoints=config.checkpoints,
            **HARDCODED_DOMAIN_FILTER_KWARGS
        )
        stages.append(domain_filter_stage)
        # Text filtering stage
        filtering_stage = pipe_stages.TextPreprocessingStage(
            compiled_filters=config.compiled_filters,
            checkpoints=config.checkpoints,
            max_tokens=config.max_tokens
        )
        stages.append(filtering_stage)
        # Keyword extraction stage
        keyword_stage = pipe_stages.KeywordExtractionStage(
            checkpoints=config.checkpoints,
            model_type=config.keyword_model,
            backbone_model=config.model_name,
            seed_kws=config.seed_kws,
            top_k=config.keyword_top_k
        )
        stages.append(keyword_stage)
        return pipe_stages.Pipeline(stages=stages, loader_stage=parsing_stage)




#~ IDEA: create a builder class that takens the pipeline and generates a certain immutable frozen dataclass
    #~ with all the data necessary to keep in memory throughout the entire pipeline (checkpoints mostly)
    #~ alt: just manage this all from the Pipeline class
# TODO


# class PipelineFactory:
#     @staticmethod
#     def create_pipeline(config: Config) -> Pipeline:
#         stages: List[pipe_stages.PipelineStage] = []

#         ckpt_config: CheckpointSettings = config.checkpoints

#         # TODO: set each stage conditionally based on configuration details for dynamic pipeline entry
#         # Parsing and Loading Stage - may separate these into two stages later
#         # the loading function should handle dynamic loading itself based on the config
#             #! the main catch is passing an input file, which in the future may not correspond to the initial HTML but some checkpoint
#         stages.append(pipe_stages.LoadParsedEntriesStage(name="Load Entries", config=config))

#         if config.scraper_type != "none":
#             #~ REMINDER: also have the option to pass a callable but it's not implemented yet
#             # TODO: add this function call into the pipeline stage itself
#             fetcher_fn = create_fetcher(config.scraper_type)
#             if ckpt_config.reuse_scraped:
#                 stages.append(pipe_stages.LoadScrapedDataStage(name="Load Scraped Data", config=config))
#             else:
#                 stages.append(pipe_stages.WebScrapingStage(
#                     name="Web Scraping",
#                     fetcher_fn=fetcher_fn,
#                     config=config,
#                     save_ckpt=config.checkpoints.save_scraped,
#                     # TODO:
#                         #todo 1. make this a utility function that instantiates the appropriate SaveDataCallback based on the data
#                         #todo 2. create this object right after fetcher_fn and make it optionally None based on config
#                     save_cb = pipe_stages.SaveDataCallback.factory(
#                         ckpt_config.default_paths.scraped,
#                         "Scraped Data",
#                         config.checkpoints.cache_dir,
#                         config.deduplicate) if ckpt_config.save_scraped else None
#                     )
#                 )

#         if config.checkpoints.reuse_cleaned:
#             stages.append(pipe_stages.LoadCleanedDataStage(name="Load Cleaned Data", config=config))
#         else:
#             preprocessor = TextPreprocessingHandler(
#                 # Domain Boilerplate Filter - add a stage for populating the filter words and otherwise continue using it through the handler
#                 domain_filter=DomainBoilerplateFilter.load_boilerplate_map("output/domain_boilerplate.json"),
#                 cleaning_filter=TextCleaningFilter(ignore_patterns=config.compiled_filters)
#             )
#             stages.append(pipe_stages.TextFilteringStage(
#                 name="Text Filtering", 
#                 preprocessor=preprocessor,
#                 config=config,
#                 save_ckpt=config.checkpoints.save_cleaned,
#                 save_cb=pipe_stages.SaveDataCallback(save_json, "Cleaned Data", save_ckpt_path=get_hashed_path_from_string_path(config, "cleaned"))
#             ))

#         # Keyword Model Stage - #! should also be an optional step if we want to load data with keywords from disk
#         # TODO: remove the preprocessor from the Keyword Models and ensure it gets its own stage

#         stages.append(pipe_stages.KeywordExtractionStage(
#             name="Keyword Extraction",
#             model=KeyBertKeywordModel(model_name=config.model_name, candidate_labels=config.seed_kws, top_k=config.keyword_top_k),
#             config=config,
#             save_ckpt=config.checkpoints.save_final_output,
#             save_cb=pipe_stages.SaveDataCallback(save_json, "Keyword Data", save_ckpt_path=get_hashed_path_from_string_path(config, "keyword"))
#         ))

#         # TODO: flesh out the embedding and clustering steps in embedding.py before integrating it here

#         return Pipeline(stages=stages)