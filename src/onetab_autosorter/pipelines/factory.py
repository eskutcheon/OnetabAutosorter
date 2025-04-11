from typing import Optional, Dict, List, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel
from onetab_autosorter.config.config import Config, CheckpointSettings
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter

import onetab_autosorter.pipelines.pipeline_stages as pipe_stages



# TODO: honestly this might as well be a function if more complex logic is in the stages
class Pipeline:
    stages: List[pipe_stages.PipelineStage]

    def run(self, initial_input: Any) -> Any:
        data = initial_input
        for stage in self.stages:
            # TODO: modify this to return data and an additional (possibly empty) dictionary of additional args/kwargs to propagate to the next stage
            data = stage.run(data)
            stage.save_data(data)
                # save the data by calling a dispatcher function that calls the appropriate saving function based on the data type
                    # it should take a variable number of positional arguments for potentially multiple data
                ###some_central_saving_function(data, output_path=stage.save_ckpt_path)  # replace with actual saving logic
        return data


class PipelineFactory:
    @staticmethod
    def create_pipeline(config: Config) -> Pipeline:
        stages: List[pipe_stages.PipelineStage] = []

        ckpt_config: CheckpointSettings = config.checkpoints

        # TODO: set each stage conditionally based on configuration details for dynamic pipeline entry
        # Parsing and Loading Stage - may separate these into two stages later
        # the loading function should handle dynamic loading itself based on the config
            #! the main catch is passing an input file, which in the future may not correspond to the initial HTML but some checkpoint
        stages.append(pipe_stages.LoadParsedEntriesStage(name="Load Entries", config=config))

        if config.scraper_type != "none":
            #~ REMINDER: also have the option to pass a callable but it's not implemented yet
            fetcher_fn = get_fetcher_function(config.scraper_type)
            save_cb = pipe_stages.SaveDataCallback.callback_creator(ckpt_config.scraped, "Scraped Data", config.checkpoints.cache_dir, config.deduplicate)
            stages.append(pipe_stages.WebScrapingStage(
                name="Web Scraping",
                fetcher_fn=fetcher_fn,
                config=config,
                save_ckpt=config.checkpoints.save_scraped,
                # TODO:
                    #todo 1. make this a utility function that instantiates the appropriate SaveDataCallback based on the data
                    #todo 2. create this object right after fetcher_fn and make it optionally None based on config
                save_cb=pipe_stages.SaveDataCallback(save_json, "Scraped Data", save_ckpt_path=get_cache_file_path(config, "scraped"))
            ))

        if config.checkpoints.reuse_cleaned:
            stages.append(pipe_stages.LoadCleanedDataStage(name="Load Cleaned Data", config=config))
        else:
            preprocessor = TextPreprocessingHandler(
                # Domain Boilerplate Filter - add a stage for populating the filter words and otherwise continue using it through the handler
                domain_filter=DomainBoilerplateFilter.load_boilerplate_map("output/domain_boilerplate.json"),
                cleaning_filter=TextCleaningFilter(ignore_patterns=config.compiled_filters)
            )
            stages.append(pipe_stages.TextFilteringStage(
                name="Text Filtering", 
                preprocessor=preprocessor,
                config=config,
                save_ckpt=config.checkpoints.save_cleaned,
                save_cb=pipe_stages.SaveDataCallback(save_json, "Cleaned Data", save_ckpt_path=get_cache_file_path(config, "cleaned"))
            ))

        # Keyword Model Stage - #! should also be an optional step if we want to load data with keywords from disk
        # TODO: remove the preprocessor from the Keyword Models and ensure it gets its own stage

        stages.append(pipe_stages.KeywordExtractionStage(
            name="Keyword Extraction",
            model=KeyBertKeywordModel(model_name=config.model_name, candidate_labels=config.seed_kws, top_k=config.keyword_top_k),
            config=config,
            save_ckpt=config.checkpoints.save_final_output,
            save_cb=pipe_stages.SaveDataCallback(save_json, "Keyword Data", save_ckpt_path=get_cache_file_path(config, "keyword"))
        ))

        # TODO: flesh out the embedding and clustering steps in embedding.py before integrating it here

        return Pipeline(stages=stages)
