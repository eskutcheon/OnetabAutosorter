import os
from dataclasses import dataclass, field
from re import Pattern
from typing import Optional, Dict, List, Literal, Callable, Union, Any, Tuple
# local imports
from onetab_autosorter.config.config import Config, CheckpointSettings #, get_cfg_from_cli
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter

from onetab_autosorter.utils.io_utils import compute_hash, get_hashed_path_from_string, save_json, load_json, get_hashed_path_from_hash


#? NOTE: a lot of the skeleton for this file was written by Copilot - still need to de-shittify some parts


@dataclass
class SaveDataCallback:
    """ Callback for saving data at different stages of the pipeline """
    save_fn: Callable[..., Any]
    data_description: str = ""
    save_ckpt_path: Optional[str] = None
    #? It makes more sense to keep args and kwargs here since they'll likely be set at instantiation
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def save(self, data: Any):
        print(f"Saving {self.data_description} to {self.save_ckpt_path}...")
        self.save_fn(data, self.save_ckpt_path, *self.args, **self.kwargs)

    def __repr__(self):
        return f"SaveDataCallback({self.data_description}, {self.save_ckpt_path})"

    @staticmethod
    def factory(input_path: str, desc: str, *args, **kwargs) -> "SaveDataCallback":
        """ static dispatcher method for creating the callback class based on the input path """
        # might end up making a separate dispatcher function that returns the function handle, then this just becomes 2 lines
        file_ext = os.path.splitext(input_path)[1]
        if file_ext == ".json":
            save_fn = save_json
        # planning for HTML, CSV, and other formats in the future
        else:
            raise NotImplementedError(f"File type {file_ext} not yet supported for saving.")
        return SaveDataCallback(save_fn, desc, input_path, *args, **kwargs)



@dataclass
class PipelineStage:
    name: str
    run_fn: Callable
    cache_dir: str
    reuse_cache: bool
    save_cache: bool
    cache_file_path: Optional[str] = None
    # queue up args for the Pipeline object to pass to the next stage
    to_propagate: Optional[Tuple[List, Dict]] = field(default_factory=lambda: ([], {}))
    # save_ckpt: bool = False
    # #? adding a separate save function to act as a callback since making it its own stage makes less sense for a data "pipe"
    # save_cb: SaveDataCallback = None


    def run(self, data: Any, *args, **kwargs) -> Any:
        #self.set_cache_file(cache_id, data)
        if self.reuse_cache:
            cached_data = self.load_from_cache()
            if cached_data:
                #! may cause issues if it's coming before setting the propagated args in the run_fn
                return cached_data, self.to_propagate
        print(f"Running stage: {self.name}")
        #? NOTE: currently have it set so that self.to_propagate is set within the set self.run_fn
        data = self.run_fn(data, *args, **kwargs)
        #? NOTE: this will never run after loading from cache, so shouldn't see unnecessary saving
        if self.save_cache:
            self.save_data(data)
        return data, self.to_propagate

    def set_cache_file(self, cache_id: Optional[str] = None, data: Optional[Any] = None):
        if cache_id:
            self.cache_file_path = get_hashed_path_from_hash(cache_id, self.name, self.cache_dir)
        elif data:
            url_str = ",".join([entry["url"] for entry in data])
            self.cache_file_path = get_hashed_path_from_string(url_str, self.name, self.cache_dir)
        else:
            raise ValueError("Either cache_id or data must be provided to set the cache file path.")

    def load_from_cache(self) -> Optional[Any]:
        if os.path.exists(self.cache_file_path):
            print(f"[{self.name}] Loading cached data.")
            return load_json(self.cache_file_path)
        return None

    def save_data(self, data: Any):
        save_json(self.cache_file_path, data)
        print(f"[{self.name}] Data cached.")

    def _set_propagation(self, *args, **kwargs):
        self.to_propagate = (args, kwargs)



class Pipeline:
    def __init__(self, stages: List[PipelineStage], loader_stage: PipelineStage = None):
        self.stages = stages
        self.loader_stage = loader_stage
        self.cache_hash = None

    def load_initial_data(self, initial_input) -> List[Dict[str, Any]]:
        if self.loader_stage:
            data, _ = self.loader_stage.run(initial_input)
        if not data:
            raise ValueError("No data loaded from the loader stage.")
        if self.cache_hash is None:
            self.set_cache_hash(data)
        return data

    def set_cache_hash(self, data: Any):
        url_str = ",".join([entry["url"] for entry in data])
        self.cache_hash = compute_hash(url_str)


    def run(self, initial_input: Any) -> Any:
        #! should be temporary while debugging the rest of the pipeline
        data = self.load_initial_data(initial_input)  # load the initial data from the loader stage
        #data = initial_input # may be a path initially
        feedforward_args: List[Tuple[List, Dict]] = []
        for stage in self.stages:
            stage.set_cache_file(self.cache_hash, data)
            # if stage.reuse_cache:
            #     cached_data = stage.load_from_cache()
            #     if cached_data:
            #         data = cached_data
            #         continue
            # try to pop the last set of args and kwargs for the current stage and default to empty if not available
            try:
                args, kwargs = feedforward_args.pop()
            except IndexError: # if feedforward_args is empty, set args and kwargs to relevant empty data structures
                args, kwargs = [], {}
            except Exception as e: # raise any other exceptions that may occur
                raise RuntimeError(f"Error while processing stage {stage.name}: {e}")
            # TODO: think I need a better way to handle the cache hashing since it'd be better to use the initial data
                #todo: should be a lot easier with overridden run functions in the subclasses
            data, next_args = stage.run(data, *args, **kwargs)  # run the stage with the data and any additional arguments
            feedforward_args.append(next_args)  # collect additional arguments for the next stage (empty by default)
            # if self.cache_hash is None:
            #     self.set_cache_hash(data)
            # save the data by calling a dispatcher function that calls the appropriate saving function based on the data type
                # it should take a variable number of positional arguments for potentially multiple data
            ###some_central_saving_function(data, output_path=stage.save_ckpt_path)  # replace with actual saving logic
        return data



"""
create:
- WebScrapingStage
- TextFilteringStage
- KeywordExtractionStage
- EmbeddingStage
- ClusteringStage
"""

# I could make all of these simple functions for creating each stage once I write a better way to handle propagating extra arguments
    # only thing I can think to do with the current setup is adding a "cls" argument to all run_fn and calling cls.to_propagate() in the run function

class ParsingStage(PipelineStage):
    def __init__(self, file_path: str, deduplicate: bool, checkpoints: CheckpointSettings):
        from onetab_autosorter.pipelines.staging_utils import run_parser, create_parser
        super().__init__(
            name="parsed",
            # TODO: I should rewrite some parsing logic so that I'm not passing the file path twice
            run_fn=lambda _: run_parser(create_parser(file_path), file_path, deduplicate),
            cache_dir=checkpoints.cache_dir,
            reuse_cache=checkpoints.reuse_parsed,
            save_cache=checkpoints.save_parsed
        )

class WebScrapingStage(PipelineStage):
    def __init__(self, scraper_type: str, checkpoints: CheckpointSettings):
        from onetab_autosorter.pipelines.staging_utils import run_webscraping, create_fetcher, merge_entries_with_scraped
        #!!! What I'm doing right now is fucking stupid tbh
        def _run_webscraping(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            fetcher_fn = create_fetcher(scraper_type)
            self._set_propagation(fetcher_fn=fetcher_fn)
            urls = [entry["url"] for entry in data]
            # TODO: need to normalize outputs of run_webscraping for consistent return formats - I think only the Java version is outdated
            scraped = run_webscraping(fetcher_fn, urls)
            return merge_entries_with_scraped(data, scraped) #? NOTE: outputs entries with new "scraped" key
        # as lambda data: merge_entries_with_scraped(data, run_webscraping(create_fetcher(scraper_type), [e["url"] for e in data]))
        # instantiate the stage with the run function that will be called later
        super().__init__(
            name="scraped",
            run_fn = _run_webscraping, # TODO: think I'm just going to rewrite the base class `run` and stop passing run_fn and just call an overridden `self.run`
            cache_dir=checkpoints.cache_dir,
            reuse_cache=checkpoints.reuse_scraped,
            save_cache=checkpoints.save_scraped
        )


#!!! RETHINK ORDER - old order expected domain filter initialization and fitting before bulk webscraping
    # changing a line in DomainBoilerplateFilter to query the new keys for now - after update, a fetcher function wouldn't be needed
class DomainFilterFittingStage(PipelineStage):
    #? NOTE: DomainBoilerplateFilter takes a lot of additional arguments that should be accounted for
    def __init__(self, checkpoints: CheckpointSettings, **kwargs):
        from onetab_autosorter.pipelines.staging_utils import create_and_fit_domain_filter, get_domain_mapped_urls
        # TODO: may want to include a mechanism for propagating stuff (in this case, the domain filter to the handler) to the next stage
                # passing a local function to the run function just like the lambdas, but this should be more readable
        def _run_domain_filter_fitting(data: List[Dict[str, Any]], fetcher_fn: Callable = None) -> List[Dict[str, Any]]:
            #if not fetcher_fn:
            domain_filter = create_and_fit_domain_filter(
                load_from_file = not checkpoints.override_boilerplate,
                domain_map = get_domain_mapped_urls(data),
                #! TEMPORARY - adding to config later
                json_path = r"output/domain_boilerplate.json",
                # TODO: after a rewrite of the domain filtering approach, fetcher_fn may not be necessary - might just make two different entry points though
                scraper_fn = fetcher_fn if fetcher_fn else None, # just to replace the empty argument for the lambda
                **kwargs
            )
            self._set_propagation(domain_filter=domain_filter)
            return data #, domain_filter
        # instantiate the stage with the run function that will be called later
        super().__init__(
            name="domain_filter_fitting",
            run_fn = _run_domain_filter_fitting,
            cache_dir=checkpoints.cache_dir,
            reuse_cache = not checkpoints.override_boilerplate,
            save_cache=True
        )



class TextPreprocessingStage(PipelineStage):
    def __init__(self, compiled_filters: List[Pattern], checkpoints: CheckpointSettings, max_tokens: int = 200):
        from onetab_autosorter.pipelines.staging_utils import  create_preprocessor, create_text_cleaning_filter
        # instantiate the stage with the run function that will be called later
        def _run_handler(data: List[Dict[str, Any]], domain_filter: DomainBoilerplateFilter) -> TextPreprocessingHandler:
            cleaning_filter = create_text_cleaning_filter(ignore_patterns=compiled_filters)
            preprocessor = create_preprocessor(domain_filter, cleaning_filter, max_tokens=max_tokens)
            # shouldn't need to propagate anything here since keyword filtering is independent of the raw text filters
            return preprocessor.process_entries(data)
        super().__init__(
            name="text_cleaning_filter",
            run_fn = _run_handler,
            cache_dir=checkpoints.cache_dir,
            reuse_cache=checkpoints.reuse_cleaned,
            # TODO: might make this the save callback function later - override with whatever while keeping the default caching
            save_cache=checkpoints.save_cleaned
        )



class KeywordExtractionStage(PipelineStage):
    def __init__(
        self,
        checkpoints: CheckpointSettings,
        model_type: Literal["keybert", "bertopic"],
        backbone_model: str = "all-MiniLM-L6-v2",
        seed_kws: List[str] = [],
        top_k: int = 10,
        #? NOTE: almost certainly need more arguments to instantiate the keyword models for more robust extraction
    ):
        from onetab_autosorter.pipelines.staging_utils import create_keyword_model
        # instantiate the stage with the run function that will be called later
        def _run_keyword_model(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            keyword_model = create_keyword_model(model_type, backbone_model, seed_kws=seed_kws, top_k=top_k)
            #self.to_propagate(keyword_model=keyword_model)
            return keyword_model.run(data)
        super().__init__(
            name="keyword_extraction",
            run_fn = _run_keyword_model,
            cache_dir=checkpoints.cache_dir,
            reuse_cache=checkpoints.reuse_keywords,
            save_cache=checkpoints.save_keywords
        )







# TODO: refactor all stage subclasses to no longer save the entire config object, just the relevant parts for each stage
    # the PipelineFactory can still handle conditional logic from the CheckpointSettings object

#? NOTE: planned to remove the @dataclass decorator since subclasses of dataclass can get tricky with inheritance,
    #? but it should be fine since none of these actually use an explicit constructor


# TODO: while I'm not really making any assignments to the Config instance, it would still be safest to pass deepcopies to avoid any accidental mutations
