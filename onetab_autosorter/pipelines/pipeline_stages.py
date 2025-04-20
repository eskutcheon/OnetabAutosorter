import os
#~ may make a protocol to annotate something as a dataclass just for clarity later
    #~ use typing.Protocol: https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
from dataclasses import dataclass, field, asdict
from re import Pattern
from typing import Optional, Dict, List, Literal, Callable, Union, Any, Tuple, Generator
# local imports
# TODO: need to make this a more intelligent config managed by `Pipeline` so that we can start from any stage by loading data
from onetab_autosorter.config.config import StageCacheSettings
# TODO: replace with imports from new types file later - only used for type annotation at the moment
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.utils.utils import compute_hash, get_hashed_path_from_string, save_json, load_json, get_hashed_path_from_hash


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
    #run_fn: Callable
    cache_dir: str
    reuse_cache: bool
    save_cache: bool
    hash_based_on_data: bool = False
    cache_file_path: Optional[str] = None
    # #? adding a separate save function to act as a callback since making it its own stage makes less sense for a data "pipe"
    # save_cb: SaveDataCallback = None


    def run(self, data: Any, *args, **kwargs) -> Tuple[Any, Tuple[List, Dict]]:
        """ Template method that orchestrates the execution flow """
        # Set up cache file path if not already set
        if not self.cache_file_path and hasattr(data, "__iter__") and not isinstance(data, str):
            self.set_cache_file(None, data)
        # Try loading from cache
        cached_data = self.load_from_cache() if self.reuse_cache else None
        # Process the data - either from cache or by running the actual processing
        if cached_data is not None:
            print(f"[{self.name}] Using cached data")
            result_data = cached_data
            # Even when using cached data, we need to create objects for propagation
            self.create_stage_objects(result_data, *args, **kwargs)
        else:
            print(f"[{self.name}] Processing data")
            # Prepare any objects needed by this stage
            self.create_stage_objects(data, *args, **kwargs)
            # Process the actual data
            result_data = self.process_data(data, *args, **kwargs)
            # Save to cache if needed
            if self.save_cache and self.cache_file_path:
                self.save_data(result_data)
        # get objects to propagate to the next stage
        propagation = self.get_propagation_objects()
        return result_data, propagation


    def create_stage_objects(self, data: Any, *args, **kwargs) -> None:
        """ Create any objects needed by this stage. Subclasses should override """
        raise NotImplementedError("Subclasses must implement create_stage_objects")

    def process_data(self, data: Any, *args, **kwargs) -> Any:
        """ Process the data. Subclasses must override """
        raise NotImplementedError("Subclasses must implement process_data")

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        """ Get objects to propagate to the next stage. Subclasses should override if needed """
        return [], {}

    def set_cache_file(self, cache_id: Optional[str] = None, data: Optional[Any] = None):
        """ Set the cache file path based on the data or a given ID """
        if cache_id:
            self.cache_file_path = get_hashed_path_from_hash(cache_id, self.name, self.cache_dir)
        elif data:
            url_str = ",".join([entry.get("url", "") for entry in data if isinstance(entry, dict)])
            self.cache_file_path = get_hashed_path_from_string(url_str, self.name, self.cache_dir)
        else:
            raise ValueError("Either cache_id or data must be provided to set the cache file path.")

    def load_from_cache(self) -> Optional[Any]:
        """ Load data from cache """
        if self.cache_file_path and os.path.exists(self.cache_file_path):
            print(f"[{self.name}] Loading cached data.")
            return load_json(self.cache_file_path)
        return None

    def save_data(self, data: Any):
        """ Save data to cache """
        save_json(self.cache_file_path, data)
        print(f"[{self.name}] Data cached.")



# TODO: decide where to move this utility function later - only used in Pipeline for now
def aggregate_settings_to_dict(settings_metadata: List[object]) -> Dict[str, Any]:
    """ Aggregate settings metadata from dataclass objects into a dictionary """
    if not isinstance(settings_metadata, (list, tuple)):
        settings_metadata = [settings_metadata]
    aggregated_metadata = {}
    for cfg_obj in settings_metadata:
        try:
            aggregated_metadata.update(asdict(cfg_obj))
        except TypeError:
            raise TypeError("Settings metadata must be a dataclass or a dictionary.")
        except Exception as e:
            raise RuntimeError(f"Error while processing settings metadata: {e}") from e
    return aggregated_metadata


class Pipeline:
    def __init__(self, stages: List[PipelineStage], loader_stage: PipelineStage = None, settings_metadata: List[object] = []):
        self.stages = stages
        self.loader_stage = loader_stage
        self.argument_stack = []  # stack of arguments and keyword arguments to be propagated
        # populate a dict of metadata aggregated from class objects passed in settings_metadata for more robust hashing
        self.settings_metadata = aggregate_settings_to_dict(settings_metadata)
        self.data_only_hash = None
        self.cache_hash = None

    #& UNTESTED
    def __iter__(self, initial_input: Any) -> Generator[Any, None, None]:
        """ iterable to generate data in each stage while holding off on the next stage until requested """
        data = self.load_initial_data(initial_input)  # load the initial data from the loader stage
        for stage in self.stages:
            data = self.run_stage(stage, data)
            if not self.cache_hash: # data should be populated after at least the first non-loader stage
                self.set_cache_hash(data)
            yield data

    def run(self, initial_input: Any) -> Any:
        data = self.load_initial_data(initial_input)  # load the initial data from the loader stage
        # feedforward_args: List[Tuple[List, Dict]] = []
        for stage in self.stages:
            data = self.run_stage(stage, data)  # run each stage with the data and any additional arguments
            # save the data by calling a dispatcher function that calls the appropriate saving function based on the data type
                # it should take a variable number of positional arguments for potentially multiple data
            if not self.cache_hash: # data should be populated after at least the first non-loader stage
                self.set_cache_hash(data)
        return data

    def run_stage(self, stage: PipelineStage, data: Any) -> Tuple[Any, Tuple[List, Dict]]:
        """ Run a specific stage of the pipeline with the given data and arguments.
            Args:
                stage (PipelineStage): The stage to run.
                data (Any): The input data for the stage.
                *args: Additional positional arguments for the stage.
                **kwargs: Additional keyword arguments for the stage.
            Returns:
                Tuple[Any, Tuple[List, Dict]]: The processed data and objects to propagate to the next stage.
        """
        if self.cache_hash:
            # if the stage is only data-dependent, don't use setting metadata to compute the hash (ensures no recomputation when not needed)
            hash = self.data_only_hash if stage.hash_based_on_data else self.cache_hash
            stage.set_cache_file(hash, data)
        # try to pop the last set of args and kwargs for the current stage and default to empty if not available
        args, kwargs = self.pop_argument_stack(stage.name)
        # run the stage with the data and any additional arguments from the argument stack
        data, next_args = stage.run(data, *args, **kwargs)
        self.argument_stack.append(next_args)  # collect additional arguments for the next stage (empty by default)
        return data

    def load_initial_data(self, initial_input) -> List[Dict[str, Any]]:
        if not self.loader_stage:
            #? NOTE: should be temporary but this is how I'm handling the initial loading phase of the pipeline for now
            raise RuntimeError("No loader stage defined for the pipeline.")
        data, _ = self.loader_stage.run(initial_input)
        if not data:
            raise ValueError("No data loaded from the loader stage.")
        if self.cache_hash is None:
            self.set_cache_hash(data)
        return data
        #return initial_input

    def set_cache_hash(self, data: Any):
        # TODO: make this condition a little more robust to different types and data formats
        if isinstance(data, list) and data and isinstance(data[0], dict) and "url" in data[0]:
            data_only_hash = ",".join([entry["url"] for entry in data])
            # if the stage is only data-dependent, don't use setting metadata to compute the hash (ensures no recomputation when not needed)
            if self.settings_metadata:
                metadata_str = ",".join([f"{k}={v}" for k, v in self.settings_metadata.items() if v])
                metadata_str = ",".join(metadata_str) + data_only_hash
                self.cache_hash = compute_hash(metadata_str)
            else:
                self.cache_hash = compute_hash(data_only_hash)
            self.data_only_hash = compute_hash(data_only_hash)
        else:
            raise ValueError("Data must be a list of dictionaries with 'url' keys to compute the cache hash.")

    def pop_argument_stack(self, stage_name: Optional[str] = "_") -> Tuple[List, Dict]:
        """ Pop the last set of arguments and keyword arguments from the feedforward_args stack.
            Args:
                feedforward_args (List[Tuple[List, Dict]]): A stack of arguments and keyword arguments to be propagated.
            Returns:
                Tuple[List, Dict]: A tuple containing the popped arguments (list) and keyword arguments (dict).
            Raises:
                RuntimeError: If an error occurs while processing the stack.
        """
        try:
            args, kwargs = self.argument_stack.pop()
        except IndexError: # if feedforward_args is empty, set args and kwargs to corresponding empty data structures
            args, kwargs = [], {}
        except (TypeError, ValueError) as e:  # handle specific exceptions
            raise RuntimeError(f"Error while processing stage {stage_name}: {e}") from e
        return args, kwargs


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
    def __init__(self, file_path: str, stage_settings: StageCacheSettings):
        self.file_path = file_path
        self.parser = None
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=stage_settings.load_cache,
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def create_stage_objects(self, data: Any, *args, **kwargs) -> None:
        from onetab_autosorter.pipelines.staging_utils import create_parser
        # TODO: I should rewrite some parsing logic so that I'm not passing the file path twice
            #? NOTE: there needs to be some checks when attempting to reuse the parser for other files that it's the right file type
        self.parser = create_parser(self.file_path)

    def process_data(self, data: Any, *args, **kwargs) -> List[Dict[str, Any]]:
        from onetab_autosorter.pipelines.staging_utils import run_parser
        return run_parser(self.parser, self.file_path)


class WebScrapingStage(PipelineStage):
    def __init__(self, scraper_type: str, stage_settings: StageCacheSettings):
        self.scraper_type = scraper_type
        self.fetcher_fn = None
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=stage_settings.load_cache,
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def create_stage_objects(self, data: Any, *args, **kwargs) -> None:
        from onetab_autosorter.pipelines.staging_utils import create_fetcher
        self.fetcher_fn = create_fetcher(self.scraper_type)

    def process_data(self, data: List[Dict[str, Any]], *args, **kwargs) -> List[Dict[str, Any]]:
        from onetab_autosorter.pipelines.staging_utils import run_webscraping, merge_entries_with_scraped
        urls = [entry["url"] for entry in data]
        # TODO: need to normalize outputs of run_webscraping for consistent return formats - I think only the Java version is outdated
        scraped = run_webscraping(self.fetcher_fn, urls)
        return merge_entries_with_scraped(data, scraped)

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        return [], {"fetcher_fn": self.fetcher_fn}



class DomainFilterFittingStage(PipelineStage):
    def __init__(self, stage_settings: StageCacheSettings, **kwargs):
        self.domain_filter = None
        self.filter_kwargs = kwargs
        self.json_path = r"output/domain_boilerplate.json"  #! TEMPORARY - add to config later
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=False, # stage_settings.load_cache, #! still haven't addressed loading the domain boilerplate filter from file yet
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def create_stage_objects(self, data: List[Dict[str, Any]], fetcher_fn: Callable = None, *args, **kwargs) -> None:
        from onetab_autosorter.pipelines.staging_utils import create_domain_filter
        #? NOTE: DomainBoilerplateFilter takes a lot of additional arguments that should be accounted for
        # if we're loading from cache, create the domain filter from the saved file
        if self.reuse_cache:
            #! FIXME: need to remove the saving in create_and_fit_domain_filter while allowing loading the object
            self.domain_filter = create_domain_filter(load_from_file=True, json_path=self.json_path, **self.filter_kwargs)
        else:
            # otherwise create and fit it with the current data
            self.domain_filter = create_domain_filter(load_from_file=False, **self.filter_kwargs)

    def process_data(self, data: List[Dict[str, Any]], fetcher_fn: Callable = None, *args, **kwargs) -> List[Dict[str, Any]]:
        from onetab_autosorter.pipelines.staging_utils import get_domain_mapped_urls
        # Run the full fitting process
        domain_map = get_domain_mapped_urls(data)
        MIN_REPEAT_COUNT = 3
        #? NOTE: after a rewrite of the domain filtering approach, fetcher_fn may not be necessary - might just make two different entry points though
        self.domain_filter.run_preliminary_search(domain_map, MIN_REPEAT_COUNT, fetcher_fn)
        # Save the domain filter if requested
        if self.save_cache:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            self.domain_filter.save_boilerplate_map(self.json_path)
        return data

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        return [], {"domain_filter": self.domain_filter}


class TextPreprocessingStage(PipelineStage):
    def __init__(self, compiled_filters: List[Pattern], stage_settings: StageCacheSettings, max_tokens: int = 200):
        self.compiled_filters = compiled_filters
        self.max_tokens = max_tokens
        self.cleaning_filter = None
        self.preprocessor = None
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=stage_settings.load_cache,
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def create_stage_objects(self, data: List[Dict[str, Any]], domain_filter: DomainBoilerplateFilter = None, *args, **kwargs) -> None:
        from onetab_autosorter.pipelines.staging_utils import create_text_cleaning_filter, create_preprocessor
        self.cleaning_filter = create_text_cleaning_filter(ignore_patterns=self.compiled_filters)
        self.preprocessor = create_preprocessor(domain_filter, self.cleaning_filter, max_tokens=self.max_tokens)

    def process_data(self, data: List[Dict[str, Any]], domain_filter: DomainBoilerplateFilter = None, *args, **kwargs) -> List[Dict[str, Any]]:
        return self.preprocessor.process_entries(data)

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        # shouldn't need to propagate anything here since keyword filtering is independent of the raw text filters, but that could change
        return [], {} #{"preprocessor": self.preprocessor}


class KeywordExtractionStage(PipelineStage):
    def __init__(
        self,
        stage_settings: StageCacheSettings,
        model_type: Literal["keybert", "bertopic"],
        backbone_model: str = "all-MiniLM-L6-v2",
        seed_kws: List[str] = [],
        top_k: int = 10,
        #? NOTE: almost certainly need more arguments to instantiate the keyword models for more robust extraction
    ):
        self.model_type = model_type
        self.backbone_model = backbone_model
        self.seed_kws = seed_kws or []
        self.top_k = top_k
        self.keyword_model = None
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=stage_settings.load_cache,
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def create_stage_objects(self, data: List[Dict[str, Any]], preprocessor = None, *args, **kwargs) -> None:
        from onetab_autosorter.pipelines.staging_utils import create_keyword_model
        self.keyword_model = create_keyword_model(
            self.model_type,
            self.backbone_model,
            seed_kws=self.seed_kws,
            top_k=self.top_k
        )

    def process_data(self, data: List[Dict[str, Any]], preprocessor = None, *args, **kwargs) -> List[Dict[str, Any]]:
        return self.keyword_model.run(data)

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        # shouldn't need to propagate anything here since keyword filtering should be independent of embedding, but that could change
        return [], {"keyword_model": self.keyword_model}



class EmbeddingStage(PipelineStage):
    """ Generate embeddings for bookmark entries' data to cluster in the next stage """
    def __init__(
        self,
        stage_settings: StageCacheSettings,
        embedding_model: str = "all-MiniLM-L6-v2",
        include_features: List[str] = None,
        batch_size: int = 32,
        normalize_numeric: bool = True
    ):
        self.embedding_model = embedding_model
        self.include_features = include_features or ["keyword", "path", "subdomain", "date"]
        self.batch_size = batch_size
        self.normalize_numeric = normalize_numeric
        self.results = None
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=stage_settings.load_cache,
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def process_data(self, data: List[Dict[str, Any]], keyword_model = None, *args, **kwargs) -> Dict[str, Any]:
        """ Generate embeddings using the EmbeddedDataFrameBuilder. """
        from onetab_autosorter.models.embeddings import EmbeddedDataFrameBuilder
        # Get the keyword field name from the keyword model if available
        keyword_field = "keywords"
        if keyword_model:
            keyword_field = getattr(keyword_model, "output_field", "keywords")
        # Use the factory method to generate embeddings in one step
        self.results = EmbeddedDataFrameBuilder.dataframe_factory(
            entries=data,
            embedding_model=self.embedding_model,
            keyword_field=keyword_field,
            include_features=self.include_features,
            batch_size=self.batch_size,
            normalize_numeric=self.normalize_numeric,
            include_embeddings=True,
            verbose=True
        )
        return asdict(self.results) # dictionary with dataframe, embeddings (ndarray), and dictionary of feature info

    def create_stage_objects(self, data, *args, **kwargs):
        pass

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        """ Pass the embedding results to the next stage. """
        if not self.results:
            return [], {}
        return [], {"embedding_results": asdict(self.results)}



class ClusteringStage(PipelineStage):
    """ Cluster bookmark entries using embeddings """
    def __init__(
        self,
        stage_settings: StageCacheSettings,
        algorithm: str = "hdbscan",
        min_cluster_size: int = 5,
        add_keyword_labels: bool = True,
        add_zero_shot_labels: bool = False,
        candidate_labels: List[str] = None
    ):
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.add_keyword_labels = add_keyword_labels
        self.add_zero_shot_labels = add_zero_shot_labels
        self.candidate_labels = candidate_labels
        self.clustering_results = None
        super().__init__(
            name=stage_settings.stage_name,
            cache_dir=stage_settings.cache_dir,
            reuse_cache=stage_settings.load_cache,
            save_cache=stage_settings.save_cache,
            hash_based_on_data=stage_settings.only_data_dependent
        )

    def process_data(self, data: Any, embedding_results = None, *args, **kwargs) -> Dict[str, Any]:
        """ Apply clustering to embeddings """
        from onetab_autosorter.models.clustering import ClusteringBuilder
        if not embedding_results:
            raise ValueError("No embedding results provided to clustering stage")
        df = embedding_results["df"]
        embeddings = embedding_results["embeddings"]
        if embeddings is None:
            raise ValueError("No embeddings found in embedding results")
        # Use the factory method to perform clustering in one step
        self.clustering_results = ClusteringBuilder.cluster_factory(
            df=df,
            embeddings=embeddings,
            algorithm=self.algorithm,
            min_cluster_size=self.min_cluster_size,
            add_keyword_labels=self.add_keyword_labels,
            add_zero_shot_labels=self.add_zero_shot_labels,
            candidate_labels=self.candidate_labels,
            verbose=True
        )
        cluster_results = self.clustering_results
        return {
            "df": cluster_results.df,
            "clusters": cluster_results.clusters,
            # "cluster_counts": cluster_results.cluster_counts,
            # "zero_shot_counts": cluster_results.zero_shot_counts,
            # "outliers": cluster_results.outliers,
            # "model": cluster_results.model
        }

    def create_stage_objects(self, data, *args, **kwargs):
        pass

    def get_propagation_objects(self) -> Tuple[List, Dict]:
        """ Pass clustering results to the next stage (if any) """
        if not self.clustering_results:
            return [], {}
        return [], {"clustering_results": {k: v for k, v in asdict(self.clustering_results).items() if k in ["df", "clusters"]}}