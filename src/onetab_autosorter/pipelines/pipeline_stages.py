import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Callable, Union, Any
# local imports
from onetab_autosorter.keyword_extraction import KeyBertKeywordModel
from onetab_autosorter.config.config import Config, CheckpointSettings #, get_cfg_from_cli
from onetab_autosorter.utils.utils import detect_bookmark_format, deduplicate_entries
from onetab_autosorter.preprocessors.handler import TextPreprocessingHandler
from onetab_autosorter.preprocessors.domain_filter import DomainBoilerplateFilter
from onetab_autosorter.preprocessors.text_filters import TextCleaningFilter

from onetab_autosorter.pipelines.staging_utils import load_entries, scrape_and_cache, merge_entries_with_scraped, load_scraped_entries
from onetab_autosorter.utils.io_utils import compute_hash, cache_path, get_cache_file, save_json, load_json





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
    def callback_creator(input_path: str, desc: str, *args, **kwargs) -> "SaveDataCallback":
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
    fn: Callable[..., Any]
    save_ckpt: bool = False
    #? adding a separate save function to act as a callback since making it its own stage makes less sense for a data "pipe"
    save_cb: SaveDataCallback = None

    def __post_init__(self):
        if self.save_ckpt and self.save_cb is None:
            raise ValueError("save_ckpt is True but no `SaveDataCallback` was provided.")

    def run(self, data: Any, *args, **kwargs) -> Any:
        print(f"Running stage: {self.name}")
        #? NOTE: not sure if I should just allow args and kwargs input to this function rather than
            #? saving them in the class and updating them if needed, which seems like more complexity than needed
        return self.fn(data, *args, **kwargs)

    def save_data(self, data: Any, *args, **kwargs):
        if self.save_cb:
            try:
                self.save_cb.save(data, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error saving data: {e}")



"""
create:
- WebScrapingStage
- TextFilteringStage
- KeywordExtractionStage
- EmbeddingStage
- ClusteringStage

"""

# TODO: refactor all stage subclasses to no longer save the entire config object, just the relevant parts for each stage
    # the PipelineFactory can still handle conditional logic from the CheckpointSettings object

#? NOTE: planned to remove the @dataclass decorator since subclasses of dataclass can get tricky with inheritance,
    #? but it should be fine since none of these actually use an explicit constructor

@dataclass
class LoadParsedEntriesStage(PipelineStage):
    config: Config

    def run(self, _):
        cache_file = get_cache_file(self.config.input_file, "parsed", self.config)
        if self.config.checkpoints.reuse_parsed and os.path.exists(cache_file):
            print("Loading parsed entries from cache.")
            return load_json(cache_file)
        entries = load_entries(self.config.input_file, self.config)
        if self.config.checkpoints.save_parsed:
            save_json(entries, cache_file)
        return entries


@dataclass
class WebScrapingStage(PipelineStage):
    fetcher_fn: Callable
    config: Config

    def run(self, entries: List[Dict]):
        urls = [entry["url"] for entry in entries]
        # TODO: replace
        scraped_results = scrape_and_cache(urls, self.fetcher_fn, self.config)
        for entry in entries:
            entry["raw_html"] = scraped_results.get(entry["url"], "")
        # save callback invoked conditionally (internally checks if save_ckpt is True):
        self.save_data(entries)
        return entries


@dataclass
class LoadScrapedDataStage(PipelineStage):
    config: Config

    def run(self, entries: List[Dict]):
        cache_dir = self.config.checkpoints.cache_dir
        #! FIXME: doesn't exist - figure out how to approach later
        loaded_scraped = load_scraped_entries(cache_dir)
        # Merge loaded scraped data with current entries based on URL
        merged_entries = merge_entries_with_scraped(entries, loaded_scraped)
        return merged_entries


@dataclass
class TextFilteringStage(PipelineStage):
    preprocessor: TextPreprocessingHandler
    config: Config

    def run(self, entries: List[Dict]):
        # Filter entries using the preprocessor
        #! FIXME: method doesn't exist - only process_text exists, assuming string input
        filtered_entries = self.preprocessor.filter_entries(entries)
        # save callback invoked conditionally (internally checks if save_ckpt is True):
        self.save_data(filtered_entries)
        return filtered_entries


@dataclass
class KeywordExtractionStage(PipelineStage):
    keyword_model: KeyBertKeywordModel
    config: Config

    def run(self, entries: List[Dict]):
        # Extract keywords using the KeyBERT model
        entries_with_keywords = self.keyword_model.run(entries)
        # save callback invoked conditionally (internally checks if save_ckpt is True):
        self.save_data(entries_with_keywords)
        return entries_with_keywords


@dataclass
class EmbeddingStage(PipelineStage):
    config: Config

    def run(self, entries: List[Dict]):
        # Placeholder for embedding logic
        # save callback invoked conditionally (internally checks if save_ckpt is True):
        self.save_data(entries)
        return entries



@dataclass
class ClusteringStage(PipelineStage):
    config: Config

    def run(self, entries: List[Dict]):
        # Placeholder for clustering logic
        # save callback invoked conditionally (internally checks if save_ckpt is True):
        self.save_data(entries)
        return entries