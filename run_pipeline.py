import sys
import os

# Ensure src/ is in the import path
sys.path.insert(0, os.path.abspath("src"))
# TODO: make this use a common pipeline runner function later
#from onetab_autosorter.run import run_pipeline_with_keybert, run_pipeline_with_bertopic
from onetab_autosorter.config.config import Config, get_cfg_from_cli
from onetab_autosorter.pipelines.factory import create_pipeline #PipelineFactory
from pprint import pprint



def test_pipeline(config: Config):
    pipeline = create_pipeline(config)
    data = pipeline.run(config.input_file)
    #pprint(data[:10], indent=4)
    return data

if __name__ == "__main__":
    # parse args here or just call run_pipeline(...)
    config = get_cfg_from_cli()
    data = test_pipeline(config)

    # if config.scraper_type == "java":
    #     print("Using Java-based scraper for HTML webcrawling.")
    #     raise NotImplementedError("Java-based scraper not yet updated in latest version.")
    #     #run_pipeline(config, "java")
    # else:
    #     #run_pipeline_with_bertopic(config)
    #     run_pipeline_with_keybert(config)
