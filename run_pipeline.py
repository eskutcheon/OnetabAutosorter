
import sys
import os
# Ensure src/ is in the import path
sys.path.insert(0, os.path.abspath("src"))
from onetab_autosorter.config.config import Config
from onetab_autosorter.pipelines.factory import create_pipeline #PipelineFactory
# from pprint import pprint



def test_pipeline(config: Config):
    pipeline = create_pipeline(config)
    data = pipeline.run(config.input_file)
    #pprint(data[:10], indent=4)
    return data

if __name__ == "__main__":
    # parse args here or just call run_pipeline(...)
    from onetab_autosorter.config.cli import build_config_from_args
    config = build_config_from_args()
    data = test_pipeline(config)
