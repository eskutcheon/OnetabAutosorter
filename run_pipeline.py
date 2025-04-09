import sys
import os

# Ensure src/ is in the import path
sys.path.insert(0, os.path.abspath("src"))
# TODO: make this use a common pipeline runner function later
from src.onetab_autosorter.pipeline import run_pipeline, run_pipeline_with_bertopic, run_pipeline_with_keybert
from onetab_autosorter.config.config import get_cfg_from_cli


if __name__ == "__main__":
    # parse args here or just call run_pipeline(...)
    config = get_cfg_from_cli()
    if config.use_java_scraper:
        print("Using Java-based scraper for HTML webcrawling.")
        run_pipeline(config, "java")
    else:
        #run_pipeline_with_bertopic(config)
        run_pipeline_with_keybert(config)
