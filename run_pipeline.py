import sys
import os

# Ensure src/ is in the import path
sys.path.insert(0, os.path.abspath("src"))
# TODO: make this use a common pipeline runner function later
from src.onetab_autosorter.pipeline import run_pipeline, run_pipeline_with_scraper, run_pipeline_with_domain_filter, run_pipeline_with_bertopic
from src.onetab_autosorter.config import get_cfg_from_cli

# FETCHER_REGISTRY = {
#     "default": default_html_fetcher,
#     "java": fetch_summary,
#     # "gpt": fetch_gpt_summary (future),
# }


if __name__ == "__main__":
    # parse args here or just call run_pipeline(...)
    config = get_cfg_from_cli()
    if config.use_java_scraper:
        print("Using Java-based scraper for HTML webcrawling.")
        run_pipeline_with_scraper(config)
    else:
        # TODO: figure out a more natural way to incorporate domain filtering into existing pipelines - REQUIRES MAJOR REWRITES
        #run_pipeline(config)
        #run_pipeline_with_domain_filter(config)
        run_pipeline_with_bertopic(config)
