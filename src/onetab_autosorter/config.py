import os
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    input_file: str
    output_json: str = r"output/cleaned_output.json"
    model_name: str = "all-MiniLM-L6-v2"
    keyword_top_k: int = 10
    deduplicate: bool = False
    dedupe_url_max_len: int = 200
    use_java_scraper: bool = False
    chunk_size: int = 30  # Number of entries to process in each chunk
    init_domain_filter: bool = False # whether to initialize the domain filter results from previous runs

    def __post_init__(self):
        # ensure the output path exists
        output_dir = os.path.dirname(self.output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # ensure chunk_size is a positive integer
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        print(f"Configuration initialized: {self}")


# TODO: considering moving this to the config.py file so they can reference the same constants
def get_cfg_from_cli():
    parser = argparse.ArgumentParser(description="Bookmark Clustering Pipeline")
    parser.add_argument("input_file", help="Input HTML or JSON file (depending on what you're attempting to parse)")
    parser.add_argument("-o", "--output", default=r"output/output_entries.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="KeyBERT model")
    parser.add_argument("--top_k", type=int, default=10, help="Top K keywords")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate URLs")
    parser.add_argument("--max_url_len", type=int, default=200, help="Max URL length to dedup")
    parser.add_argument("--use_java_scraper", action="store_true", help="Use Java-based scraper for HTML webcrawling (if available)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Number of entries to process in each chunk (for webcrawling)")
    parser.add_argument("--init_domain_filter", action="store_true", help = "Whether to initialize the domain filter with results from previous runs.")
    args = parser.parse_args()
    return Config(
        input_file=args.input_file,
        output_json=args.output,
        model_name=args.model,
        keyword_top_k=args.top_k,
        deduplicate=args.deduplicate,
        dedupe_url_max_len=args.max_url_len,
        use_java_scraper=args.use_java_scraper,
        chunk_size=args.chunk_size,
        init_domain_filter=args.init_domain_filter
    )

