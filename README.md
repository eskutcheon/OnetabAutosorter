# Onetab Autosorter

A powerful Python-based pipeline for automatically organizing, clustering, and annotating bookmarks and tab exports (e.g. from OneTab or browser bookmarks). This project extracts and cleans link metadata, supplements it with web-scraped content, and applies advanced NLP-based keyword extraction, transformer-based embeddings, and hierarchical density-based clustering to sort the bookmarks into named categories.

---


## Key Features
- HTML & browser bookmark parsing (OneTab, Netscape/Chromium/Firefox exports)
- Web scraping with async/threaded strategies and domain-aware rate limiting
- Text preprocessing pipeline (domain + semantic filters)
  - Regex-based text filtering with a pluggable pattern registry
  - Language-aware domain filtering with boilerplate removal
- Zero-shot keyword extraction (KeyBERT & BERTopic)
- Multimodal embeddings combining text and metadata
- Flexible clustering with HDBSCAN or KMeans + zero-shot topic labeling
- Composable pipeline stages with caching and hash-based tracking
---

## Current Pipeline Architecture
1. Parse Input Bookmarks

Supports:
- `.html` (OneTab or browser exports)
- `.json` (previous pipeline output)

2. Scrape Supplemental Web Content
- Uses a configurable fetcher (limited, async, java, etc.)
- Java microservice (CURRENTLY NOT INTEGRATED) allows for fast batched scraping
- Integrated with WebScraper class with rate limiting and domain interleaving

3. Fit Domain Boilerplate Filter
- Detects and filters common phrases per domain (TF-IDF + rule-based)
- Filters via a Perceptron Tagger

4. Clean and Normalize Text

Applies pattern-based filtering:
- LaTeX, code, navigation text, etc
- Repeated phrases, formatting junk

5. Extract Keywords
- KeyBERT or BERTopic
- Optional candidate seed keyword labels from browser folders or command line (or YAML) input

6. Embed + Cluster (Optional / Modular)
- `polars`-based tabular DataFrames for efficient management
- SentenceTransformer embeddings + metadata
- Clustering via HDBSCAN or KMeans
- Keyword-based + zero-shot labeling (facebook/bart-large-mnli)

---

## CLI Usage

To get a description of available CLI arguments, run
```bash
python -m onetab_autosorter --help
```

The pipeline may be run as a module as in the previous example using a positional argument for the source bookmarks file:
```bash
python -m onetab_autosorter bookmarks.html
```
or using the `run_pipeline.py` file:
```bash
python run_pipeline.py bookmarks.html
```

#### Basic Pipeline Execution
```bash
python -m onetab_autosorter bookmarks.html \
  -o output/processed.json \
  --scraper_type limited \
  --keyword_model keybert \
  --top_k 10
```

#### Skip Scraping, Just Clean and Extract
If scraped data is unwanted:
```bash
python -m onetab_autosorter bookmarks.json --scraper_type none \
  --max_tokens 300 \
  --keyword_model bertopic
```

#### Using Existing Bookmark Folder Names as Candidate Labels
If your bookmarks are already grouped in folders (e.g., a Netscape/Firefox export):
```bash
python -m onetab_autosorter bookmarks.html --labels_from_html browser_folders.html --use_zero_shot_labels
```
This helps with topic labeling and zero-shot classification after clustering. Note that this option is treated separately from the positional input argument and labels may be seeded from totally unrelated bookmark folders.

#### Enable Checkpointing for Faster Dev Iterations
Enable caching for all stages:
```bash
python -m onetab_autosorter bookmarks.html --checkpoint_mode all
```
Or you can selectively cache specific stages:
```bash
python -m onetab_autosorter bookmarks.html --cache_stages scraped keywords cleaned
```

### Using a Full YAML Config
You can also consolidate all arguments in one file, e.g.
```yaml
# saved as my_run.yaml
input_file: bookmarks.html
output: output/final_output.json
scraper_type: limited
keyword_model: keybert
top_k: 10
checkpoint_mode: minimal
use_zero_shot_labels: true
labels_from_html: browser_bookmarks.html
```
Then you can run:
```bash
python -m onetab_autosorter --opts my_run.yaml
```

COMING SOON: a couple sample YAML files for ease of use

---


## Input Formats
- OneTab HTML export
- JSON intermediate files
- Native `bookmarks.html` from Chrome/Firefox (CURRENTLY UNTESTED)

---

## Configuration
All pipeline steps and filter sequences are defined in:
- config.py: CLI + YAML config
- patterns_registry.py: reusable regex patterns
- default_filter_order.yaml: user-editable filter order

---

## Modular Design
Each pipeline stage inherits from `PipelineStage`, defined in pipeline_stages.py:
- `ParsingStage`
- `WebScrapingStage`
- `DomainFilterFittingStage`
- `TextPreprocessingStage`
- `KeywordExtractionStage`
- `EmbeddingStage`
- `ClusteringStage`
- `LabelingStage` (coming soon)

Each stage:
- Can cache outputs
- Propagates pipeline objects for reuse
- Tracks arguments via hashing
- Easily replaced or re-ordered

---

## Planned Enhancements
I eventually plan to implement:
- (short term) more intelligent text cleaning with advanced tokenizer models
- (short term) Storing exportable keyword/topic sets for graph-building, bookmarking, or visualization
- Better folder-aware cluster labeling and bookmark re-export
- Security hardening and more adaptive scraping fallback logic

