# -*- coding: utf-8 -*-
from collections import Counter
from functools import wraps
from tqdm.auto import tqdm
from typing import Optional, Dict, List, Literal, Callable, Union, Any
import numpy as np
import torch
import polars as pl
import torch
from torch.utils.data import DataLoader
# from sklearn.metrics.pairwise import cosine_distances
from transformers import pipeline
# local imports
from onetab_autosorter.models.cluster_utils import BaseClusterer, KMeansClusterer, HDBSCANClusterer, ClusteringResults


# 67 options for candidate labels for testing zero-shot classification
DEFAULT_ZERO_SHOT_LABELS = [
    'robotics', 'gardening', 'art', 'research papers', 'job search', 'utilities', 'misc', 'travel',
    'health', 'piracy', 'recipes', 'data science', 'privacy', 'modding', 'local businesses',
    'music', 'sports', 'hobbies', 'photography', 'web development', 'natural science', 'education',
    'social media', 'self-improvement', 'fitness', 'movies', "hobbies", "social issues", 'reading',
    'lifestyle', 'food', 'artificial intelligence', 'culture', 'philosophy', "nsfw", "religion",
    'television', 'manga', 'anime', 'finance', 'social sciences', 'computer vision', 'academia',
    'books', 'nature', 'cooking', 'tech', 'sexuality', 'software', 'reference', 'coding', 'writing',
    'drugs', 'projects', 'aesthetics', 'math', 'news', 'environmentalism', 'politics', 'gaming',
    'current events', 'consumerism', 'history', 'fashion', 'articles', 'entertainment', 'comics',
]



def clustering_step(step_name: str):
    """ decorator to handle common patterns in clustering steps matching the embedding builder class """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(cls: 'ClusteringBuilder', *args, **kwargs):
            if cls.verbose:
                print(f"Performing {step_name}...")
            try:
                result = func(cls, *args, **kwargs)
                return result
            except Exception as e:
                if cls.verbose:
                    print(f"Error in {step_name}: {e}")
                raise
        return wrapper
    return decorator


class ClusteringBuilder:
    """ Builder class for clustering bookmark embeddings
        1. Applies clustering algorithms to group similar bookmarks
        2. Adds cluster assignments to the original DataFrame
        3. (Optionally) adds labeling steps
    """

    def __init__(self, algorithm: str = "hdbscan", batch_size: int = 16, verbose: bool = False):
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.verbose = verbose
        self.df: Optional[pl.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_model: Optional[BaseClusterer] = None
        # TODO: might wanna change labels and scores back to local variables
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_scores: Optional[np.ndarray] = None
        self.results: Optional[ClusteringResults] = None

    @clustering_step("clustering")
    def fit(self, df: pl.DataFrame, embeddings: np.ndarray, **kwargs) -> "ClusteringBuilder":
        """ Fit the selected clustering model to the embeddings
            Args:
                df: original DataFrame from EmbeddedDataFrameBuilder
                embeddings: embedding vectors from EmbeddedDataFrameBuilder
                **kwargs: additional parameters for the clustering algorithm
            Returns:
                self: for method chaining
        """
        self.df = df
        self.embeddings = embeddings
        self.run_cluster_fitting(embeddings, **kwargs)
        # add cluster results to DataFrame
        df_with_clusters = self.df.with_columns([
            pl.Series("cluster_id", self.cluster_labels),
            pl.Series("cluster_conf", np.round(self.cluster_scores, 4))
        ])
        # build cluster/outlier tracking
        clusters: Dict[int, List[int]] = {}
        outliers: List[int] = []
        for idx, c_id in enumerate(self.cluster_labels):
            if c_id >= 0: # valid cluster ID
                clusters.setdefault(c_id, []).append(idx)
            else: # noise point
                outliers.append(idx)
        cluster_counts = {cid: len(ixs) for cid, ixs in clusters.items()}
        self.results = ClusteringResults(
            df=df_with_clusters,
            clusters=clusters,
            cluster_counts=cluster_counts,
            outliers=outliers,
            model=self.cluster_model,
        )
        if self.verbose:
            n_clusters = len(clusters)
            n_clustered = sum(len(ixs) for ixs in clusters.values())
            print(f"Found {n_clusters} clusters containing {n_clustered} items.")
            print(f"Detected {len(outliers)} outliers/noise points.")
        return self

    def run_cluster_fitting(self, embeddings: np.ndarray, **kwargs):
        """ select and fit the chosen clustering object
            Args:
                embeddings: Embedding vectors from EmbeddedDataFrameBuilder
                **kwargs: Additional parameters for the clustering algorithm
            Returns:
                self: For method chaining
        """
        algo = self.algorithm.lower()
        if algo == "hdbscan":
            # somewhat arbitrary defaults for HDBSCAN - set elsewhere later for easy reference
            self.cluster_model = HDBSCANClusterer(
                min_cluster_size=kwargs.get("min_cluster_size", 5),
                min_samples=kwargs.get("min_samples"),
                metric=kwargs.get("metric", "euclidean"),
            )
        elif algo == "kmeans":
            self.cluster_model = KMeansClusterer(n_clusters=kwargs.get("n_clusters", 10), random_state=kwargs.get("random_state", 42))
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algo}")
        # Unified calls
        self.cluster_labels = self.cluster_model.fit_predict(embeddings)
        self.cluster_scores = self.cluster_model.confidence_scores(embeddings)

    @clustering_step("keyword labeling")
    def add_keyword_labels(self, top_k: int = 5) -> "ClusteringBuilder":
        """ simple approach (first trial): pick top-K frequent keywords as labels for each cluster """
        if not self.results:
            raise RuntimeError("Must run fit() before labeling.")
        # store mapping between cluster IDs and combined keywords
        cluster_keywords: Dict[int, str] = {}
        for cid, indices in self.results.clusters.items():
            # gather cluster keywords, combine them, and get top-K frequent words
            cluster_texts = [kws for i in indices if (kws := self.df["keywords_text"][i]) and i < len(self.df)]
            all_words = " ".join(cluster_texts).split()
            word_counts = Counter(all_words)
            top_words = [word for word, _ in word_counts.most_common(top_k)]
            cluster_keywords[cid] = " | ".join(top_words) if top_words else "Unclustered"
        # add labels to df as new column "cluster_label"
        labeled_df = self.results.df.with_columns([
            pl.col("cluster_id").map_elements(
                lambda x: cluster_keywords.get(x, "Unclustered") if x >= 0 else "Noise", return_dtype=pl.String
            ).alias("cluster_label")
        ])
        self.results.df = labeled_df
        return self

    @clustering_step("zero-shot labeling")
    def add_zero_shot_labels(self, candidate_labels: List[str], model_name: str = "facebook/bart-large-mnli") -> "ClusteringBuilder":
        """ Optionally label clusters using zero-shot classification
            Args:
                candidate_labels: List of candidate category labels
                model_name: Model to use for zero-shot classification
            Returns:
                self: For method chaining
        """
        if not self.results:
            raise RuntimeError("Must run fit() before labeling.")
        from onetab_autosorter.models.cluster_utils import ClusterTextDataset
        device = 0 if torch.cuda.is_available() else -1
        # TODO: add more arguments here to make full use of the pipeline structure
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        # prepare a mapping of cluster_id -> combined keywords to instantiate a Pytorch dataset
        cluster_texts = {
            cid: " ".join(self.df["keywords_text"][i] for i in idxs)
            for cid, idxs in self.results.clusters.items()
        }
        text_dataset = ClusterTextDataset(cluster_texts)
        loader = DataLoader(text_dataset, batch_size=self.batch_size, shuffle=False)
        label_map: Dict[int, List[str]] = {}
        for batch in tqdm(loader, desc="Zero-shot classification progress", unit="batch"):
            ids, texts = batch
            with torch.no_grad():
                # convert texts (returned as tuple of strings) to list of strings for the classifier
                results: List[Dict[str, List[Union[str, int, float]]]] = classifier(list(texts), candidate_labels, batch_size=self.batch_size)
            for cid, res in zip(ids, results):
                if cid not in label_map:
                    label_map[cid] = []
                best_label = res["labels"][0] if res["labels"] else "Other"
                label_map[cid].append(best_label)
        # build a new Polars DF and convert to long format (explode) to get each (cid, label) on its own row
        mode_df = pl.DataFrame({
            # add labels into DataFrame based on cluster IDs
            "cluster_id": list(label_map.keys()), "labels": list(label_map.values())
            }).explode("labels").group_by("cluster_id").agg(
            pl.col("labels").mode().first().alias("zero_shot_label") # pick the most frequent label for a given ID
        )
        df2 = self.results.df.join(mode_df, on="cluster_id", how="left")
        df2.glimpse()
        df2 = df2.with_columns(
            pl.when(pl.col("cluster_id") < 0)
              .then(pl.lit("Noise"))
              .otherwise(pl.col("zero_shot_label").fill_null("Other"))
              .alias("zero_shot_label")
        )
        # save zero-shot labels and counts to results.zero_shot_counts
        self.results.zero_shot_counts = df2.group_by("zero_shot_label").agg(pl.count("url")).to_dict()
        self.results.df = df2
        return self

    #~ may make these class properties later, but for now it would break certain logic relying on it like using `if not self.results`
    def get_results(self) -> ClusteringResults:
        if not self.results:
            raise RuntimeError("No clustering results available.")
        return self.results

    def get_dataframe(self) -> pl.DataFrame:
        if not self.results:
            raise RuntimeError("No clustering results available.")
        return self.results.df

    def predict_cluster(self, new_embedding: np.ndarray) -> int:
        """ Predict cluster ID for a new embedding after initial sorting
            Args:
                new_embedding: Embedding vector for the new item
            Returns:
                Predicted cluster ID
        """
        if not self.cluster_model:
            raise RuntimeError("No cluster model available.")
        # TODO: may need to put this in a try-except block to catch errors from the model indicating that it hasn't been fit
        # if using HDBSCAN, use approximate_predict
        # TODO: make a common predict method in the Clusterer classes later for a common interface - not using anywhere yet to worry about it
        if isinstance(self.cluster_model, HDBSCANClusterer):
            label, _ = self.cluster_model.model.approximate_predict(new_embedding.reshape(1, -1))
            return label[0]
        else:
            # for KMeans or others
            return self.cluster_model.model.predict(new_embedding.reshape(1, -1))[0]

    @staticmethod
    def cluster_factory(
        df: pl.DataFrame,
        embeddings: np.ndarray,
        algorithm: str = "hdbscan",
        add_keyword_labels: bool = True,
        add_zero_shot_labels: bool = False,
        candidate_labels: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs
    ) -> ClusteringResults:
        """ factory function to fit and label in one shot
                    Args:
                df: DataFrame from EmbeddedDataFrameBuilder
                embeddings: Embeddings from EmbeddedDataFrameBuilder
                algorithm: Clustering algorithm ('hdbscan' or 'kmeans')
                add_keyword_labels: Whether to add keyword-based labels
                add_zero_shot_labels: Whether to add zero-shot classification labels
                candidate_labels: Labels for zero-shot classification
                verbose: Whether to print progress information
                **kwargs: Additional clustering parameters like "metric", etc
            Returns:
                ClusteringResults object
        """
        if add_zero_shot_labels and not candidate_labels:
            #raise ValueError("Must provide candidate labels for zero-shot labeling.")
            candidate_labels = DEFAULT_ZERO_SHOT_LABELS
        builder = ClusteringBuilder(algorithm=algorithm, verbose=verbose)
        builder.fit(df, embeddings, **kwargs)
        if add_keyword_labels:
            builder.add_keyword_labels(top_k=kwargs.get("top_k", 5))
        if add_zero_shot_labels and candidate_labels:
            builder.add_zero_shot_labels(candidate_labels, model_name=kwargs.get("model_name", "facebook/bart-large-mnli"))
        return builder.get_results()


def inject_generated_labels(df: pl.DataFrame, label_map: Dict[int, str], label_col: str = "cluster_label") -> pl.DataFrame:
    """ utility for a manual approach to label injection
        Args:
            df: Input DataFrame with cluster_id column
            label_map: Dictionary mapping cluster IDs to label strings
            label_col: Name of the new column to create
        Returns:
            DataFrame with injected label column
    """
    # maps cluster IDs to labels
    label_series = df["cluster_id"].map_elements(lambda cid: label_map.get(cid, "Unclustered"), return_dtype=pl.String)
    return df.with_columns([pl.Series(label_col, label_series)])


# TODO: use same treatment as in the ClusteringBuilder class in `add_zero_shot_labels` - abstract to common base functionality later
def generate_cluster_labels_zero_shot(
    df: pl.DataFrame,
    candidate_labels: List[str],
    keyword_col: str = "keywords_text",
    label_col: str = "cluster_label",
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 32
) -> pl.DataFrame:
    # TODO: update docstring again after refactoring
    """ Separate global function that performs zero-shot labeling on each cluster_id
        Args:
            df: DataFrame with cluster_id and keywords_text columns
            candidate_labels: List of candidate labels for classification
            keyword_col: Column containing keywords for each entry
            label_col: Name of the new label column to create
            model_name: Model name for zero-shot classification
            batch_size: Batch size for processing texts
        Returns:
            DataFrame with added label column
    """
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    cluster_ids = df.filter(pl.col("cluster_id") >= 0)["cluster_id"].unique().to_list()
    all_texts = []
    for c_id in cluster_ids:
        c_data = df.filter(pl.col("cluster_id") == c_id)
        combined = " ".join(c_data[keyword_col].to_list())
        all_texts.append(combined)
    all_results = []
    # TODO: create dataset for this to use the pipeline better
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        with torch.no_grad():
            batch_results = classifier(batch_texts, candidate_labels, batch_size=batch_size)
        all_results.extend(batch_results)
    #~ still might want to decouple the use of the initial clustering labels from this and sort them after classification, honestly
    cluster_labels: Dict[int, str] = {} # label mapping between cluster IDs and labels
    for c_id, result in zip(cluster_ids, all_results):
        best_label = result["labels"][0] if result["labels"] else "Unclassified"
        cluster_labels[c_id] = best_label
    return df.with_columns([
        pl.col("cluster_id").map_elements(
            lambda x: cluster_labels.get(x, "Unclustered"),
            return_dtype=pl.String #pl.Utf8
        ).alias(label_col)
    ])
