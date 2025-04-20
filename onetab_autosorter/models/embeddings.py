# -*- coding: utf-8 -*-
import json
# import hashlib
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Set, Callable, Union, Optional, Any
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
#from datetime import datetime

DEFAULT_FEATURE_CATEGORIES = ["keyword", "path", "subdomain", "date"] #, "group"]

# TODO: move later
from torch.utils.data import Dataset, DataLoader

class TextPairDataset(Dataset):
    """ simple Dataset over two parallel lists of strings """
    def __init__(self, metas: List[str], contents: List[str]):
        assert len(metas) == len(contents)
        self.metas = metas
        self.contents = contents

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        return self.metas[idx], self.contents[idx]


@dataclass
class EmbeddingResults:
    df: Optional[pl.DataFrame] = None
    embeddings: Optional[np.ndarray] = None
    feature_info: Dict[str, str] = field(default_factory=dict)




def feature_category(category_name: str):
    """ decorator that gates feature-adding methods by category """
    def decorator(func: Callable[["EmbeddedDataFrameBuilder", pl.DataFrame], pl.DataFrame]):
        @wraps(func)
        def wrapper(cls: "EmbeddedDataFrameBuilder", df: pl.DataFrame) -> pl.DataFrame:
            # check if category should be included
            if category_name not in cls.include_features:
                return df
            if cls.verbose: # log if verbose
                print(f"Adding {category_name} features...")
            try:
                # run the actual feature addition method
                return func(cls, df)
            except Exception as e:
                if cls.verbose:
                    print(f"Error adding {category_name} features: {e}")
                return df
        return wrapper
    return decorator


class EmbeddedDataFrameBuilder:
    """ Builder class for creating embeddings for bookmark clustering.
        - prioritizes the most meaningful features for clustering bookmarks, with emphasis on keywords as
        the primary descriptors. It excludes superficial features that might add noise to the clustering process.
        - Feature Categories:
            1. Keyword Features - Derived from generated keywords (primary focus)
            2. Path Features - URL path depth to capture content specificity
            3. Subdomain Indicator - Whether content is from a specialized subdomain
            4. Clean Date - Simplified date representation
            5. Group Context - Simple group membership indicator
    """
    DOMAIN_HASH_LENGTH = 8 # length of the domain hash for embedding - doesn't need to be too long
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        keyword_field: str = "keywords",
        include_features: List[str] = None,
        batch_size: int = 32,
        normalize_numeric: bool = True,
        include_embeddings: bool = True,
        verbose: bool = False
    ):
        self.embedding_model = embedding_model
        self.keyword_field = keyword_field
        self.include_features = include_features or DEFAULT_FEATURE_CATEGORIES
        self.batch_size = batch_size
        self.normalize_numeric = normalize_numeric
        self.include_embeddings = include_embeddings
        self.verbose = verbose
        self.model: Optional[SentenceTransformer] = self._load_model()
        self.results = EmbeddingResults()
        self.added_columns: Set[str] = set()

    def _load_model(self) -> Optional[SentenceTransformer]:
        if not self.include_embeddings:
            return None
        if self.verbose:
            print(f"Generating hybrid embeddings with model='{self.embedding_model}'")
        # safe model load
        try:
            model = SentenceTransformer(self.embedding_model)
        except Exception as exc:
            if self.verbose:
                print(f"ERROR: Could not load embedding model {self.embedding_model}!")
            raise exc # don't allow the default mean pooling model to be used
        return model

    def add_columns(self, df: pl.DataFrame, columns: List[pl.Expr], feature_info: Dict[str, str]) -> pl.DataFrame:
        """ Helper method to add columns to a passed DataFrame and update feature descriptions consistently
            Args:
                df: Input DataFrame
                columns: List of column expressions to add
                feature_info: Dictionary mapping column names to descriptions
            Returns:
                DataFrame with added columns
        """
        # collect column names from expressions for tracking
        col_names = [c.meta.output_name() for c in columns]
        self.added_columns.update(col_names) # track added columns
        self.results.feature_info.update(feature_info)
        # apply the column expressions to the DataFrame
        return df.with_columns(columns)

    @feature_category("keyword")
    def add_keyword_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """ No derived features here """
        # TODO: might move the creation of keyword features from entries_to_dataframe to here
        self.results.feature_info["keywords_text"] = "Space-separated keywords extracted from content (primary descriptor)"
        return df

    @feature_category("path")
    def add_path_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """ add URL path depth to represent content specificity """
        path_expr = [pl.col("url").str.count_matches("/").alias("path_depth")]
        path_info = {"path_depth": "Number of URL path segments (indicates more specific content)"}
        return self.add_columns(df, path_expr, path_info)

    @feature_category("subdomain")
    def add_subdomain_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """ add subdomain Boolean indicator to represent specialized content """
        expr = [pl.col("domain").str.count_matches("\\.").gt(1).cast(pl.Int32).alias("is_subdomain")]
        info = {"is_subdomain": "Whether URL is on a subdomain (specialized content)"}
        return self.add_columns(df, expr, info)

    @feature_category("date")
    def add_date_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if "date" not in df.columns:
            return df
        try: # attempt to add a single normalized date feature (days since unit epoch (1970-01-01)
            expr = [pl.col("date").cast(pl.Datetime).dt.epoch(time_unit="d").alias("days_since_epoch")]
            info = {"days_since_epoch": "Days since epoch (normalized date representation)"}
            return self.add_columns(df, expr, info)
        except Exception as e:
            if self.verbose:
                print(f"Error processing date feature: {e}")
            return df


    def _project_embeddings(self, embeddings: np.ndarray, dim: int) -> np.ndarray:
        """ Project embeddings to reduce dimensionality """
        from sklearn.random_projection import GaussianRandomProjection
        projector = GaussianRandomProjection(n_components=dim)
        return projector.fit_transform(embeddings)

    # TODO: this whole function needs a TON of work - just starting by splitting it up into smaller functions first
    def generate_hybrid_embeddings(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """ hybrid approach to a tabular embedding - batch-encode metadata (title, domain, keywords) and content
            separately via PolarsDataset + DataLoader, then project & combine with prescribed weighting.
        """
        if not self.include_embeddings:
            return None
        if self.verbose:
            print(f"Generating batched embeddings via PolarsDataset and DataLoader")
        CONTENT_MAX_TOKENS = 100 # max number of tokens to use for content embedding
        # build `meta_text` and truncated `content_text` columns
        meta_text_expr = (
            pl.col("title").fill_null("") + " [SEP] " + pl.col("domain").fill_null("") + " [SEP] " + pl.col("keywords_text").fill_null("")
        ).alias("meta_text")
        df2 = df.with_columns([
            meta_text_expr,
            pl.col("contents").map_elements(
                lambda s: " ".join(s.split()[:CONTENT_MAX_TOKENS]) if s else "", return_dtype=pl.String
            ).alias("content_text")
        ])
        # create a PyTorch Dataset from the relevant columns
        ds = TextPairDataset(df2["meta_text"].to_list(), df2["content_text"].to_list())
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        # iterate over batches and encode each with SentenceTransformer
        meta_embs, content_embs = [], []
        # TODO: wrap with tqdm later
        for batch in loader:
            texts_meta, texts_content = batch
            emb_meta = self.model.encode(texts_meta, show_progress_bar=False)
            emb_content = self.model.encode(texts_content, show_progress_bar=False)
            meta_embs.append(emb_meta)
            content_embs.append(emb_content)
        # stack samples as rows in a 2D array
        meta_emb = np.vstack(meta_embs)
        content_emb = np.vstack(content_embs)
        # project content embeddings to reduce dimensionality
        content_proj = self._project_embeddings(content_emb, meta_emb.shape[1] // 3)
        # combine with weights: metadata full (1.0), content lower (0.4)
        final = np.hstack([meta_emb * 1.0, content_proj * 0.4])
        # record feature_info
        self.results.feature_info["embeddings"] = (
            f"meta({meta_emb.shape[1]}) + content_proj({content_proj.shape[1]}) dims"
        )
        return final


    def combine_embeddings_with_features(self, text_embeddings: np.ndarray, df: pl.DataFrame) -> np.ndarray:
        """ combine numeric features with concatenated text embeddings (if any) """
        if not self.normalize_numeric or not self.added_columns:
            return text_embeddings
        if self.verbose:
            print("Combining embeddings with numeric features...")
        # using only specific features for clustering enhancement - deliberately limited to avoid noise
        numeric_cols = [c for c in ("path_depth", "days_since_epoch") if c in self.added_columns]
        if not numeric_cols: # if none found, return only text embeddings
            return text_embeddings
        # lazy loading since this function may never be called or even reach this point
        from sklearn.preprocessing import StandardScaler
        numeric_data = df.select(numeric_cols).to_numpy()
        try:
            scaled_numeric = StandardScaler().fit_transform(numeric_data)
        except ValueError as e:
            if self.verbose:
                print(f"Scaling error, skipping numeric scaling: {e}")
            scaled_numeric = numeric_data
        # add categorical features if applicable
        categorical_features = ["is_subdomain"] if "is_subdomain" in self.added_columns else []
        if categorical_features:
            categorical_data = df.select(categorical_features).to_numpy()
            combined_numeric = np.hstack([scaled_numeric, categorical_data])
        else:
            combined_numeric = scaled_numeric
        # combine text embeddings with numeric features (concatenate along columns)
        final = np.hstack([text_embeddings, combined_numeric])
        self.results.feature_info["embeddings"] = (
            f"{self.results.feature_info.get('embeddings','text embeddings')} + {combined_numeric.shape[1]} numeric dims"
        )
        return final

    def build_dataframe(self, entries: List[Dict[str, Any]]) -> EmbeddingResults: #-> Dict[str, Union[pl.DataFrame, np.ndarray, Dict[str, str]]]:
        """ build a new DataFrame with chosen features + optional embeddings
            Args:
                entries: List of bookmark entry dictionaries
            Returns:
                dataclass with "df", "embeddings", and "feature_info" fields
        """
        if self.verbose:
            print(f"Processing {len(entries)} bookmark entries...")
        # reset state just in case this is called multiple times
        self.results.feature_info.clear()
        self.added_columns.clear()
        # convert entries to base dataframe with special handling for keyword weights
        df = entries_to_dataframe(entries, self.keyword_field)
        df = self.add_keyword_features(df)
        df = self.add_path_features(df)
        df = self.add_subdomain_features(df)
        df = self.add_date_features(df)
        #df = self.add_group_features(df)
        # generate text embeddings
        text_embeddings = self.generate_hybrid_embeddings(df)
        final_embeddings = None
        if text_embeddings is not None:
            # attempt to combine text embeddings with numeric features
            final_embeddings = self.combine_embeddings_with_features(text_embeddings, df)
        if self.verbose:
            print(f"Final DataFrame shape: {df.shape}")
            if final_embeddings is not None:
                print(f"Final embeddings shape: {final_embeddings.shape}")
        self.results.df = df
        self.results.embeddings = final_embeddings
        return self.results


    @staticmethod
    def dataframe_factory(
        entries: List[Dict[str, Any]],
        embedding_model: str = "all-MiniLM-L6-v2",
        keyword_field: str = "keywords",
        include_features: List[str] = None,
        batch_size: int = 32,
        normalize_numeric: bool = True,
        include_embeddings: bool = True,
        verbose: bool = False
    ) -> EmbeddingResults: #Dict[str, Union[pl.DataFrame, np.ndarray, Dict[str, str]]]:
        """ Factory method for creating an embedded dataframe in one step """
        builder = EmbeddedDataFrameBuilder(
            embedding_model=embedding_model,
            keyword_field=keyword_field,
            include_features=include_features,
            batch_size=batch_size,
            normalize_numeric=normalize_numeric,
            include_embeddings=include_embeddings,
            verbose=verbose
        )
        return builder.build_dataframe(entries)


def entries_to_dataframe(entries: List[Dict[str, Any]], keyword_field: str = "keywords") -> pl.DataFrame:
    """ convert a list of bookmark entry dicts into a Polars DataFrame with JSON-based raw_keywords """
    def get_text_summary(entry):
        content = None
        for k in ("clean_text", "cleaned", "text", "content", "scraped"):
            content = entry.get(k, None)
            if content is not None:
                return content
        return ""

    rows = []
    for e in entries:
        row = {
            "url": e.get("url", ""),
            "title": e.get("title", ""),
            "domain": e.get("domain", ""),
            "date": e.get("date"), # should maybe simulate a default date like Unix epoch start 01011970
            #"group_ids": ",".join(str(g) for g in e.get("group_ids", [])), # using days_since_epoch instead since they should be equivalent
            "contents": get_text_summary(e),
        }
        # store raw keywords dictionary as string for later processing and preserve weights for each keyword
        kws = e.get(keyword_field, {})
        # ensure kws is a dictionary with keywords as keys and confidence scores as values
        if not (isinstance(kws, dict) and all(isinstance(v, (int, float)) for v in kws.values())):
            raise RuntimeWarning(f"Expected a dict for keywords, got {type(kws)}")
        # should only use one or the other of these in the final dataframe and embeddings
        row["raw_keywords"] = json.dumps(kws, ensure_ascii=False)
        row["keywords_text"] = " ".join(sorted(kws.keys())) # alphabetical order for consistency in embeddings for different samples
        rows.append(row)
    return pl.DataFrame(rows)


###########################################################################################################
# Testing embedding and clustering steps below - shouldn't need to keep anything here in the final codebase
###########################################################################################################


def test_cluster_entries(df: pl.DataFrame, embeddings: np.ndarray) -> Dict[str, Any]:
    """ Process and cluster the embedded entries using HDBSCAN. """
    from clustering import ClusteringBuilder
    cluster_results = ClusteringBuilder.cluster_factory(
        df=df,
        embeddings=embeddings,
        algorithm="hdbscan",
        #min_cluster_size=5,
        add_keyword_labels=True,
        add_zero_shot_labels=True,
        #candidate_labels = config.seed_kws,
        verbose=True
    )
    # get the final DataFrame with clusters and labels
    clustered_df = cluster_results.df
    print(f"Found {len(cluster_results.clusters)} clusters")
    print(f"Largest cluster has {max(cluster_results.cluster_counts.values())} items")
    #~ still considering training a weakly-supervised model for labeling clusters after I train it on my own bookmark categories
    for label, count in cluster_results.cluster_counts.items():
        print(f"Cluster {label}: {count} items")
    print(f"number labeled as outliers: {len(cluster_results.outliers)}")
    zero_shot_labels = cluster_results.zero_shot_counts["zero_shot_label"].to_list()
    zero_shot_counts = cluster_results.zero_shot_counts["url"].to_list()
    print("Zero-shot labels and member counts:")
    for label, count in zip(zero_shot_labels, zero_shot_counts):
        print(f"{label}: {count} items")
    return clustered_df


def view_grouped_df(df: pl.DataFrame, n: int = 10, group_by: str = "cluster_label", agg_col: str = "url") -> None:
    """ View grouped DataFrame by group_ids """
    grouped_df = df.group_by(group_by).agg(pl.col(agg_col).count()).select(pl.exclude("contents"))
    print(f"Grouped DataFrame by {group_by}:")
    grouped_df.sort(agg_col, descending=True).glimpse()
    return grouped_df


if __name__ == "__main__":
    import os
    import json
    from pprint import pprint
    test_data_path = r"path/to/file.json"
    cluster_results_path = test_data_path.replace("keywords", "clusters").replace(".json", ".csv")
    os.makedirs(os.path.dirname(cluster_results_path), exist_ok=True)
    with open(test_data_path, "r", encoding="utf-8") as fptr:
        test_data = json.load(fptr)
    results = EmbeddedDataFrameBuilder.dataframe_factory(
        test_data,
        embedding_model="all-distilroberta-v1",#"all-MiniLM-L6-v2",
        keyword_field="keywords",
        include_features=DEFAULT_FEATURE_CATEGORIES,
        batch_size=32,
        normalize_numeric=True,
        include_embeddings=True,
        verbose=True
    )
    df: pl.DataFrame = results.df
    embeddings: np.ndarray = results.embeddings
    print("FEATURE DESCRIPTIONS:")
    pprint(results.feature_info, indent=2)
    print("df columns: ", df.columns)
    df.select(pl.exclude("contents")).glimpse(max_items_per_column=10)
    print("EMBEDDINGS SHAPE, TYPE, RANGE: ", embeddings.shape, embeddings.dtype, (embeddings.min(), embeddings.max()))
    clustered_df: pl.DataFrame = test_cluster_entries(df, embeddings)
    clustered_df.write_csv(cluster_results_path)
    # show top 10 most populated clusters
    #_ = view_grouped_df(clustered_df, n=20, group_by="cluster_label", agg_col="url")
    #labeled_df = view_grouped_df(clustered_df, n=67, group_by="zero_shot_label", agg_col="url")
