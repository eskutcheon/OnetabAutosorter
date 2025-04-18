# -*- coding: utf-8 -*-
import json
import hashlib
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Set, Callable, Union, Optional, Any
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
#from datetime import datetime

DEFAULT_FEATURE_CATEGORIES = ["keyword", "path", "subdomain", "date"] #, "group"]



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

    # @feature_category("group")
    # def add_group_features(self, df: pl.DataFrame) -> pl.DataFrame:
    #     if "group_ids" not in df.columns:
    #         return df
    #     self.results.feature_info["group_ids"] = "Comma-separated group IDs representing bookmark sessions"
    #     return df

    # TODO: this whole function needs a TON of work - just starting by splitting it up into smaller functions first
    def generate_hybrid_embeddings(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """ hybrid approach to a tabular embedding that integrates keywords, partial content, domain, etc """
        if not self.include_embeddings:
            return None
        from sklearn.random_projection import GaussianRandomProjection
        
        def transform_embedding(data: str, projector: GaussianRandomProjection, needs_fitting: bool = False) -> np.ndarray:
            # TODO: updated approach should probably encode it all at once for better efficiency
            embedding = self.model.encode([data])[0]
            if needs_fitting:
                return projector.fit_transform(embedding.reshape(1, -1))[0]
            else:
                return projector.transform(embedding.reshape(1, -1))[0]
        
        def get_fallback_embedding(dim):
            return np.zeros(dim, dtype=np.float32)
        
        # get embedding dimension from the pre-trained model
        embedding_dim = self.model.get_sentence_embedding_dimension()
        # save dataframe columns as local list variables
        keyword_dicts = df["raw_keywords"].to_list()
        titles = df["title"].to_list()
        contents = df["contents"].to_list() if "contents" in df.columns else ["" for _ in keyword_dicts]
        domains = df["domain"].to_list()
        # creating random projection object to reduce dimensionality
            # ~may want to change to KPCA with a kernel fit on each feature later
        content_dim = embedding_dim // 3
        title_dim = embedding_dim // 6
        content_projector = GaussianRandomProjection(n_components=content_dim)
        title_projector = GaussianRandomProjection(n_components=title_dim)
        all_embeddings = []
        for i in range(len(keyword_dicts)):
            try:
                keywords_dict = json.loads(keyword_dicts[i])
                # create weighted keyword embedding with full dimensionality preserved
                if keywords_dict:
                    keywords = list(keywords_dict.keys())
                    weights = np.array(list(keywords_dict.values()), dtype=float)
                    batch_emb = self.model.encode(keywords, show_progress_bar=(self.verbose and i == 0))
                    weights_sum = weights.sum()
                    if weights_sum > 0.0:
                        weights /= weights_sum
                    keyword_embedding = np.average(batch_emb, axis=0, weights=weights)
                else: # fallback is empty zeros for all embeddings
                    keyword_embedding = get_fallback_embedding(embedding_dim)
                # title embedding (projected to reduced dimensions)
                title_embedding = get_fallback_embedding(title_dim)
                if titles[i] and len(titles[i]) > 3:
                    title_embedding = transform_embedding(titles[i], title_projector, needs_fitting=(i == 0))
                # content embedding (projected to reduced dimensions)
                content_embedding = get_fallback_embedding(content_dim)
                if contents[i] and len(contents[i]) > 50:
                    #! FIXME: changed my mind on this though Copilot recommended it
                    sentences = contents[i].split(".") # TODO: include a more robust sentence tokenizer (still based on punctuation) with NLTK
                    if len(sentences) >= 3:
                        # extract first sentence and middle sentence
                        key_content = sentences[0] + ". " + sentences[len(sentences)//2]
                    else:
                        key_content = sentences[0] if sentences else ""
                    if key_content:
                        content_embedding = transform_embedding(key_content, content_projector, needs_fitting=(i == 0))
                # minimal domain hashing for additional context
                domain_context = get_fallback_embedding(self.DOMAIN_HASH_LENGTH)
                # print("testing if error is in domains[i]")
                if domains[i]:
                    domain_hash = hashlib.md5(domains[i].encode()).digest()[:self.DOMAIN_HASH_LENGTH]
                    domain_context = np.array([b/255.0 for b in domain_hash])
                # weight embeddings by importance then concatenate
                # TODO: add adjustment by how many entries actually have these features
                keyword_embedding *= 1.0
                title_embedding *= 0.6
                content_embedding *= 0.4
                domain_context  *= 0.1
                combined = np.concatenate([keyword_embedding, title_embedding, content_embedding, domain_context])
                all_embeddings.append(combined)
            except Exception as ex:
                if self.verbose:
                    print(f"Error building embedding for index {i}: {ex}")
                fallback_dims = embedding_dim + content_dim + 8
                all_embeddings.append(get_fallback_embedding(fallback_dims))
            # optional progress message
            #~ should be able to replace this with an actual progress bar after changing things up to embed all data at once with self.model
            if self.verbose and i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(keyword_dicts)} entries.")
        final_dim = embedding_dim + content_dim + self.DOMAIN_HASH_LENGTH
        self.results.feature_info["embeddings"] = (
            f"Hybrid embeddings: {embedding_dim} (keywords) + {title_dim} (title) + "
            f"{content_dim} (content) + 8 (domain) = {final_dim} dims total."
        )
        return np.vstack(all_embeddings)

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







def test_cluster_entries(df: pl.DataFrame, embeddings: np.ndarray) -> Dict[str, Any]:
    """ Process and cluster the embedded entries using HDBSCAN. """
    from onetab_autosorter.clustering import ClusteringBuilder
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
    zero_shot_counts = cluster_results.zero_shot_counts["url"]
    for label, count in zip(zero_shot_labels, zero_shot_counts):
        print(f"Zero-shot label {label}: {count} items")
    return clustered_df


def view_grouped_df(df: pl.DataFrame, n: int = 10, group_by: str = "cluster_label", agg_col: str = "url") -> None:
    """ View grouped DataFrame by group_ids """
    grouped_df = df.group_by(group_by).agg(pl.col(agg_col).count()).select(pl.exclude("contents"))
    print(f"Grouped DataFrame by {group_by}:")
    grouped_df.sort(agg_col, descending=True).glimpse()
    return grouped_df


if __name__ == "__main__":
    import json
    from pprint import pprint
    test_data_path = r"path/to/file.json"
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
    # show top 10 most populated clusters
    #_ = view_grouped_df(clustered_df, n=20, group_by="cluster_label", agg_col="url")
    #labeled_df = view_grouped_df(clustered_df, n=67, group_by="zero_shot_label", agg_col="url")
