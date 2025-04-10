from collections import Counter
from typing import Optional, Dict, List, Literal, Callable, Union, Any
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
# necessary for HDBSCAN clustering since we need a probabilities vector and an approximate_predict method for fuzzy clustering
from hdbscan import HDBSCAN
from transformers import pipeline
#from sklearn.cluster import HDBSCAN


# TODO: create mapping template for the metadata fields to be used in the embedding process
def entries_to_dataframe(entries: List[Dict[str, Any]], keyword_field: str = "keywords") -> pl.DataFrame:
    flat_data = []
    for entry in entries:
        row = {
            "url": entry["url"],
            "title": entry.get("title", ""),
            "domain": entry.get("domain", ""),
            "date": entry.get("date"),
            "group_ids": ",".join(str(g) for g in entry.get("group_ids", [])),
            "raw_date": entry.get("raw_date", ""),
        }
        # Combine keywords into space-separated text string (or weighted if needed)
        keywords = entry.get(keyword_field, {})
        if not isinstance(keywords, dict):
            raise RuntimeWarning(f"Expected a dictionary for keywords, got {type(keywords)}")
        row["keywords_text"] = " ".join(keywords.keys())
        row["keyword_weight_sum"] = sum(keywords.values())  # optional numeric signal
        flat_data.append(row)
    return pl.DataFrame(flat_data)


def embed_column(df: pl.DataFrame, column: str, model_name="all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = df[column].to_list()
    return model.encode(texts, show_progress_bar=True)


def enrich_with_metadata(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("keywords_text").str.len_chars().alias("kw_length"),
        #! simulating length just to add more numeric features - BUT these fields may be totally worthless for clustering
        pl.col("domain").str.len_chars().alias("domain_length"),
        pl.col("group_ids").str.count_matches(",").alias("group_count"),
    ])


def concatenate_embeddings(text_embeds: np.ndarray, numeric_df: pl.DataFrame) -> np.ndarray:
    numeric_matrix = numeric_df.to_numpy()
    return np.hstack([text_embeds, numeric_matrix])



def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = None) -> Dict[str, Any]:
    """ Clusters embeddings using the HDBSCAN algorithm.
        Args:
            embeddings (np.ndarray): The input embeddings to cluster.
            min_cluster_size (int): The minimum size of clusters. Defaults to 5.
            min_samples (int): The minimum number of samples in a neighborhood for a point to be considered a core point.
                            If None, it defaults to `min_cluster_size // 2`.
        Returns:
            Dict[str, Any]: A dictionary containing cluster labels, soft scores, and the HDBSCAN model.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size // 2,
        #cluster_selection_method='eom',
        # TODO: look into whether any similarity embedding scores like cosine loss like `metric="cosine"` exists for this
        metric='euclidean',
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    soft_scores = clusterer.probabilities_
    return {"labels": cluster_labels, "scores": soft_scores, "model": clusterer}


def inject_cluster_results(df: pl.DataFrame, labels: np.ndarray, scores: np.ndarray) -> pl.DataFrame:
    return df.with_columns([
        pl.Series("cluster_id", labels),
        pl.Series("cluster_conf", np.round(scores, 4))
    ])


def cluster_summary(df: pl.DataFrame, top_k: int = 5) -> Dict[int, List[str]]:
    cluster_keywords = df.filter(pl.col("cluster_id") >= 0).group_by("cluster_id").agg([
        pl.col("keywords_text").str.concat(" ").alias("combined_keywords")
    ])
    cluster_summaries = {}
    for row in cluster_keywords.iter_rows(named=True):
        words = row["combined_keywords"].split()
        common = Counter(words).most_common(top_k)
        cluster_summaries[row["cluster_id"]] = [w for w, _ in common]
    return cluster_summaries


def generate_cluster_labels_from_keywords(
    df: pl.DataFrame,
    keyword_col: str = "keywords_text",
    label_col: str = "cluster_label",
    top_k: int = 5
) -> pl.DataFrame:
    """
    Generates cluster-level summary labels from keywords and appends them to the dataframe.
    """
    cluster_summaries = cluster_summary(df, top_k=top_k)
    label_map = {cid: " | ".join(kw_list) for cid, kw_list in cluster_summaries.items()}
    #! FIXME: apply is deprecated for pl.Series - trying map_elements which isn't recommended in the docs - ensure I actually need this at all
    #label_series = df["cluster_id"].map_elements(lambda cid: label_map.get(cid, "Unclustered"), return_dtype=pl.String)
    # Use pl.when and pl.otherwise for a Polars-native approach
    label_series = pl.when(pl.col("cluster_id").is_in(label_map.keys())) \
                    .then(pl.col("cluster_id").map_dict(label_map)) \
                    .otherwise("Unclustered")
    # ALT (untested but I think it would work):
    # label_series = df["cluster_id"]
    # label_series = [ids if ids else "Unclustered" for ids in label_series]
    #label_series = df["cluster_id"].apply(lambda cid: label_map.get(cid, "Unclustered"))
    return df.with_columns([pl.Series(label_col, label_series)])




def generate_cluster_labels_llm(
    df: pl.DataFrame,
    top_k: int = 5,
    model_name: str = "google/flan-t5-large",
    # extend with title later
    prompt_template: str = "Suggest a general folder name for bookmarks related to the following keywords: {}"
) -> Dict[int, str]:
    """ Generates a cluster label using a zero-shot generative LLM based on keywords per cluster """
    cluster_kw_map = cluster_summary(df, top_k=top_k)
    generator = pipeline("text2text-generation", model=model_name)
    label_map = {}
    for cid, keywords in cluster_kw_map.items():
        prompt = prompt_template.format(", ".join(keywords))
        result = generator(prompt, max_new_tokens=12, do_sample=False)
        label_map[cid] = result[0]["generated_text"].strip()
    return label_map


def inject_generated_labels(df: pl.DataFrame, label_map: Dict[int, str], label_col: str = "cluster_label") -> pl.DataFrame:
    label_series = df["cluster_id"].map_elements(lambda cid: label_map.get(cid, "Unclustered"), return_dtype=pl.String)
    return df.with_columns([pl.Series(label_col, label_series)])


def generate_cluster_labels_zero_shot(
    df: pl.DataFrame,
    candidate_labels: List[str],
    keyword_col: str = "keywords_text",
    label_col: str = "cluster_label",
    model_name: str = "facebook/bart-large-mnli"
) -> pl.DataFrame:
    """ Generates cluster-level summary labels using a zero-shot classification model and appends them to the dataframe.
        Args:
            df (pl.DataFrame): The input dataframe containing cluster information.
            candidate_labels (List[str]): A list of candidate labels for classification.
            keyword_col (str): The column name containing keywords for each cluster. Defaults to "keywords_text".
            label_col (str): The column name where the generated labels will be stored. Defaults to "cluster_label".
            model_name (str): The name of the zero-shot classification model to use. Defaults to "facebook/bart-large-mnli".
        Returns:
            pl.DataFrame: The dataframe with an additional column containing the generated cluster labels.
        How it works:
            - For each unique cluster ID in the dataframe, the function combines the keywords from the specified column.
            - It uses a zero-shot classification model to classify the combined keywords into one of the candidate labels.
            - The label with the highest score is assigned to the cluster, and the results are appended to the dataframe.
    """
    from torch import cuda
    device = 0 if cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    cluster_labels = {}
    for cluster_id in df["cluster_id"].unique():
        cluster_data = df.filter(pl.col("cluster_id") == cluster_id)
        combined_keywords = " ".join(cluster_data[keyword_col].to_list())
        # Perform zero-shot classification
        result = classifier(combined_keywords, candidate_labels)
        if result["labels"]:
            best_label = result["labels"][0]  # Label with highest score
        else:
            best_label = "Unclassified"  # Fallback label in case of empty result
        cluster_labels[cluster_id] = best_label
    # Map the generated labels to the dataframe
    #label_series = df["cluster_id"].map_elements(lambda cid: cluster_labels.get(cid, "Unclustered"), return_dtype=pl.String)
    label_series = pl.when(pl.col("cluster_id").is_in(cluster_labels.keys())) \
                    .then(pl.col("cluster_id").map_dict(cluster_labels)) \
                    .otherwise("Unclustered")
    return df.with_columns([pl.Series(label_col, label_series)])