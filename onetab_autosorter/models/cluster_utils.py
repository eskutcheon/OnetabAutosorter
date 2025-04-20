# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Callable, Union, Any
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
# necessary for HDBSCAN clustering since we need a probabilities vector and an approximate_predict method for fuzzy clustering
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances




class ClusterTextDataset(Dataset):
    """ small PyTorch Dataset for clusters that's similar to a KeyDataset but uses a dict of texts instead of a DataFrame """
    def __init__(self, texts_map: Dict[int, str]):
        self.ids = list(texts_map.keys())
        self.texts = list(texts_map.values())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx]


@dataclass
class ClusteringResults:
    """ container for clustering outputs to maintain clean interfaces """
    df: pl.DataFrame
    clusters: Dict[int, List[int]] = field(default_factory=dict)
    cluster_counts: Dict[int, int] = field(default_factory=dict)
    zero_shot_counts: Dict[str, int] = field(default_factory=dict)
    outliers: List[int] = field(default_factory=list)
    model: Any = None # could be HDBSCAN, KMeans, etc.


# TODO: create registry for subclasses later

class BaseClusterer(ABC):
    """ common base class for different clustering algorithms including those I might try in the future """
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def confidence_scores(self, X: np.ndarray) -> np.ndarray:
        pass


@dataclass
class HDBSCANClusterer(BaseClusterer):
    min_cluster_size: int = 5
    min_samples: int = None
    metric: str = "arccos" #cosine_distances #"cosine"
    model: HDBSCAN = field(default=None, init=False)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if self.min_samples is None:
            self.min_samples = self.min_cluster_size // 2
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            prediction_data=True,
        )
        labels = self.model.fit_predict(X)
        return labels

    def confidence_scores(self, X: np.ndarray) -> np.ndarray:
        return self.model.probabilities_


@dataclass
class KMeansClusterer(BaseClusterer):
    n_clusters: int = 10
    random_state: int = 42
    model: KMeans = field(default=None, init=False)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        return self.model.fit_predict(X)

    def confidence_scores(self, X: np.ndarray) -> np.ndarray:
        """ use the inverse of distances to cluster centers as a proxy for confidence scores for KMeans """
        centers = self.model.cluster_centers_
        labels = self.model.labels_
        distances = np.array([np.linalg.norm(x - centers[label]) for x, label in zip(X, labels)])
        # convert distances to confidence scores
        max_dist = np.max(distances) + 1e-6  # TOL to avoid DivisionByZero error
        return 1.0 - (distances / max_dist)