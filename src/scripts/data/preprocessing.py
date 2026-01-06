"""Feature preprocessing utilities for SAT-based decision tree."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder


def filter_by_frequency(
    features: np.ndarray,
    *,
    min_freq: float = 0.05,
    max_freq: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove features with extreme frequencies.

    Args:
        features: (n_samples, n_features) binary feature matrix
        min_freq: Minimum fraction of samples where feature must be True
        max_freq: Maximum fraction of samples where feature can be True

    Returns:
        Tuple of (filtered_features, feature_indices_kept)
    """
    if features.size == 0:
        return features, np.array([], dtype=int)

    feature_freq = np.mean(features, axis=0)
    mask = (feature_freq >= min_freq) & (feature_freq <= max_freq)
    feature_indices = np.where(mask)[0]

    return features[:, mask], feature_indices


def select_k_best(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    score_func=mutual_info_classif,
) -> np.ndarray:
    """
    Select k best features using sklearn SelectKBest with mutual information.

    Args:
        features: (n_samples, n_features) feature matrix
        labels: (n_samples,) labels
        k: Number of top features to select
        score_func: Scoring function (default: mutual_info_classif)

    Returns:
        (n_samples, k) feature matrix with k best features
    """
    if k <= 0:
        return np.zeros((features.shape[0], 0), dtype=features.dtype)
    if k >= features.shape[1]:
        return features

    selector = SelectKBest(score_func=score_func, k=k)
    selected = selector.fit_transform(features, labels)

    return selected


def make_consistent(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove inconsistent samples.

    For duplicate features with different labels, keep only rows where
    the label matches the majority label for that feature pattern.

    Args:
        features: (n_samples, n_features) binary feature matrix
        labels: (n_samples,) binary labels

    Returns:
        Tuple of (features, labels) with inconsistent rows removed.
    """
    unique_features, inverse = np.unique(features, axis=0, return_inverse=True)

    keep_mask = np.zeros(len(features), dtype=bool)

    for i in range(len(unique_features)):
        mask = inverse == i
        label_group = labels[mask]
        unique_label_vals = np.unique(label_group)

        if len(unique_label_vals) == 1:
            # Consistent: keep all rows with this feature pattern
            keep_mask[mask] = True
        else:
            # Inconsistent: keep only rows matching majority label
            n_positive = np.sum(label_group)
            majority_label = n_positive >= len(label_group) / 2
            consistent_rows = label_group == majority_label
            keep_mask[mask] = consistent_rows

    return features[keep_mask], labels[keep_mask]


def one_hot_encode(
    raw_features: np.ndarray,
) -> np.ndarray:
    """
    One-hot encode raw features.

    Handles both numeric and categorical features automatically.

    Args:
        raw_features: Raw feature matrix from PMLB

    Returns:
        Binary feature matrix after one-hot encoding
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return encoder.fit_transform(raw_features).astype(bool)


def binarize_labels_ovr(
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert multi-class labels to binary via one-vs-rest.

    Selects the rarest class as positive (minority class) for balanced
    binary classification.

    Args:
        raw_labels: Raw labels from PMLB

    Returns:
        Tuple of (binary_labels, minority_label_value)
    """
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    minority_idx = np.argmin(counts)
    minority_label = unique_labels[minority_idx]
    binary_labels = (raw_labels == minority_label).astype(bool)

    return binary_labels, minority_label
