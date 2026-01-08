"""Feature preprocessing utilities for SAT-based decision tree."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder


class MedianThresholdLabelBinarizer:
    """Binarize categorical labels using split closest to 50/50.

    For categorical labels (e.g., "unacc", "acc", "good", "vgood"),
    finds the binary partition where positive/negative frequencies are
    closest to 50/50.

    Algorithm:
        1. Get all unique labels and their counts
        2. Try all possible binary partitions (2^(n-1) for n classes)
        3. Pick partition where |freq_positive - 0.5| is minimized
        4. Labels in positive group → 1, else 0

    This works for any categorical labels without assumptions.
    """

    def __init__(self):
        self.positive_labels_: list | None = None
        self.split_ratio_: float | None = None

    def fit(self, labels: np.ndarray) -> "MedianThresholdLabelBinarizer":
        """Find binary split closest to 50/50."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_labels)
        n_total = len(labels)

        best_diff = float("inf")
        best_positive_labels: list = []

        # For n classes, iterate through all non-empty, non-full subsets
        # Use bitmask: class i is positive if bit i is set
        for mask in range(1, (1 << n_classes) - 1):
            positive_count = sum(counts[i] for i in range(n_classes) if mask & (1 << i))
            ratio = positive_count / n_total
            diff = abs(ratio - 0.5)

            if diff < best_diff:
                best_diff = diff
                best_positive_labels = [
                    unique_labels[i] for i in range(n_classes) if mask & (1 << i)
                ]
                self.split_ratio_ = ratio

        self.positive_labels_ = best_positive_labels
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels to binary: label in positive group → 1."""
        if self.positive_labels_ is None:
            raise RuntimeError("Must call fit() before transform()")
        return np.isin(labels, self.positive_labels_)


def binarize_labels_ovr(
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Convert multi-class labels to binary via one-vs-rest.

    Selects the minority class (smallest count) as positive.
    This creates an imbalanced binary problem.

    Args:
        raw_labels: Raw labels from PMLB

    Returns:
        Tuple of (binary_labels, positive_label_value)
    """
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    minority_idx = int(np.argmin(counts))
    positive_label = unique_labels[minority_idx]

    binary_labels = raw_labels == positive_label

    return binary_labels, int(positive_label)


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
