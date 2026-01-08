"""PMLB data loading and preprocessing utilities."""

from __future__ import annotations

import numpy as np
from pmlb import fetch_data

from scripts.data.preprocessing import (
    binarize_labels_ovr,
    filter_by_frequency,
    make_consistent,
    MedianThresholdLabelBinarizer,
    one_hot_encode,
    select_k_best,
)


def load_dataset(
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch raw dataset from PMLB.

    Args:
        name: Name of the PMLB dataset to fetch
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    features, labels = fetch_data(name, return_X_y=True, local_cache_dir=".cache")
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def preprocess(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    encode_features: str = "onehot",
    binarize_labels: str = "ovr",
    min_feature_freq: float = 0.05,
    feature_selection: str | None = "kbest",
    n_features: int | None = None,
    ensure_consistent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features and labels for SAT-based decision tree.

    Args:
        features: Raw feature matrix (will be encoded if encode_features != "none")
        labels: Raw labels (will be binarized if binarize_labels != "none")
        encode_features: "onehot" to one-hot encode, "none" to skip
        binarize_labels: "ovr" for one-vs-rest, "threshold" for 50/50 split
        min_feature_freq: Minimum feature frequency for filtering
        feature_selection: "kbest" to select top k features, or None
        n_features: Number of features to keep (if feature_selection="kbest")
        ensure_consistent: Remove inconsistent samples (same features, diff labels)

    Returns:
        Tuple of (processed_features, processed_labels)
    """
    # Step 1: Encode features
    if encode_features == "onehot":
        features = one_hot_encode(features)

    # Step 2: Binarize labels
    if binarize_labels == "ovr":
        labels, _ = binarize_labels_ovr(labels)
    elif binarize_labels == "threshold":
        binarizer = MedianThresholdLabelBinarizer()
        binarizer.fit(labels)
        labels = binarizer.transform(labels)

    # Step 3: Filter by frequency (remove rare or too-common features)
    if min_feature_freq > 0:
        features, _ = filter_by_frequency(features, min_freq=min_feature_freq)

    # Step 4: Select k-best features
    if feature_selection == "kbest" and n_features is not None:
        features = select_k_best(features, labels, n_features)

    # Step 5: Ensure consistency
    if ensure_consistent:
        features, labels = make_consistent(features, labels)

    return features, labels


def load_and_preprocess(
    dataset_name: str,
    *,
    encode_features: str = "onehot",
    binarize_labels: str = "ovr",
    min_feature_freq: float = 0.05,
    feature_selection: str | None = "kbest",
    n_features: int | None = None,
    ensure_consistent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch and preprocess a PMLB dataset in one call.

    Args:
        dataset_name: Name of the PMLB dataset
        random_state: Random seed for reproducibility
        encode_features: "onehot" to encode, "none" to skip
        binarize_labels: "ovr" for one-vs-rest, "threshold" for 50/50 split
        min_feature_freq: Minimum feature frequency for filtering
        feature_selection: "kbest" to select top k features, or None
        n_features: Number of features to keep (if feature_selection="kbest")
        ensure_consistent: Remove inconsistent samples

    Returns:
        Tuple of (processed_features, processed_labels)
    """
    features, labels = load_dataset(dataset_name)

    return preprocess(
        features,
        labels,
        encode_features=encode_features,
        binarize_labels=binarize_labels,
        min_feature_freq=min_feature_freq,
        feature_selection=feature_selection,
        n_features=n_features,
        ensure_consistent=ensure_consistent,
    )
