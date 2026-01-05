"""Comparison experiment between DT1 and sklearn DecisionTree."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dt1.exceptions import UpperBoundTooStrictError
from scripts.data.pmlb import load_and_preprocess
from scripts.training.dt1 import train_dt1
from scripts.training.sklearn_dt import train_sklearn_dt

# Suppress DT1's consistency check warnings for cleaner output
warnings.filterwarnings("ignore", module="dt1")


def compare_dt1_vs_sklearn(
    dataset_name: str,
    sampling_rates: list[float],
    *,
    n_runs: int = 3,
    train_size: float = 0.8,
    timeout: float = 60.0,
    random_state: int = 42,
    min_feature_freq: float = 0.05,
    n_features: int | None = None,
    sklearn_max_depth: int | None = None,
) -> pd.DataFrame:
    """
    Compare DT1 with sklearn DecisionTreeClassifier on same data.

    Args:
        dataset_name: PMLB dataset name
        sampling_rates: List of sampling fractions to test
        n_runs: Number of runs per sampling rate (for averaging)
        train_size: Fraction for training split
        timeout: Timeout per tree size in seconds
        random_state: Random seed for reproducibility
        min_feature_freq: Minimum feature frequency for filtering
        n_features: Number of features to select (None = no selection)
        sklearn_max_depth: Max depth for sklearn tree (None = unlimited)

    Returns:
        DataFrame with comparison metrics
    """
    print(f"Loading dataset: {dataset_name}")
    X_full, y_full = load_and_preprocess(
        dataset_name,
        random_state=random_state,
        min_feature_freq=min_feature_freq,
        feature_selection="kbest" if n_features else None,
        n_features=n_features,
    )

    n_full = len(X_full)
    n_features = X_full.shape[1]
    print(f"Full dataset: {n_full} samples, {n_features} features")
    print(f"Label distribution: {np.sum(y_full)} positive, {np.sum(~y_full)} negative")
    print()

    # Split into train/test once
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, train_size=train_size, random_state=random_state
    )
    print(f"Train set: {len(X_train_full)} samples | Test set: {len(X_test)} samples")
    print(f"DT1 timeout: {timeout}s | Sklearn max_depth: {sklearn_max_depth}")
    print("=" * 110)

    # Table header
    header = (
        f"{'r':>4} | {'Samples':>7} | {'DT1 Size':>8} | {'DT1 Test':>9} | "
        f"{'DT1 Time':>9} | {'Sklearn Size':>9} | {'Sklearn Test':>10} | {'Sklearn Time':>10}"
    )
    print(header)
    print("-" * 110)

    results: list[dict[str, Any]] = []
    overall_start = time.time()

    for r in sampling_rates:
        # Sample from training set
        n_samples = int(len(X_train_full) * r)
        if n_samples < 2:
            continue

        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_train_full), size=n_samples, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]

        # Collect DT1 metrics from multiple runs
        dt1_runs: list[dict[str, Any]] = []
        for run in range(n_runs):
            run_start = time.time()
            try:
                dt1_result = train_dt1(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    timeout=timeout,
                )
                dt1_runs.append(
                    {
                        "tree_size": dt1_result.n_nodes,
                        "test_acc": dt1_result.test_accuracy,
                        "elapsed": dt1_result.elapsed,
                        "status": "OK",
                    }
                )
            except (UpperBoundTooStrictError, Exception) as e:
                dt1_runs.append(
                    {
                        "tree_size": None,
                        "test_acc": None,
                        "elapsed": time.time() - run_start,
                        "status": f"ERR: {type(e).__name__}",
                    }
                )

        # Collect sklearn metrics (single run, deterministic)
        sklearn_start = time.time()
        sklearn_result = train_sklearn_dt(
            X_train,
            y_train,
            X_test,
            y_test,
            max_depth=sklearn_max_depth,
            random_state=random_state,
        )
        sklearn_elapsed = time.time() - sklearn_start

        # Average DT1 runs
        dt1_ok = [r for r in dt1_runs if r["status"] == "OK"]
        if dt1_ok:
            avg_dt1_size = np.mean([r["tree_size"] for r in dt1_ok])
            avg_dt1_test = np.mean([r["test_acc"] for r in dt1_ok])
            avg_dt1_time = np.mean([r["elapsed"] for r in dt1_ok])
            dt1_status = "OK"
        else:
            avg_dt1_size = None
            avg_dt1_test = None
            avg_dt1_time = None
            dt1_status = dt1_runs[0]["status"]

        row = {
            "r": r,
            "samples": len(X_train),
            "dt1_size": avg_dt1_size,
            "dt1_test_acc": avg_dt1_test,
            "dt1_time": avg_dt1_time,
            "sklearn_size": sklearn_result.n_nodes,
            "sklearn_test_acc": sklearn_result.test_accuracy,
            "sklearn_time": sklearn_elapsed,
            "dt1_status": dt1_status,
        }
        results.append(row)

        # Print row
        dt1_size_str = f"{avg_dt1_size:.0f}" if avg_dt1_size is not None else "-"
        dt1_test_str = f"{avg_dt1_test:.2%}" if avg_dt1_test is not None else "-"
        dt1_time_str = f"{avg_dt1_time:.1f}" if avg_dt1_time is not None else "-"
        sklearn_size_str = f"{sklearn_result.n_nodes}"
        sklearn_test_str = f"{sklearn_result.test_accuracy:.2%}"
        sklearn_time_str = f"{sklearn_elapsed:.2f}"

        print(
            f"{r:>4.2f} | {len(X_train):>7} | {dt1_size_str:>8} | {dt1_test_str:>9} | "
            f"{dt1_time_str:>9}s | {sklearn_size_str:>9} | {sklearn_test_str:>10} | {sklearn_time_str:>10}s | {dt1_status}"
        )

    print("=" * 110)
    print(f"Total time: {time.time() - overall_start:.1f}s")

    return pd.DataFrame(results)
