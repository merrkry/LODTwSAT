"""Sampling rate study experiment."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dt1.exceptions import UpperBoundTooStrictError
from scripts.data.pmlb import load_and_preprocess
from scripts.training.dt1 import DT1Result, train_dt1

# Suppress DT1's consistency check warnings for cleaner output
warnings.filterwarnings("ignore", module="dt1")


def run_sampling_experiment(
    dataset_name: str,
    sampling_rates: list[float],
    *,
    n_runs: int = 3,
    train_size: float = 0.8,
    timeout: float = 60.0,
    max_duration: float | None = None,
    output_path: str | None = None,
    random_state: int = 42,
    min_feature_freq: float = 0.05,
    n_features: int | None = None,
) -> pd.DataFrame:
    """
    Run sampling rate study for DT1 classifier.

    For each sampling rate, trains multiple times and averages metrics.

    Args:
        dataset_name: PMLB dataset name
        sampling_rates: List of sampling fractions to test
        n_runs: Number of runs per sampling rate (for averaging)
        train_size: Fraction for training split
        timeout: Timeout per tree size in seconds
        max_duration: Maximum total experiment duration in seconds
        output_path: Path to save CSV results
        random_state: Random seed for reproducibility
        min_feature_freq: Minimum feature frequency for filtering
        n_features: Number of features to select (None = no selection)

    Returns:
        DataFrame with columns:
        r, samples, features, tree_size, train_acc, test_acc, time_s
    """
    print(f"Loading dataset: {dataset_name}")
    X_full, y_full = load_and_preprocess(
        dataset_name,
        min_feature_freq=min_feature_freq,
        feature_selection="kbest" if n_features else None,
        n_features=n_features,
    )

    n_full = len(X_full)
    n_features = X_full.shape[1]
    print(f"Full dataset: {n_full} samples, {n_features} features")
    print(f"Label distribution: {np.sum(y_full)} positive, {np.sum(~y_full)} negative")
    print()

    # Split into train/test once (uses same split for all sampling rates)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, train_size=train_size, random_state=random_state
    )
    print(f"Train set: {len(X_train_full)} samples | Test set: {len(X_test)} samples")
    print(f"Timeout per tree: {timeout}s")
    print("=" * 90)

    # Table header
    header = (
        f"{'r':>4} | {'Samples':>7} | {'Features':>8} | "
        f"{'Tree Size':>10} | {'Train Acc':>9} | {'Test Acc':>9} | {'Time (s)':>9}"
    )
    print(header)
    print("-" * 90)

    results: list[dict[str, Any]] = []
    overall_start = time.time()

    for r in sampling_rates:
        # Check duration limit
        elapsed_total = time.time() - overall_start
        if max_duration and elapsed_total >= max_duration:
            print(f"\nDuration limit reached after {elapsed_total:.1f}s. Stopping.")
            break

        # Sample from training set
        n_samples = int(len(X_train_full) * r)
        if n_samples < 2:
            continue

        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_train_full), size=n_samples, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]

        # Collect metrics from multiple runs
        run_results: list[dict[str, Any]] = []
        for run in range(n_runs):
            run_start = time.time()
            try:
                result: DT1Result = train_dt1(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    timeout=timeout,
                )
                run_elapsed = time.time() - run_start
                run_results.append(
                    {
                        "tree_size": result.n_nodes,
                        "train_acc": result.train_accuracy,
                        "test_acc": result.test_accuracy,
                        "elapsed": result.elapsed,
                        "status": "OK",
                    }
                )
            except (UpperBoundTooStrictError, Exception) as e:
                run_elapsed = time.time() - run_start
                run_results.append(
                    {
                        "tree_size": None,
                        "train_acc": None,
                        "test_acc": None,
                        "elapsed": run_elapsed,
                        "status": f"ERR: {type(e).__name__}",
                    }
                )

        # Average successful runs
        ok_runs = [r for r in run_results if r["status"] == "OK"]
        if ok_runs:
            avg_tree_size = np.mean([r["tree_size"] for r in ok_runs])
            avg_train_acc = np.mean([r["train_acc"] for r in ok_runs])
            avg_test_acc = np.mean([r["test_acc"] for r in ok_runs])
            avg_elapsed = np.mean([r["elapsed"] for r in ok_runs])
            status = "OK"
        else:
            avg_tree_size = None
            avg_train_acc = None
            avg_test_acc = None
            avg_elapsed = None
            status = run_results[0]["status"]

        row = {
            "r": r,
            "samples": len(X_train),
            "features": X_train.shape[1],
            "tree_size": avg_tree_size,
            "train_acc": avg_train_acc,
            "test_acc": avg_test_acc,
            "time_s": avg_elapsed,
        }
        results.append(row)

        # Print row
        tree_str = f"{avg_tree_size:.0f}" if avg_tree_size is not None else "-"
        train_str = f"{avg_train_acc:.2%}" if avg_train_acc is not None else "-"
        test_str = f"{avg_test_acc:.2%}" if avg_test_acc is not None else "-"
        time_str = f"{avg_elapsed:.1f}" if avg_elapsed is not None else "-"

        print(
            f"{r:>4.2f} | {len(X_train):>7} | {X_train.shape[1]:>8} | "
            f"{tree_str:>10} | {train_str:>9} | {test_str:>9} | {time_str:>9}s | {status}"
        )

    print("=" * 90)
    print(f"Total time: {time.time() - overall_start:.1f}s")

    # Save to CSV
    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return df
