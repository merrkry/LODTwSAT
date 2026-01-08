#!/usr/bin/env python3
"""Random sampling experiment comparing CART and DT1 classifiers.

Data Pipeline per Run:
    1. Load dataset from PMLB (e.g., 'cars')
    2. Preprocess:
       - One-hot encode categorical features
       - Binarize labels (one-vs-rest, median class = positive)
       - Filter features by frequency (0.05 <= freq <= 0.95)
       - Ensure consistency (remove conflicting samples)
    3. Split: 80% train, 20% test (fixed random_state=42)
    4. Sample training subset at rate r with unique random state per run
    5. Train CART (sklearn DecisionTreeClassifier)
    6. Train DT1 with timeout, assert 100% training accuracy
    7. Collect metrics

Metrics per Run (CSV columns):
    - dataset, rate, run, samples, features
    - cart_size, cart_test_acc
    - dt1_size, dt1_test_acc, dt1_time, status

Aggregation per Batch (rate):
    - Average of all numeric metrics across runs
    - Count of timeouts and errors
    - Skip larger rates if all runs timeout

Usage:
    python exp/random_sampling.py --datasets cars --batch-size 10 --timeout 600
    python exp/random_sampling.py --datasets cars,heart --rates 0.05,0.10 --dev
    python exp/random_sampling.py --output custom.csv

Flags:
    --dev: Development mode (batch-size=2, timeout=5s)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from scripts.data.pmlb import load_and_preprocess
from scripts.data.preprocessing import filter_by_frequency, make_consistent
from scripts.training.dt1 import DT1Timeout, train_dt1
from scripts.training.sklearn_dt import train_sklearn_dt
from scripts.table import print_batch_row

warnings.filterwarnings("ignore", module="dt1")

DEFAULT_DATASETS = "cars"
DEFAULT_RATES = "0.01,0.02,0.05,0.10,0.15,0.20,0.25"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 600
DEV_TIMEOUT = 5


def default_output_path() -> str:
    """Generate default output path with ISO 8601 timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"output/{timestamp}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random sampling experiment: CART vs DT1 comparison"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=DEFAULT_DATASETS,
        help=f"Comma-separated dataset names (default: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default=DEFAULT_RATES,
        help=f"Comma-separated sampling rates (default: {DEFAULT_RATES})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of runs per (dataset, rate) combination (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"DT1 timeout per run in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: output/<timestamp>.csv)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=f"Use {DEV_TIMEOUT}s timeout for development",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        default=False,
        help="Output per-run CSV instead of aggregated per-batch",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=None,
        help="Number of worker threads for parallel execution (default: CPU cores)",
    )
    return parser.parse_args()


def preprocess_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    min_feature_freq: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter by frequency and ensure consistency. Returns features, labels, feature_indices."""
    features, feature_indices = filter_by_frequency(features, min_freq=min_feature_freq)
    features, labels = make_consistent(features, labels)
    return features, labels, feature_indices


def run_single(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rate: float,
    run_idx: int,
    timeout: int,
) -> dict[str, Any]:
    """Run a single experiment for one sampling rate and random state."""
    n_samples = max(2, int(len(X_train_full) * rate))
    rng = np.random.default_rng(run_idx)
    indices = rng.choice(len(X_train_full), size=n_samples, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]

    X_train, y_train, feature_indices = preprocess_dataset(X_train, y_train)
    X_test_filtered = X_test[:, feature_indices]

    samples = len(X_train)
    features = X_train.shape[1]

    cart_result = train_sklearn_dt(X_train, y_train, X_test_filtered, y_test)
    cart_size = cart_result.n_nodes
    cart_test_acc = cart_result.test_accuracy

    dt1_status = "OK"
    dt1_size: int | None = None
    dt1_test_acc: float | None = None
    dt1_time: float | None = None

    try:
        dt1_result = train_dt1(
            X_train, y_train, X_test_filtered, y_test, timeout=timeout
        )
        dt1_time = dt1_result.elapsed

        assert dt1_result.train_accuracy == 1.0, (
            f"DT1 train accuracy is {dt1_result.train_accuracy}, expected 1.0"
        )

        dt1_size = dt1_result.n_nodes
        dt1_test_acc = dt1_result.test_accuracy
    except DT1Timeout:
        dt1_status = "TIMEOUT"
        dt1_time = timeout
    except AssertionError as e:
        raise AssertionError(f"DT1 train accuracy: {e}")
    except Exception as e:
        raise RuntimeError(f"DT1 error: {type(e).__name__}: {e}")

    cart_test_acc_rounded = (
        round(cart_test_acc, 4) if cart_test_acc is not None else None
    )
    dt1_test_acc_rounded = round(dt1_test_acc, 4) if dt1_test_acc is not None else None
    dt1_time_rounded = round(dt1_time, 4) if dt1_time is not None else None

    return {
        "samples": samples,
        "features": features,
        "cart_size": cart_size,
        "cart_test_acc": cart_test_acc_rounded,
        "dt1_size": dt1_size,
        "dt1_test_acc": dt1_test_acc_rounded,
        "dt1_time": dt1_time_rounded,
        "status": dt1_status,
    }


def aggregate_batch(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run results into batch statistics."""
    agg: dict[str, Any] = {}

    samples_list = [r["samples"] for r in runs]
    features_list = [r["features"] for r in runs]
    cart_sizes = [r["cart_size"] for r in runs if r["cart_size"] is not None]
    cart_tests = [r["cart_test_acc"] for r in runs if r["cart_test_acc"] is not None]

    agg["samples"] = int(np.mean(samples_list))
    agg["features"] = int(np.mean(features_list))
    agg["cart_size"] = float(np.mean(cart_sizes)) if cart_sizes else None
    agg["cart_test_acc"] = round(float(np.mean(cart_tests)), 4) if cart_tests else None

    dt1_sizes = [r["dt1_size"] for r in runs if r["dt1_size"] is not None]
    dt1_tests = [r["dt1_test_acc"] for r in runs if r["dt1_test_acc"] is not None]
    dt1_times = [r["dt1_time"] for r in runs if r["dt1_time"] is not None]

    agg["dt1_size"] = round(float(np.mean(dt1_sizes)), 1) if dt1_sizes else None
    agg["dt1_test_acc"] = round(float(np.mean(dt1_tests)), 4) if dt1_tests else None
    agg["dt1_time"] = round(float(np.mean(dt1_times)), 4) if dt1_times else None

    n_timeouts = sum(1 for r in runs if r["status"] == "TIMEOUT")

    if n_timeouts == len(runs):
        agg["status"] = "ALL_TIMEOUT"
        agg["dt1_size"] = None
        agg["dt1_test_acc"] = None
        agg["dt1_time"] = None
    elif n_timeouts > 0:
        agg["status"] = "PARTIAL_TIMEOUT"
    else:
        agg["status"] = "OK"

    return agg


def print_batch_summary(
    rate: float, agg: dict[str, Any], widths: dict[str, int]
) -> None:
    """Print batch summary to stdout."""
    print_batch_row(rate, agg, widths)


def write_csv(output_path: str, results: list[dict[str, Any]]) -> None:
    """Write results to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "rate",
        "run",
        "samples",
        "features",
        "cart_size",
        "cart_test_acc",
        "dt1_size",
        "dt1_test_acc",
        "dt1_time",
        "status",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {path}")


def write_csv_aggregated(output_path: str, results: list[dict[str, Any]]) -> None:
    """Write aggregated batch results to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "rate",
        "samples",
        "features",
        "cart_size",
        "cart_test_acc",
        "dt1_size",
        "dt1_test_acc",
        "dt1_time",
        "status",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {path}")


def _run_single_worker(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rate: float,
    run_idx: int,
    timeout: int,
) -> dict[str, Any]:
    """Worker function for parallel execution."""
    return run_single(
        X_train_full, y_train_full, X_test, y_test, rate, run_idx, timeout
    )


def run_batch_parallel(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rate: float,
    batch_size: int,
    timeout: int,
    n_workers: int,
) -> list[dict[str, Any]]:
    """Run a batch of experiments in parallel using multiprocessing."""
    args_list = [
        (X_train_full, y_train_full, X_test, y_test, rate, run_idx, timeout)
        for run_idx in range(batch_size)
    ]

    pool = multiprocessing.Pool(processes=n_workers)
    async_result = pool.starmap_async(_run_single_worker, args_list)

    try:
        results = async_result.get()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise KeyboardInterrupt("Experiment interrupted by user")
    finally:
        pool.close()
        pool.join()

    return results


def run_experiment(
    dataset: str,
    rates: list[float],
    batch_size: int,
    timeout: int,
    output_path: str,
    per_run: bool = False,
    n_workers: int = 1,
) -> None:
    """Run the full experiment for one dataset."""
    print(f"\n{dataset}")
    print("-" * 80)

    from scripts.data.pmlb import preprocess

    X_full, y_full = preprocess(
        *load_and_preprocess(dataset),
        encode_features="onehot",
        binarize_labels="ovr",
        min_feature_freq=0.0,
        feature_selection=None,
        ensure_consistent=False,
    )

    n_full = len(X_full)
    n_train = int(n_full * 0.8)
    n_test = n_full - n_train
    n_features = X_full.shape[1]

    print(
        f"Full: {n_full} samples, {n_features} features | Train: {n_train} | Test: {n_test}"
    )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, train_size=0.8, random_state=42
    )

    from scripts.table import compute_column_widths, print_header

    widths = compute_column_widths(n_train, n_features)
    print_header(widths)

    all_runs: list[dict[str, Any]] = []
    prev_all_timeout = False

    for rate in rates:
        if prev_all_timeout:
            n_samples = max(2, int(len(X_train_full) * rate))
            agg = {
                "samples": n_samples,
                "features": X_train_full.shape[1],
                "cart_size": None,
                "cart_test_acc": None,
                "dt1_size": None,
                "dt1_test_acc": None,
                "dt1_time": None,
                "status": "SKIPPED",
            }
            for run_idx in range(batch_size):
                result = {
                    "dataset": dataset,
                    "rate": rate,
                    "run": run_idx + 1,
                    "samples": n_samples,
                    "features": X_train_full.shape[1],
                    "cart_size": None,
                    "cart_test_acc": None,
                    "dt1_size": None,
                    "dt1_test_acc": None,
                    "dt1_time": None,
                    "status": "SKIPPED",
                }
                all_runs.append(result)
            print_batch_summary(rate, agg, widths)
            continue

        runs = run_batch_parallel(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            rate,
            batch_size,
            timeout,
            n_workers,
        )
        for result in runs:
            result["dataset"] = dataset
            result["rate"] = rate
            all_runs.append(result)

        agg = aggregate_batch(runs)
        print_batch_summary(rate, agg, widths)

        prev_all_timeout = agg["status"] == "ALL_TIMEOUT"

    if per_run:
        write_csv(output_path, all_runs)
    else:
        aggregated = []
        for rate in rates:
            rate_runs = [r for r in all_runs if r["rate"] == rate]
            if not rate_runs:
                continue
            if rate_runs[0].get("status") == "SKIPPED":
                agg = {
                    "dataset": dataset,
                    "rate": rate,
                    "samples": rate_runs[0]["samples"],
                    "features": rate_runs[0]["features"],
                    "cart_size": None,
                    "cart_test_acc": None,
                    "dt1_size": None,
                    "dt1_test_acc": None,
                    "dt1_time": None,
                    "status": "SKIPPED",
                }
            else:
                agg = aggregate_batch(rate_runs)
                agg["dataset"] = dataset
                agg["rate"] = rate
            aggregated.append(agg)
        write_csv_aggregated(output_path, aggregated)


def main() -> None:
    args = parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    rates = [float(r.strip()) for r in args.rates.split(",")]
    batch_size = 2 if args.dev else args.batch_size
    timeout = DEV_TIMEOUT if args.dev else args.timeout
    output_path = args.output if args.output is not None else default_output_path()
    per_run = args.per_run
    n_workers = (
        args.worker_threads
        if args.worker_threads is not None
        else multiprocessing.cpu_count()
    )

    print(f"Datasets: {', '.join(datasets)}")
    print(f"Rates: {rates}")
    print(f"Batch size: {batch_size}")
    print(f"Timeout per DT1 run: {timeout}s")
    print(f"Worker threads: {n_workers}")
    print(f"Output: {output_path}")
    print(f"Mode: {'per-run' if per_run else 'aggregated'}")

    for dataset in datasets:
        run_experiment(
            dataset, rates, batch_size, timeout, output_path, per_run, n_workers
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
