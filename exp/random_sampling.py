#!/usr/bin/env python3
"""Random sampling experiment comparing CART and DT1 classifiers.

Data Pipeline per Run:
    1. Load dataset from PMLB (e.g., 'cars')
    2. Preprocess:
       - One-hot encode categorical features
       - Binarize labels (one-vs-rest, minority class = positive)
       - Filter features by frequency (0.05 <= freq <= 0.95)
       - Ensure consistency (remove conflicting samples)
    3. Sample training subset at rate r with unique random state per run
    4. Test set = remaining samples (not sampled)
    5. Train CART (sklearn DecisionTreeClassifier)
    6. Train DT1 with timeout, assert 100% training accuracy
    7. Collect metrics

Metrics per Run (CSV columns):
    - dataset, rate, run, samples, features
    - cart_size, cart_acc
    - dt1_size, dt1_acc, dt1_time, status

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
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.data.preprocessing import filter_by_frequency, make_consistent
from scripts.training.dt1 import train_dt1
from scripts.training.sklearn_dt import train_sklearn_dt
from scripts.table import print_batch_row

warnings.filterwarnings("ignore", module="dt1")

# Global verbosity flag
VERBOSE = False


def vprint(*args, **kwargs) -> None:
    """Print with [VERBOSE] prefix if verbose mode is enabled."""
    if VERBOSE:
        print("  [VERBOSE]", *args, **kwargs)


DEFAULT_DATASETS = "cars"
DEFAULT_RATES = "0.01,0.02,0.05,0.10,0.15,0.20,0.25"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 600
DEV_TIMEOUT = 5


def default_output_path() -> str:
    """Generate default output path with ISO 8601 timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"output/{timestamp}.csv"


def parse_rates(rates_str: str | None, range_str: str | None) -> list[float]:
    """Parse sampling rates from comma-separated string or range string."""
    if range_str is not None:
        parts = range_str.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid rates range: {range_str}. Use start:end:step format."
            )
        start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
        rates = []
        r = start
        while r <= end + 1e-9:
            rates.append(round(r, 4))
            r += step
        return rates

    if rates_str is None:
        return [float(r.strip()) for r in DEFAULT_RATES.split(",")]
    return [float(r.strip()) for r in rates_str.split(",")]


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
        "--rates-range",
        type=str,
        default=None,
        help="Sampling rates as start:end:step (e.g., 0.01:0.10:0.01)",
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
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
    dataset: str,
    rate: float,
    run_idx: int,
    timeout: int,
    n_features: int,
) -> dict[str, Any]:
    """Run a single experiment with full data pipeline per run."""
    # Step 1: Load raw data
    from scripts.data.pmlb import load_dataset

    X_raw, y_raw = load_dataset(dataset)

    # Step 2: Preprocess (one-hot encode, binarize, frequency filter on whole dataset)
    from scripts.data.pmlb import preprocess

    X_full, y_full = preprocess(
        X_raw,
        y_raw,
        encode_features="onehot",
        binarize_labels="threshold",
        min_feature_freq=0.05,  # Filter on whole dataset
        feature_selection=None,
        ensure_consistent=False,
    )

    # Step 3: Sample at rate r (unique random state per run)
    n_samples = max(2, int(len(X_full) * rate))
    rng = np.random.default_rng(run_idx)
    indices = rng.choice(len(X_full), size=n_samples, replace=False)

    # Training set = sampled data, test set = rest
    X_train_full = X_full[indices]
    y_train_full = y_full[indices]
    test_mask = np.ones(len(X_full), dtype=bool)
    test_mask[indices] = False
    X_test_full = X_full[test_mask]
    y_test_full = y_full[test_mask]

    # Step 4: Ensure consistency on training set (last step, clean data for solver)
    X_train, y_train, feature_indices = preprocess_dataset(X_train_full, y_train_full)

    # Test set: same feature filtering as training, but NOT made consistent
    # (test set should reflect real-world distribution, not filtered)
    X_test = X_test_full[:, feature_indices]
    y_test = y_test_full

    samples = len(X_train)

    # Step 5: Train CART (same training set as DT1)
    cart_result = train_sklearn_dt(X_train, y_train, X_test, y_test)
    cart_size = cart_result.n_nodes
    cart_acc = cart_result.test_accuracy

    # Step 6: Train DT1 with timeout, assert 100% training accuracy
    dt1_status = "OK"
    dt1_size: int | None = None
    dt1_acc: float | None = None
    dt1_time: float | None = None
    dt1_optimal: bool = False

    try:
        from dt1 import TimeoutBehavior

        dt1_result = train_dt1(
            X_train,
            y_train,
            X_test,
            y_test,
            timeout=timeout,
            timeout_behavior=TimeoutBehavior.RETURN_TREE,
        )
        dt1_time = dt1_result.elapsed

        # Check if we had an optimal build (no timeout in any size attempt)
        timings = dt1_result.build_result.timings
        if timings:
            last_timing = timings[-1]
            dt1_optimal = last_timing.status not in ("TIMEOUT", "ERROR")

        assert dt1_result.train_accuracy == 1.0, (
            f"DT1 train accuracy is {dt1_result.train_accuracy}, expected 1.0"
        )

        dt1_size = dt1_result.n_nodes
        dt1_acc = dt1_result.test_accuracy
    except AssertionError as e:
        raise AssertionError(f"DT1 train accuracy: {e}")
    except Exception as e:
        raise RuntimeError(f"DT1 error: {type(e).__name__}: {e}")

    cart_acc_rounded = round(cart_acc, 4) if cart_acc is not None else None
    dt1_acc_rounded = round(dt1_acc, 4) if dt1_acc is not None else None
    dt1_time_rounded = round(dt1_time, 4) if dt1_time is not None else None

    return {
        "samples": samples,
        "features": n_features,  # Per-dataset statistic
        "cart_size": cart_size,
        "cart_acc": cart_acc_rounded,
        "dt1_size": dt1_size,
        "dt1_acc": dt1_acc_rounded,
        "dt1_time": dt1_time_rounded,
        "dt1_optimal": dt1_optimal,
        "status": dt1_status,
    }


def aggregate_batch(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run results into batch statistics.

    Averages are computed on runs with valid trees (dt1_size is not None).
    Metrics:
    - n_total: total runs in batch
    - n_optimal: runs that completed all sizes without timeout
    - n_fail: runs that never found a valid tree
    - dt1_size, dt1_acc, dt1_time: averages over runs with valid trees
    """
    agg: dict[str, Any] = {}

    n_total = len(runs)
    agg["n_total"] = n_total

    # Runs with valid trees (returned a tree, regardless of optimality)
    has_tree = [r for r in runs if r["dt1_size"] is not None]

    # Count optimal runs (completed all sizes without timeout)
    n_optimal = sum(1 for r in has_tree if r.get("dt1_optimal", False))
    agg["n_optimal"] = n_optimal

    # Count failed runs (never got a valid tree)
    n_fail = sum(1 for r in runs if r["dt1_size"] is None)
    agg["n_fail"] = n_fail

    # Average metrics over runs with valid trees
    if has_tree:
        agg["samples"] = int(np.mean([r["samples"] for r in has_tree]))
        agg["features"] = runs[0]["features"]
        agg["cart_size"] = float(np.mean([r["cart_size"] for r in has_tree]))
        agg["cart_acc"] = round(float(np.mean([r["cart_acc"] for r in has_tree])), 4)
        agg["dt1_size"] = round(float(np.mean([r["dt1_size"] for r in has_tree])), 1)
        agg["dt1_acc"] = round(float(np.mean([r["dt1_acc"] for r in has_tree])), 4)
        agg["dt1_time"] = round(float(np.mean([r["dt1_time"] for r in has_tree])), 4)
    else:
        agg["samples"] = int(np.mean([r["samples"] for r in runs]))
        agg["features"] = runs[0]["features"]
        agg["cart_size"] = None
        agg["cart_acc"] = None
        agg["dt1_size"] = None
        agg["dt1_acc"] = None
        agg["dt1_time"] = None

    # Overall status
    if n_fail == n_total:
        agg["status"] = "ALL_FAIL"
    elif n_fail > 0:
        agg["status"] = "PARTIAL_FAIL"
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
        "cart_acc",
        "dt1_size",
        "dt1_acc",
        "dt1_time",
        "dt1_optimal",
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
        "n_total",
        "n_optimal",
        "n_fail",
        "samples",
        "features",
        "cart_size",
        "cart_acc",
        "dt1_size",
        "dt1_acc",
        "dt1_time",
        "status",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {path}")


def _run_single_worker(
    dataset: str,
    rate: float,
    run_idx: int,
    timeout: int,
    n_features: int,
) -> dict[str, Any]:
    """Worker function for parallel execution."""
    return run_single(dataset, rate, run_idx, timeout, n_features)


def run_batch_parallel(
    n_workers: int,
    dataset: str,
    rate: float,
    batch_size: int,
    timeout: int,
    n_features: int,
    n_full: int,
) -> list[dict[str, Any]]:
    """Run a batch of experiments in waves with per-run timeout.

    Runs are split into waves of n_workers. Each wave runs in parallel.
    Use wait() with timeout to ensure all workers complete or timeout together.
    """
    results: list[dict[str, Any]] = []

    # Expected samples per run (used for timeout/error results)
    expected_samples = max(2, int(n_full * rate))

    # Split runs into waves: [0,1,2,3], [4,5,6,7], [8,9]
    run_indices = list(range(batch_size))
    waves = [
        run_indices[i : i + n_workers] for i in range(0, len(run_indices), n_workers)
    ]

    vprint(f"batch_size={batch_size}, n_workers={n_workers}, timeout={timeout}")
    vprint(f"waves={len(waves)}: {waves}")

    for wave_idx, wave_runs in enumerate(waves):
        vprint(f"Starting wave {wave_idx + 1}/{len(waves)}: runs={wave_runs}")

        futures: list[Any] = []

        # Use a safeguard buffer to allow DT1's internal timeout to complete naturally
        SAFEGUARD_BUFFER = 60  # seconds
        safeguard_timeout = timeout + SAFEGUARD_BUFFER

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for run_idx in wave_runs:
                fut = executor.submit(
                    _run_single_worker, dataset, rate, run_idx, timeout, n_features
                )
                futures.append(fut)

            # Wait for ALL futures: either complete normally or safeguard timeout
            # wait() returns when all done OR timeout reached (whichever first)
            done, not_done = wait(
                futures, timeout=safeguard_timeout, return_when=ALL_COMPLETED
            )

            vprint(
                f"Wave {wave_idx + 1}: {len(done)} done, {len(not_done)} still running (safeguard: {safeguard_timeout}s)"
            )

        # Collect results: done futures have results, not_done are TIMEOUT (safeguard killed them)
        for run_idx, fut in enumerate(futures):
            if fut in done:
                try:
                    result = fut.result()
                    results.append(result)
                    vprint(
                        f"Run {run_idx}: {result.get('status')} (dt1_time={result.get('dt1_time')})"
                    )
                except Exception as e:
                    results.append(
                        {
                            "samples": expected_samples,
                            "features": n_features,
                            "cart_size": None,
                            "cart_acc": None,
                            "dt1_size": None,
                            "dt1_acc": None,
                            "dt1_time": None,
                            "status": f"ERROR: {type(e).__name__}",
                        }
                    )
                    vprint(f"Run {run_idx}: ERROR {type(e).__name__}")
            else:
                # Safeguard timeout - mark as TIMEOUT (thread was killed)
                results.append(
                    {
                        "samples": expected_samples,
                        "features": n_features,
                        "cart_size": None,
                        "cart_acc": None,
                        "dt1_size": None,
                        "dt1_acc": None,
                        "dt1_time": None,
                        "status": "TIMEOUT",
                    }
                )
                vprint(f"Run {run_idx}: TIMEOUT (safeguard killed thread)")

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

    # Load raw data once per dataset to get n_features
    from scripts.data.pmlb import load_dataset

    X_raw, y_raw = load_dataset(dataset)

    from scripts.data.pmlb import preprocess

    X_full, y_full = preprocess(
        X_raw,
        y_raw,
        encode_features="onehot",
        binarize_labels="threshold",
        min_feature_freq=0.05,  # Frequency filter on whole dataset
        feature_selection=None,
        ensure_consistent=False,
    )

    n_full = len(X_full)
    n_features = X_full.shape[1]

    print(f"Full: {n_full} samples, {n_features} features")

    # Compute column widths based on max possible samples (highest rate) for consistent table
    max_samples = max(2, int(n_full * rates[-1]))
    from scripts.table import compute_column_widths, print_header

    widths = compute_column_widths(max_samples, n_features)
    print_header(widths)

    all_runs: list[dict[str, Any]] = []
    prev_all_timeout = False

    for rate in rates:
        if prev_all_timeout:
            n_samples = max(2, int(n_full * rate))
            agg = {
                "samples": n_samples,
                "features": n_features,
                "cart_size": None,
                "cart_acc": None,
                "dt1_size": None,
                "dt1_acc": None,
                "dt1_time": None,
                "status": "SKIPPED",
            }
            for run_idx in range(batch_size):
                result = {
                    "dataset": dataset,
                    "rate": rate,
                    "run": run_idx + 1,
                    "samples": n_samples,
                    "features": n_features,
                    "cart_size": None,
                    "cart_acc": None,
                    "dt1_size": None,
                    "dt1_acc": None,
                    "dt1_time": None,
                    "status": "SKIPPED",
                }
                all_runs.append(result)
            print_batch_summary(rate, agg, widths)
            continue

        vprint(f"Starting batch: rate={rate}")

        runs = run_batch_parallel(
            n_workers,
            dataset,
            rate,
            batch_size,
            timeout,
            n_features,
            n_full,
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
                    "cart_acc": None,
                    "dt1_size": None,
                    "dt1_acc": None,
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
    rates = parse_rates(args.rates, args.rates_range)
    batch_size = 2 if args.dev else args.batch_size
    timeout = DEV_TIMEOUT if args.dev else args.timeout
    output_path = args.output if args.output is not None else default_output_path()
    per_run = args.per_run
    global VERBOSE
    VERBOSE = args.verbose
    n_workers = (
        args.worker_threads if args.worker_threads is not None else os.cpu_count()
    ) or 1

    print(f"Datasets: {', '.join(datasets)}")
    print(f"Rates: {rates}")
    print(f"Batch size: {batch_size}")
    print(f"Timeout per DT1 run: {timeout}s")
    print(f"Worker threads: {n_workers}")
    print(f"Output: {output_path}")
    print(f"Mode: {'per-run' if per_run else 'aggregated'}")
    print(f"Verbose: {VERBOSE}")

    for dataset in datasets:
        run_experiment(
            dataset,
            rates,
            batch_size,
            timeout,
            output_path,
            per_run,
            n_workers,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
