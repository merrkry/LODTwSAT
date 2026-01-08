"""Training utilities for DT1 classifier."""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import numpy as np

from dt1 import DT1Classifier


class DT1Timeout(Exception):
    """Raised when DT1 training exceeds timeout."""

    pass


def _train_dt1_worker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_size: int | None,
    timeout: float,
    verbose: bool,
    result: dict[str, Any],
) -> None:
    """Worker function for DT1 training."""
    try:
        start = time.time()
        clf = DT1Classifier(
            X_train,
            y_train,
            max_size=max_size,
            timeout=timeout,
            verbose=verbose,
        )
        elapsed = time.time() - start
        train_acc = float(np.mean(clf.predict(X_train) == y_train))
        tree = clf._decision_tree
        n_nodes = int(np.sum(tree.labels != 0))
        result["ok"] = {
            "classifier": clf,
            "train_accuracy": train_acc,
            "elapsed": elapsed,
            "n_nodes": n_nodes,
        }
    except Exception as e:
        result["error"] = e


@dataclasses.dataclass(frozen=True)
class DT1Result:
    """Result from training DT1 classifier."""

    classifier: DT1Classifier
    train_accuracy: float
    test_accuracy: float | None
    elapsed: float
    n_nodes: int

    @property
    def build_result(self):
        """Access the build result with timing information."""
        return self.classifier.build_result


def train_dt1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    timeout: float = 60.0,
    max_size: int | None = None,
    verbose: bool = False,
) -> DT1Result:
    """
    Train DT1 classifier with timing and accuracy metrics.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features for evaluation
        y_test: Optional test labels for evaluation
        timeout: Timeout for SAT solver in seconds (per tree size attempt)
        max_size: Maximum tree size (None = auto-compute)
        verbose: Print progress information

    Returns:
        DT1Result with classifier, accuracies, and timing

    Raises:
        DT1Timeout: If training exceeds overall timeout
    """
    result: dict[str, Any] = {}
    thread = threading.Thread(
        target=_train_dt1_worker,
        args=(X_train, y_train, max_size, timeout, verbose, result),
    )
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise DT1Timeout("DT1 training timed out")

    if "error" in result:
        raise result["error"]

    clf = result["ok"]["classifier"]
    train_acc = result["ok"]["train_accuracy"]
    elapsed = result["ok"]["elapsed"]
    n_nodes = result["ok"]["n_nodes"]

    test_acc: float | None = None
    if X_test is not None and y_test is not None:
        test_acc = float(np.mean(clf.predict(X_test) == y_test))

    return DT1Result(
        classifier=clf,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        elapsed=elapsed,
        n_nodes=n_nodes,
    )
