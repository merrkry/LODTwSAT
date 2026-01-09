"""Training utilities for DT1 classifier."""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from dt1 import DT1Classifier, TimeoutBehavior


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
    timeout_behavior: TimeoutBehavior = TimeoutBehavior.ERROR,
) -> DT1Result:
    """
    Train DT1 classifier with timing and accuracy metrics.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features for evaluation
        y_test: Optional test labels for evaluation
        timeout: Total timeout in seconds (default: 60)
        max_size: Maximum tree size (None = auto-compute)
        verbose: Print progress information
        timeout_behavior: Behavior when timeout is reached (default: ERROR)

    Returns:
        DT1Result with classifier, accuracies, and timing

    Raises:
        UpperBoundTooStrictError: If timeout is reached with no valid tree
            and timeout_behavior is ERROR
    """
    start = time.time()
    clf = DT1Classifier(
        X_train,
        y_train,
        max_size=max_size,
        timeout=timeout,
        solver="cadical195",
        verbose=verbose,
        timeout_behavior=timeout_behavior,
    )
    elapsed = time.time() - start

    train_acc = float(np.mean(clf.predict(X_train) == y_train))
    n_nodes = clf._decision_tree.size

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
