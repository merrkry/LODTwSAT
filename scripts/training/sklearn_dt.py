"""Training utilities for sklearn DecisionTreeClassifier."""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclasses.dataclass(frozen=True)
class SklearnDTResult:
    """Result from training sklearn DecisionTreeClassifier."""

    classifier: DecisionTreeClassifier
    train_accuracy: float
    test_accuracy: float | None
    n_nodes: int
    depth: int | None


def train_sklearn_dt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    max_depth: int | None = None,
    random_state: int | None = None,
    criterion: str = "gini",
) -> SklearnDTResult:
    """
    Train sklearn DecisionTreeClassifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features for evaluation
        y_test: Optional test labels for evaluation
        max_depth: Maximum tree depth (None = unlimited)
        random_state: Random seed for reproducibility
        criterion: Splitting criterion ("gini" or "entropy")

    Returns:
        SklearnDTResult with classifier, accuracies, and tree stats
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        criterion=criterion,
    )
    clf.fit(X_train, y_train)

    # Compute training accuracy
    train_acc = float(np.mean(clf.predict(X_train) == y_train))

    # Compute test accuracy if available
    test_acc: float | None = None
    if X_test is not None and y_test is not None:
        test_acc = float(np.mean(clf.predict(X_test) == y_test))

    return SklearnDTResult(
        classifier=clf,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_nodes=clf.tree_.node_count,
        depth=clf.get_depth(),
    )
