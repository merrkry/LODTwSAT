import numpy as np
import pytest

from dt1.types import FeatureMatrix, LabelVector


@pytest.fixture
def and_dataset() -> tuple[FeatureMatrix, LabelVector]:
    """AND truth table: only (1,1) -> True"""
    features = np.array(
        [[False, False], [False, True], [True, False], [True, True]],
        dtype=bool,
    )
    labels = np.array([False, False, False, True], dtype=bool)
    return features, labels


@pytest.fixture
def or_dataset() -> tuple[FeatureMatrix, LabelVector]:
    """OR truth table: only (0,0) -> False"""
    features = np.array(
        [[False, False], [False, True], [True, False], [True, True]],
        dtype=bool,
    )
    labels = np.array([False, True, True, True], dtype=bool)
    return features, labels


@pytest.fixture
def xor_dataset() -> tuple[FeatureMatrix, LabelVector]:
    """XOR truth table: (0,0) and (1,1) -> False, (0,1) and (1,0) -> True"""
    features = np.array(
        [[False, False], [False, True], [True, False], [True, True]],
        dtype=bool,
    )
    labels = np.array([False, True, True, False], dtype=bool)
    return features, labels


@pytest.fixture
def trivial_dataset() -> tuple[FeatureMatrix, LabelVector]:
    """Single sample, single feature - always satisfiable"""
    features = np.array([[False]], dtype=bool)
    labels = np.array([True], dtype=bool)
    return features, labels


@pytest.fixture
def all_same_label_dataset() -> tuple[FeatureMatrix, LabelVector]:
    """All samples have the same label"""
    features = np.array(
        [[False, False], [False, True], [True, False], [True, True]],
        dtype=bool,
    )
    labels = np.array([True, True, True, True], dtype=bool)
    return features, labels
