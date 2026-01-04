import numpy as np
import pytest

from dt1 import DT1Classifier
from dt1.tree import (
    DecisionTree,
    NODE_LABEL_IRRELEVANT,
    NODE_LABEL_POSITIVE,
    NOTE_LABEL_NEGATIVE,
)
from dt1.exceptions import InvalidTrainingSetError, UpperBoundTooStrictError


def is_consistent(features: np.ndarray, labels: np.ndarray) -> bool:
    """Check if training set is consistent: no duplicate features with different labels."""
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if np.array_equal(features[i], features[j]):
                if labels[i] != labels[j]:
                    return False
    return True


class TestConsistencyCheck:
    """Tests for training set consistency verification."""

    def test_consistent_dataset(self):
        """AND dataset is consistent."""
        features = np.array(
            [[False, False], [False, True], [True, False], [True, True]],
            dtype=bool,
        )
        labels = np.array([False, False, False, True], dtype=bool)
        assert is_consistent(features, labels)

    def test_inconsistent_dataset(self):
        """Same features with different labels is inconsistent."""
        features = np.array([[False], [False]], dtype=bool)
        labels = np.array([True, False], dtype=bool)
        assert not is_consistent(features, labels)

    def test_inconsistent_complex(self):
        """Duplicate features with different labels detected."""
        features = np.array([[True, False], [True, False], [False, True]], dtype=bool)
        labels = np.array([True, False, True], dtype=bool)
        assert not is_consistent(features, labels)


class TestDT1Classifier:
    """Integration tests for DT1Classifier."""

    def test_trivial_dataset(self, trivial_dataset):
        """Single sample with single feature should always work."""
        features, labels = trivial_dataset

        clf = DT1Classifier(features, labels, 3)
        predictions = clf.predict(features)

        assert predictions.shape == (1,)
        assert predictions[0] == labels[0]

    def test_simple_and_dataset(self, and_dataset):
        """AND truth table should be learnable with 100% accuracy."""
        features, labels = and_dataset

        clf = DT1Classifier(features, labels)
        predictions = clf.predict(features)

        assert np.array_equal(predictions, labels)

    def test_simple_or_dataset(self, or_dataset):
        """OR truth table should be learnable with 100% accuracy."""
        features, labels = or_dataset

        clf = DT1Classifier(features, labels)
        predictions = clf.predict(features)

        assert np.array_equal(predictions, labels)

    def test_all_same_label(self, all_same_label_dataset):
        """Dataset with all same labels should be trivially learnable."""
        features, labels = all_same_label_dataset

        clf = DT1Classifier(features, labels, max_size=5)
        predictions = clf.predict(features)

        assert np.all(predictions)
        assert np.array_equal(predictions, labels)

    def test_100_percent_accuracy(self, and_dataset):
        """Verify the classifier achieves 100% training accuracy."""
        features, labels = and_dataset

        clf = DT1Classifier(features, labels)
        predictions = clf.predict(features)

        accuracy = np.sum(predictions == labels) / len(labels)
        assert accuracy == 1.0

    def test_prediction_shape(self, and_dataset):
        """Predictions should have correct shape."""
        features, labels = and_dataset

        clf = DT1Classifier(features, labels)
        predictions = clf.predict(features)

        assert predictions.shape == (4,)

    def test_single_prediction(self, and_dataset):
        """predict_single should return a single boolean."""
        features, labels = and_dataset

        clf = DT1Classifier(features, labels)

        for i in range(len(features)):
            result = clf.predict_single(features[i])
            assert isinstance(result, bool)
            assert result == labels[i]

    def test_consistency_enforced(self):
        """Inconsistent dataset should fail fast with InvalidTrainingSetError."""
        features = np.array([[False], [False]], dtype=bool)
        labels = np.array([True, False], dtype=bool)

        # Inconsistent dataset (same feature with different labels) should fail immediately
        # during validation, before attempting SAT solving
        with pytest.raises(InvalidTrainingSetError):
            DT1Classifier(features, labels, max_size=5)
