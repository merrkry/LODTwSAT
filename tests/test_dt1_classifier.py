import numpy as np
import pytest

from dt1 import DT1Classifier
from dt1.tree import DecisionTree
from dt1.exceptions import InvalidTrainingSetError


def is_consistent(features: np.ndarray, labels: np.ndarray) -> bool:
    """Check if training set is consistent: no duplicate features with different labels."""
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if np.array_equal(features[i], features[j]):
                if labels[i] != labels[j]:
                    return False
    return True


class TestDecisionTree:
    """Unit tests for DecisionTree class."""

    def test_leaf_prediction_single_node(self):
        """A single leaf node tree should return its label."""
        left = np.array([0, 0], dtype=np.int32)
        right = np.array([0, 0], dtype=np.int32)
        features = np.array([0, 0], dtype=np.int32)
        labels = np.array([-1, 1], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)
        result = tree.predict(np.array([True], dtype=bool))
        assert result is True

    def test_leaf_prediction_negative_label(self):
        """A single leaf node with label 0 should return False."""
        left = np.array([0, 0], dtype=np.int32)
        right = np.array([0, 0], dtype=np.int32)
        features = np.array([0, 0], dtype=np.int32)
        labels = np.array([-1, 0], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)
        result = tree.predict(np.array([False], dtype=bool))
        assert result is False

    def test_two_level_tree_left_branch(self):
        """Test tree with root and two leaves - go left on feature=0."""
        left = np.array([0, 2, 0, 0], dtype=np.int32)
        right = np.array([0, 3, 0, 0], dtype=np.int32)
        features = np.array([0, 1, 0, 0], dtype=np.int32)
        labels = np.array([-1, -1, 0, 1], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)
        result = tree.predict(np.array([False], dtype=bool))
        assert result is False

    def test_two_level_tree_right_branch(self):
        """Test tree with root and two leaves - go right on feature=1."""
        left = np.array([0, 2, 0, 0], dtype=np.int32)
        right = np.array([0, 3, 0, 0], dtype=np.int32)
        features = np.array([0, 1, 0, 0], dtype=np.int32)
        labels = np.array([-1, -1, 0, 1], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)
        result = tree.predict(np.array([True], dtype=bool))
        assert result is True

    def test_complex_tree_three_levels(self):
        """Test tree with 3 levels of decision nodes."""
        left = np.array([0, 2, 3, 0, 0, 6, 0, 0], dtype=np.int32)
        right = np.array([0, 5, 4, 0, 0, 7, 0, 0], dtype=np.int32)
        features = np.array([0, 1, 2, 0, 0, 2, 0, 0], dtype=np.int32)
        labels = np.array([-1, -1, -1, 0, 1, 1, 0, 0], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)

        assert tree.predict(np.array([False, False], dtype=bool)) is False
        assert tree.predict(np.array([False, True], dtype=bool)) is True
        assert tree.predict(np.array([True, False], dtype=bool)) is False
        assert tree.predict(np.array([True, True], dtype=bool)) is False


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

    def test_consistent_training_set_required(self, and_dataset):
        """Classifier should only accept consistent training sets."""
        features, labels = and_dataset
        assert is_consistent(features, labels), "Test setup error: dataset must be consistent"

    def test_trivial_dataset(self, trivial_dataset):
        """Single sample with single feature should always work."""
        features, labels = trivial_dataset

        clf = DT1Classifier(features, labels, max_size=3)
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

    def test_max_size_too_small_raises_error(self):
        """max_size < 3 should raise InvalidTrainingSetError."""
        features = np.array([[False], [True]], dtype=bool)
        labels = np.array([False, True], dtype=bool)

        with pytest.raises(InvalidTrainingSetError):
            DT1Classifier(features, labels, max_size=1)

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
        """Inconsistent dataset should eventually fail."""
        features = np.array([[False], [False]], dtype=bool)
        labels = np.array([True, False], dtype=bool)

        # An inconsistent dataset should not be learnable
        # (no decision tree can achieve 100% accuracy)
        with pytest.raises(InvalidTrainingSetError):
            DT1Classifier(features, labels, max_size=5)