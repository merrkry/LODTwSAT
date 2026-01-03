import numpy as np
import pytest

from dt1.tree import DecisionTree


class TestDecisionTree:
    """Unit tests for DecisionTree class."""

    def test_leaf_prediction_single_node(self):
        """A single leaf node tree should return its label."""
        # Arrays must be size 2 (1-indexed, index 0 is dummy)
        left = np.array([0, 0], dtype=np.int32)
        right = np.array([0, 0], dtype=np.int32)
        features = np.array([0, 0], dtype=np.int32)
        labels = np.array([-1, 1], dtype=np.int32)  # index 1 is the leaf

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)
        # Any feature vector should return the leaf label
        result = tree.predict(np.array([True], dtype=bool))
        assert result is True

    def test_leaf_prediction_negative_label(self):
        """A single leaf node with label 0 should return False."""
        left = np.array([0, 0], dtype=np.int32)
        right = np.array([0, 0], dtype=np.int32)
        features = np.array([0, 0], dtype=np.int32)
        labels = np.array([-1, 0], dtype=np.int32)  # index 1 is the leaf

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)
        result = tree.predict(np.array([False], dtype=bool))
        assert result is False

    def test_two_level_tree_left_branch(self):
        """Test tree with root and two leaves - go left on feature=0."""
        # Node 1 (root) splits to node 2 (left) and node 3 (right)
        # Arrays must be size 4 (indices 0-3)
        left = np.array([0, 2, 0, 0], dtype=np.int32)  # node 1 -> node 2
        right = np.array([0, 3, 0, 0], dtype=np.int32)  # node 1 -> node 3
        features = np.array([0, 1, 0, 0], dtype=np.int32)  # node 1 uses feature 1
        labels = np.array([-1, -1, 0, 1], dtype=np.int32)  # node 2: 0, node 3: 1

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)

        # Feature 0 -> go left -> should get label 0
        result = tree.predict(np.array([False], dtype=bool))
        assert result is False

    def test_two_level_tree_right_branch(self):
        """Test tree with root and two leaves - go right on feature=1."""
        left = np.array([0, 2, 0, 0], dtype=np.int32)
        right = np.array([0, 3, 0, 0], dtype=np.int32)
        features = np.array([0, 1, 0, 0], dtype=np.int32)
        labels = np.array([-1, -1, 0, 1], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)

        # Feature 1 -> go right -> should get label 1
        result = tree.predict(np.array([True], dtype=bool))
        assert result is True

    def test_complex_tree_three_levels(self):
        """Test tree with 3 levels of decision nodes."""
        # Tree structure:
        #       1 (feature 1)
        #      / \
        #     2   5 (feature 2)
        #    / \ / \
        #   3  4 6  7
        # Arrays must be size 8 (indices 0-7)
        left = np.array([0, 2, 3, 0, 0, 6, 0, 0], dtype=np.int32)
        right = np.array([0, 5, 4, 0, 0, 7, 0, 0], dtype=np.int32)
        features = np.array([0, 1, 2, 0, 0, 2, 0, 0], dtype=np.int32)
        labels = np.array([-1, -1, -1, 0, 1, 1, 0, 0], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)

        # Feature 1 = 0, Feature 2 = 0 -> node 3 -> label 0
        result = tree.predict(np.array([False, False], dtype=bool))
        assert result is False

        # Feature 1 = 0, Feature 2 = 1 -> node 4 -> label 1
        result = tree.predict(np.array([False, True], dtype=bool))
        assert result is True

        # Feature 1 = 1, Feature 2 = 0 -> node 6 -> label 0
        result = tree.predict(np.array([True, False], dtype=bool))
        assert result is False

        # Feature 1 = 1, Feature 2 = 1 -> node 7 -> label 0
        result = tree.predict(np.array([True, True], dtype=bool))
        assert result is False

    def test_right_branch_selection(self):
        """Verify that feature=True selects right child."""
        left = np.array([0, 2, 0, 0], dtype=np.int32)
        right = np.array([0, 3, 0, 0], dtype=np.int32)
        features = np.array([0, 1, 0, 0], dtype=np.int32)
        labels = np.array([-1, -1, 0, 1], dtype=np.int32)

        tree = DecisionTree(left=left, right=right, features=features, labels=labels)

        # Both features True should go right
        result = tree.predict(np.array([True], dtype=bool))
        assert result is True
