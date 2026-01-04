from dt1.types import FeatureVector, TreeNodeInfo


# Label encoding constants for decision tree nodes
NOTE_LABEL_NEGATIVE: int = -1
"""Negative label (predicts False)"""
NODE_LABEL_POSITIVE: int = 1
"""Positive label (predicts True)"""
NODE_LABEL_IRRELEVANT: int = 0
"""Internal node marker (not a leaf)"""


class DecisionTree:
    """
    Internally, all indices are 1-indexed.
    """

    left: TreeNodeInfo
    """
    ID of left child, 0 if not present
    """

    right: TreeNodeInfo
    """
    ID of right child, 0 if not present
    """

    features: TreeNodeInfo
    """
    Feature IDs (1-indexed) at internal nodes.
    0 for leaf nodes.
    """

    labels: TreeNodeInfo
    """
    Labels at leaf nodes.
    -1/1 for negative/positive labels, 0 for internal nodes.
    """

    def __init__(
        self,
        left: TreeNodeInfo,
        right: TreeNodeInfo,
        features: TreeNodeInfo,
        labels: TreeNodeInfo,
    ) -> None:
        self.left = left
        self.right = right
        self.features = features
        self.labels = labels

    def _is_leaf(self, id: int) -> bool:
        is_leaf: bool = self.features.item(id) == 0
        if is_leaf:
            assert (
                self.left.item(id) == 0
                and self.right.item(id) == 0
                and self.labels.item(id) != NODE_LABEL_IRRELEVANT
            )
        else:
            assert (
                self.left.item(id) != 0
                or self.right.item(id) != 0
                and self.labels.item(id) == NODE_LABEL_IRRELEVANT
            )
        return is_leaf

    def predict(self, features: FeatureVector) -> bool:
        id: int = 1
        while not self._is_leaf(id):
            assert id > 0
            feature: int = self.features.item(id)
            if features.item(feature - 1):  # True for right branch
                id = self.right.item(id)
            else:  # False for left branch
                id = self.left.item(id)
        assert self.labels.item(id) != NODE_LABEL_IRRELEVANT
        return self.labels.item(id) == NODE_LABEL_POSITIVE
