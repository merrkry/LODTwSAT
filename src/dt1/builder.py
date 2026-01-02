import typing
import sklearn.tree

from dt1.tree import DecisionTree
from dt1.types import FeatureMatrix, LabelVector


def build_dt1_classifier(
    features: FeatureMatrix, labels: LabelVector, max_size: int | None
) -> DecisionTree | None:
    if max_size is None:
        cart_tree = sklearn.tree.DecisionTreeClassifier().fit(features, labels)
        max_size = cart_tree.tree_.node_count

    max_size = typing.cast(int, max_size)

    # We assume root is not a leaf
    if max_size < 3:
        return None

    for size in range(3, max_size + 1, -1):
        pass
