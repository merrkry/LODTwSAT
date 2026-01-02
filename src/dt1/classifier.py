import numpy

from dt1.builder import build_dt1_classifier
from dt1.exceptions import InvalidTestSetError, InvalidTrainingSetError
from dt1.tree import DecisionTree
from dt1.types import FeatureMatrix, FeatureVector, LabelVector


class DT1Classifier:
    _n_samples: int
    _n_features: int
    _decision_tree: DecisionTree

    def __init__(
        self, features: FeatureMatrix, labels: LabelVector, max_size: int | None = None
    ) -> None:
        """
        Initialize and train a DT1 classifier.
        :param max_size: upper bound of decision tree size.
            If None, scikit's decision tree builder will be called for upper bound approximation.
        """
        if features.shape[0] != labels.shape[0]:
            raise InvalidTrainingSetError(
                "Inconsistent number of samples between features and labels."
            )

        self._n_samples = features.shape[0]
        self._n_features = features.shape[1]

        dt = build_dt1_classifier(features, labels, max_size)
        if dt is None:
            raise InvalidTrainingSetError(
                "Could not build a valid DT1 classifier with the given max_size."
            )
        self._decision_tree = dt

    def predict(self, features: FeatureMatrix) -> LabelVector:
        if features.shape[1] != self._n_features:
            raise InvalidTestSetError(
                "Number of features in test set does not match training set."
            )

        return numpy.array([self.predict_single(row) for row in features])

    def predict_single(self, features: FeatureVector) -> bool:
        if features.shape[0] != self._n_features:
            raise InvalidTestSetError(
                "Number of features in test sample does not match training set."
            )

        return self._decision_tree.predict(features)
