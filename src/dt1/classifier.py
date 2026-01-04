import numpy

from dt1.builder import build_dt1_classifier
from dt1.exceptions import (
    InvalidTestSetError,
    InvalidTrainingSetError,
    UpperBoundTooStrictError,
)
from dt1.tree import DecisionTree
from dt1.types import FeatureMatrix, FeatureVector, LabelVector


def _check_consistent(features: FeatureMatrix, labels: LabelVector) -> None:
    """
    Verify the training set is consistent: no duplicate features with different labels.
    Raises InvalidTrainingSetError if inconsistent.

    Uses vectorized numpy operations for efficiency on larger datasets.
    """
    # Find unique rows and the inverse mapping (which original row maps to which unique)
    unique_features, inverse = numpy.unique(features, axis=0, return_inverse=True)

    # For each unique feature, check if all corresponding labels are the same
    unique_labels = numpy.array([labels[inverse == i] for i in range(len(unique_features))])

    for i, label_group in enumerate(unique_labels):
        unique_label_vals = numpy.unique(label_group)
        if len(unique_label_vals) > 1:
            feature_tuple = tuple(unique_features[i])
            raise InvalidTrainingSetError(
                f"Inconsistent training set: feature {feature_tuple} has conflicting labels."
            )


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
        :raises InvalidTrainingSetError: if features/labels mismatch or inconsistent dataset
        :raises UpperBoundTooStrictError: if user-provided max_size is too strict
        """
        if features.shape[0] != labels.shape[0]:
            raise InvalidTrainingSetError(
                "Inconsistent number of samples between features and labels."
            )

        _check_consistent(features, labels)

        self._n_samples = features.shape[0]
        self._n_features = features.shape[1]

        dt = build_dt1_classifier(features, labels, max_size)
        if dt is None:
            # User-provided max_size was too strict to build a valid tree
            raise UpperBoundTooStrictError(
                f"Could not build a valid DT1 classifier with max_size={max_size}."
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
